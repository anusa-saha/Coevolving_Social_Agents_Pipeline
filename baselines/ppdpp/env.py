import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# fastchat serves only the vicuna and llama2 backends and is unavailable on some
# platforms; openai only the API backend. Import both lazily so neither is required
# for a run that does not use it.
try:
    from fastchat.model import load_model, get_conversation_template
except ImportError:
    load_model = get_conversation_template = None
try:
    import openai
except ImportError:
    openai = None

from utils import *
from prompt import *
_CSA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CSA_ROOT not in sys.path:
    sys.path.insert(0, _CSA_ROOT)
from csa_core.verifier import score, floor_score
#from unidecode import unidecode
import nltk
import re
import time
import json

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer', 'csa': 'Chair'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller', 'csa': 'Advisor'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages,
                  'csa': CSAMessages}

LOCAL_BACKENDS = ('vicuna', 'llama2', 'qwen')

# Never hardcode a key here: this file is git-tracked. Set OPENAI_API_KEY in the
# environment instead.
YOUR_API_KEY = os.environ.get('OPENAI_API_KEY', '')

_STOP = set('a an the is are was were be been being of to in on at for with and or but '
            'that this these those it its as by from we you i they he she them us our your '
            'has have had do does did not no yes will would can could should may might'.split())


def _content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9%$./-]+", text.lower())
            if t not in _STOP and len(t) > 1]


def _overlap(fact_text, utterance):
    ftok = set(_content_tokens(fact_text))
    if not ftok:
        return 0.0
    return len(ftok & set(_content_tokens(utterance))) / len(ftok)

class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        # vicuna_model/vicuna_tokenizer is the locally served backend whatever its
        # family; the test env reuses the train env's weights rather than reloading.
        backends = {args.system, args.user, args.critic}
        if backends & set(LOCAL_BACKENDS):
            if mode == 'train':
                if 'qwen' in backends:
                    self.vicuna_tokenizer = AutoTokenizer.from_pretrained(args.qwen_path)
                    self.vicuna_model = AutoModelForCausalLM.from_pretrained(
                        args.qwen_path, dtype=getattr(torch, args.qwen_dtype),
                        device_map=args.qwen_device_map)
                    self.vicuna_model.eval()
                    if self.vicuna_tokenizer.pad_token_id is None:
                        self.vicuna_tokenizer.pad_token = self.vicuna_tokenizer.eos_token
                else:
                    if load_model is None:
                        raise ImportError('fastchat is required for vicuna/llama2: '
                                          'pip install fschat')
                    self.vicuna_model, self.vicuna_tokenizer = load_model(
                        args.model_path,
                        args.device,
                        args.num_gpus,
                        args.max_gpu_memory,
                        args.load_8bit,
                        args.cpu_offloading,
                        debug=args.debug,
                    )
            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode

        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },
            'csa': {
                'not reached': -1.0,
                'ignores': -0.5,
                'some of': 0.5,
                'all of the decisive': 1.0,
            },
        }

        set_random_seed(args.seed)

        
    def reset(self):
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Patient", "content":self.case['situation']}]
        elif self.args.data_name == 'cima':
            self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, {"role":"Student", "content":self.case['dialog'][1]['text']}]
        elif self.args.data_name == 'cb':
            self.conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % self.case['item_name']}, {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (self.case['item_name'], self.case['seller_price'])}]
        elif self.args.data_name == 'csa':
            self._csa_reset()
        print(self.conversation)
        return self.conversation

    # ------------------------------------------------------------------ csa
    def _csa_reset(self):
        """Per-case turn budget, turn-order cursor and disclosure bookkeeping."""
        case = self.case
        self.dm = case['decision_maker']
        self.order = list(case['interaction_config']['turn_order'])
        self.order_ptr = 0
        self.utterances = 0
        # turn_cap counts utterances; the planner acts once per chair turn, so the
        # episode length in PPDPP's sense is the number of chair slots inside the cap.
        self.utterance_cap = int(case['interaction_config']['turn_cap'])
        dm_slots = sum(1 for i in range(self.utterance_cap)
                       if self.order[i % len(self.order)] == self.dm)
        self.max_turn = max(1, min(dm_slots, self.args.max_turn))

        self.names = {a['agent_id']: a['name'] for a in case['agents']}
        self.revealed = set()
        self.reveal_turn = {}
        self.reveal_elicited = {}     # fact_id -> True if a question preceded it
        self.addressed = set()        # advisors the chair has directed a question at
        self.leaks = []               # agent stated a fact it never saw
        self.settlement = {}
        self.settle_turn = None       # chair step the settlement came from
        self.settled_by = None        # 'chair' | 'extractor'
        self.last_score = None
        self.last_score_norm = None
        self.prev_phi = {'disclosure': 0.0, 'elicitation': 0.0, 'coverage': 0.0}
        self.last_act = None
        self.n_calls = 0              # per-episode cost accounting, see generate_response
        self.calls_by_role = {}
        self.prompt_chars = 0
        self.act_history = []         # for act distribution / bigram diagnostics
        # Every upstream dataset seeds an opening turn, and agent.build_input reads its
        # loop variable after the loop, so an empty transcript raises. The seed is the
        # shared framing every agent already holds, so it discloses nothing.
        self.conversation = [{"role": "Meeting",
                              "content": "The group convenes to decide: %s"
                                         % case['description']}]

    def step(self, action):
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        print(action)
        if self.args.data_name == 'csa':
            return self._csa_step(action)

        messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action)
        response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
        response = self.postprocess_response(response, user_role[self.args.data_name])
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        reward = self.compute_reward(self.args.critic, messages, self.case)

        if self.args.data_name == 'esc':
            if reward > 0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cima':
            if reward == 1:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
        elif self.args.data_name == 'cb':
            if reward >= 0:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
                
        self.cur_conver_step += 1
        return self.conversation, reward, done
    
    # ------------------------------------------------------------------ csa
    def _csa_step(self, action):
        """Advance the turn order until the chair has spoken once.

        PPDPP's contract is one planner action per system utterance. Here the system
        utterance is the chair's, and any advisors whose slots fall before it speak
        first, unsteered -- they are environment, exactly as the simulated user is
        upstream. A scenario whose order gives the chair several slots per cycle
        therefore yields several steps per cycle, one action each.
        """
        self.last_act = action
        self.act_history.append(action)
        while True:
            speaker = self.order[self.order_ptr]
            self.order_ptr = (self.order_ptr + 1) % len(self.order)
            self.utterances += 1

            if speaker == self.dm:
                msgs = CSAMessages(self.case, 'system', self.conversation, action)
                # A settlement is a JSON object with up to seven decision fields plus
                # commitments and fact lists. At the utterance budget it truncates
                # mid-object, the parse fails, and every check then fails with it -- so
                # the settling turn gets the settlement budget, not the utterance one.
                saved = self.args.max_new_tokens
                if action == CSA_SETTLING_ACT:
                    self.args.max_new_tokens = getattr(
                        self.args, 'csa_settlement_max_tokens', 512)
                try:
                    resp = self.generate_response(self.args.system, msgs,
                                                  self.names[speaker])
                finally:
                    self.args.max_new_tokens = saved
                # postprocess_response trims at sentence boundaries and drops a trailing
                # fragment that does not end in punctuation. That is right for prose and
                # destroys a JSON object, so a settlement turn is left intact.
                if not (action == CSA_SETTLING_ACT or self._csa_parse_json(resp)):
                    resp = self._csa_postprocess(resp, speaker)
                self.conversation.append({"role": self.names[speaker], "content": resp})
                print(self.conversation[-1])
                if action in CSA_ELICITING_ACTS:
                    self._csa_note_addressed(resp)
                break

            msgs = CSAMessages(self.case, 'user', self.conversation, agent_id=speaker)
            resp = self.generate_response(self.args.user, msgs, self.names[speaker])
            resp = self._csa_postprocess(resp, speaker)
            self.conversation.append({"role": self.names[speaker], "content": resp})
            print(self.conversation[-1])
            self._csa_note_disclosures(speaker, resp)
            self._csa_note_leaks(speaker, resp)

            if self.utterances >= self.utterance_cap:
                break

        last_turn = self.cur_conver_step == self.max_turn - 1
        if getattr(self.args, 'csa_reward', 'critic') == 'verifier':
            return self._csa_step_verifier(action, last_turn)
        return self._csa_step_critic(last_turn)

    def _csa_step_critic(self, last_turn):
        """Arm A: upstream's LLM critic, unchanged."""
        every = getattr(self.args, 'critic_every', 1)
        if every > 1 and not last_turn and (self.cur_conver_step + 1) % every:
            reward = 0.0
        else:
            msgs = CSAMessages(self.case, 'critic', self.conversation)
            reward = self.compute_reward(self.args.critic, msgs, self.case)

        # PPDPP sets the success threshold per dataset: esc fires on reward > 0.5,
        # cima on reward == 1. The cima rule is used here.
        if reward == 1:
            print('--> Goal completed !')
            done = 1
        elif last_turn:
            print('--> Maximum number of turns reached !')
            done = -1
        else:
            print('--> On-going !')
            done = 0

        # Measurement only; never enters `reward` in this arm. The settlement is taken
        # exactly as the verifier arm takes it: before this the critic arm never recorded
        # one, so every episode was scored against an empty settlement.
        self._csa_take_settlement(final=bool(done))
        if done:
            self._csa_finalise_score()
        self.cur_conver_step += 1
        return self.conversation, reward, done

    def _csa_step_verifier(self, action, last_turn):
        """Arm B: reward from the dataset's executable checks.

        Terminal value from the settlement; in between, potential-based shaping over
        disclosure, elicitation and coverage -- all free, since those detectors are
        lexical. Scoring is attempted on every chair turn that produced parseable JSON,
        so success is detectable at any turn rather than only on a settling turn.
        """
        parsed = self._csa_take_settlement(final=last_turn)
        scored = bool(parsed) or last_turn
        if scored:
            s = self._csa_finalise_score()
            terminal = self._csa_terminal_reward(s)
            # Success is a threshold on the decisive-check fraction, not a conjunction
            # over every check. `all_content and all_prov` requires 5-11 simultaneous
            # exact matches per case; at the measured per-check rates that is
            # unreachable, so it reported 0.0 regardless of policy. `joint` is still
            # scored and logged for the strict reading -- it no longer drives training.
            tau = getattr(self.args, 'csa_done_tau', 0.6)
            invalid = bool(self.leaks) and getattr(self.args, 'csa_leak_invalidates', 1)
            if s['schema_valid'] and not invalid and s['dca'] >= tau:
                print('--> Goal completed !')
                self.cur_conver_step += 1
                return self.conversation, terminal, 1
            if last_turn:
                print('--> Maximum number of turns reached !')
                self.cur_conver_step += 1
                return self.conversation, terminal, -1

        reward = -0.1 + self._csa_shaping()
        print('--> On-going ! %s' % self._csa_potentials())
        self.cur_conver_step += 1
        return self.conversation, reward, 0

    def _csa_take_settlement(self, final, ep=None):
        """Keep the chair's latest parseable settlement and, on the final turn with none,
        fall back to the extractor. Both reward arms call this, so both are scored alike.
        Returns what this turn's chair message parsed to ({} when nothing).

        `ep`: when given (batched rollout), read/write that episode's bookkeeping
        instead of self's. Defaults to self, so the single-episode call sites below are
        unchanged."""
        src = ep if ep is not None else self
        parsed = self._csa_parse_json(src.conversation[-1]['content'])
        if parsed:
            src.settlement = parsed
            src.settle_turn, src.settled_by = src.cur_conver_step, 'chair'
        if final and not src.settlement:
            src.settlement = self._csa_extract_settlement(ep=ep)
            if src.settlement:
                src.settle_turn, src.settled_by = src.cur_conver_step, 'extractor'
        return parsed

    def _csa_terminal_reward(self, s, ep=None):
        """Multi-dimensional reward on PPDPP's [-1, 1] ordinal ladder.

        Three graded axes measuring different things, plus an integrity gate:

          pool  -- did private information surface at all      (disclosure_rate)
          use   -- did surfaced information change the decision correctly  (dca)
          close -- competence on checks no decisive fact flips (close)

        pool and use are kept apart on purpose: a chair can elicit everything and
        still decide wrong, or decide right by luck without eliciting. dca alone
        conflates the two, and that distinction is what `flips` exists to measure.

        Integrity is a gate rather than a fourth summand, so a well-formed fabrication
        cannot outscore an honest partial answer, and the hallucination penalty is
        always subtractive -- scaling it would make crediting undisclosed facts
        *improve* a negative score.

        `coverage` is deliberately absent. It is satisfied by naming advisors -- the
        detector matches surnames as substrings, so a chair naming itself inflates it --
        and belongs in the shaping potential, where Ng et al. (1999) guarantees it
        cannot change the optimal policy. See _csa_shaping.
        """
        src = ep if ep is not None else self
        if not s['schema_valid']:
            return -1.0
        if bool(src.leaks) and getattr(self.args, 'csa_leak_invalidates', 1):
            return -1.0
        pool, use, close = s['disclosure_rate'], s['dca'], s['close']
        if pool == 0.0 and use == 0.0:
            return -0.5
        a = getattr(self.args, 'csa_w_use', 0.5)
        b = getattr(self.args, 'csa_w_pool', 0.3)
        g = getattr(self.args, 'csa_w_close', 0.2)
        tot = a + b + g
        raw = (a * use + b * pool + g * close) / (tot if tot else 1.0)
        r = 2.0 * raw - 1.0
        r -= getattr(self.args, 'csa_w_halluc_pen', 0.5) * s['hallucinated_credit']
        if getattr(self.args, 'csa_w_accept', 0.0) > 0:
            r += self.args.csa_w_accept * self._csa_acceptance(src.settlement)[0]
        return max(-1.0, min(1.0, r))

    def _csa_potentials(self, ep=None):
        """Monotone progress measures. All lexical, so all free."""
        src = ep if ep is not None else self
        decisive = [x['fact_id'] for x in src.case.get('decisive_facts', [])]
        n = max(1, len(decisive))
        advisors = {a['agent_id'] for a in src.case['agents']} - {src.dm}
        return {
            'disclosure': sum(f in src.revealed for f in decisive) / n,
            'elicitation': sum(src.reveal_elicited.get(f, False) for f in decisive) / n,
            'coverage': len(src.addressed) / max(1, len(advisors)),
        }

    def _csa_shaping(self, ep=None):
        """Potential-based: gamma*Phi(s') - Phi(s), which provably cannot change the
        optimal policy [Ng et al., 1999]. Only the rate of learning changes."""
        src = ep if ep is not None else self
        w = {'disclosure': getattr(self.args, 'csa_w_shape_disc', 1.0),
             'elicitation': getattr(self.args, 'csa_w_shape_elic', 0.0),
             'coverage': getattr(self.args, 'csa_w_shape_cover', 0.0)}
        phi = self._csa_potentials(ep=ep)
        total = 0.0
        for k, val in phi.items():
            if w[k]:
                total += w[k] * (self.args.gamma * val - src.prev_phi[k])
        src.prev_phi = phi
        return total

    def _csa_resolve_provenance(self, settlement, ep=None):
        """Recover the fact ids a settlement is grounded in from its content, not its
        labels.

        The chair never sees `PF1`. Private fact ids exist only in the dataset and in
        the advisors' views, so `'PF1' in justification_fact_ids` can only be satisfied
        by guessing -- the measured provenance pass rate is 4.8%, and the chair instead
        cites the shared-context ids `S1..S4` it can actually see.

        Naming the private ids in the chair's prompt would hand it the label for
        information it is supposed to elicit, making provenance free to satisfy without
        pooling anything. Instead the settlement's own prose is matched back to fact
        texts with the same lexical detector that populates `revealed`.

        Only facts already disclosed in dialogue are eligible, so resolution can never
        manufacture provenance for information that was never pooled -- it recovers
        credit the chair earned and had no vocabulary to express.
        """
        src = ep if ep is not None else self
        if not isinstance(settlement, dict):
            return settlement
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        prose = [str(v) for v in (settlement.get('decisions') or {}).values()]
        for c in settlement.get('commitments') or []:
            if isinstance(c, dict):
                prose.extend(str(c.get(k, '')) for k in ('type', 'target', 'detail'))
        blob = ' '.join(prose)
        hits = {fid for fid in src.revealed
                if fid in src.case['private_facts']
                and _overlap(src.case['private_facts'][fid]['text'], blob) >= thr}
        if not hits:
            return settlement
        out = dict(settlement)
        for field in ('justification_fact_ids', 'credited_facts'):
            have = [x for x in (out.get(field) or []) if isinstance(x, str)]
            out[field] = have + sorted(hits - set(have))
        return out

    def _csa_finalise_score(self, ep=None):
        src = ep if ep is not None else self
        settlement = src.settlement
        if getattr(self.args, 'csa_resolve_provenance', 1):
            settlement = self._csa_resolve_provenance(settlement, ep=ep)
        s = score(src.case, settlement, src.revealed)
        src.last_score = s
        src.last_score_norm = score(src.case, settlement, src.revealed, norm=True)
        s['reveal_turn'] = dict(src.reveal_turn)
        s['reveal_elicited'] = dict(src.reveal_elicited)
        s['turns'] = src.cur_conver_step + 1
        s['max_turn'] = src.max_turn
        s['leaks'] = list(src.leaks)
        s.update(self._csa_potentials(ep=ep))
        return s

    def _csa_note_addressed(self, chair_utterance, ep=None):
        """Which advisors the chair has directed an eliciting turn at."""
        src = ep if ep is not None else self
        for aid, name in src.names.items():
            if aid != src.dm and name.split()[-1].lower() in chair_utterance.lower():
                src.addressed.add(aid)

    def _csa_note_disclosures(self, speaker, utterance, ep=None):
        """Record which of this advisor's private facts the utterance disclosed, and
        whether the chair's immediately preceding turn was an eliciting act.

        Every disclosure-based metric inherits this detector's error, so the threshold
        is calibrated once on a hand-labelled sample, then frozen and reported.
        """
        src = ep if ep is not None else self
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        for fid, fact in src.case['private_facts'].items():
            if fid in src.revealed or fact['owner'] != speaker:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                src.revealed.add(fid)
                src.reveal_turn[fid] = src.cur_conver_step
                src.reveal_elicited[fid] = src.last_act in CSA_ELICITING_ACTS
                print('--> disclosed %s by %s at turn %d (%s)'
                      % (fid, speaker, src.cur_conver_step,
                         'elicited' if src.reveal_elicited[fid] else 'volunteered'))

    def _csa_note_leaks(self, speaker, utterance, ep=None):
        """An agent stating a fact it never saw and that nobody had yet disclosed.

        Either the view filter leaked or the model hallucinated the fact into being.
        Both invalidate the episode, and neither would otherwise be noticed.
        """
        src = ep if ep is not None else self
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        view = src.case['views'].get(speaker, [])
        for fid, fact in src.case['private_facts'].items():
            if fid in view or fid in src.revealed:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                src.leaks.append({'fact': fid, 'by': speaker,
                                  'turn': src.cur_conver_step})
                print('--> LEAK: %s stated %s without seeing it' % (speaker, fid))

    def _csa_acceptance(self, settlement):
        """Judged acceptance conditions: one narrow yes/no per condition, greedy.

        Optional and off by default. Reintroduces a judge, so validate its agreement
        with the executable checks before giving it any weight.
        """
        conds = self.case.get('acceptance_conditions') or []
        if not conds:
            return 0.0, []
        key = json.dumps(settlement, sort_keys=True)[:2000]
        cache = getattr(self, '_accept_cache', None)
        if cache is None:
            cache = self._accept_cache = {}
        if key in cache:
            return cache[key]
        verdicts = []
        for c in conds:
            msgs = CSAMessages(self.case, 'acceptance', [], action=settlement, agent_id=c)
            try:
                out = self.generate_response(self.args.system, msgs, 'critic')
            except Exception:
                out = ''
            # Unparseable counts as failure, matching how a raising check is treated.
            verdicts.append(str(out).strip().lower().startswith('yes'))
        cache[key] = (sum(verdicts) / len(verdicts), verdicts)
        return cache[key]

    def _csa_postprocess(self, response, speaker, ep=None):
        """Trim where the model starts speaking for somebody else.

        Upstream strips on the other role's bare name, which is safe for 'Patient' but
        with named colleagues would truncate "Dr. Chen recommends dalbavancin" at the
        name. The speaker label -- name plus colon -- is stripped instead.
        """
        src = ep if ep is not None else self
        for agent_id, name in src.names.items():
            if agent_id == speaker:
                continue
            response = self.postprocess_response(response, name + ':')
        return self.postprocess_response(response, src.names[speaker] + ':')

    def _csa_extract_settlement(self, ep=None):
        """Fallback only: used when the chair never emitted parseable JSON."""
        src = ep if ep is not None else self
        msgs = CSAMessages(src.case, 'settlement', src.conversation)
        budget = getattr(self.args, 'csa_settlement_max_tokens', 512)
        saved, self.args.max_new_tokens = self.args.max_new_tokens, budget
        try:
            raw = self.generate_response(self.args.system, msgs, 'critic', ep=ep)
        finally:
            self.args.max_new_tokens = saved
        return self._csa_parse_json(raw)

    @staticmethod
    def _csa_parse_json(raw):
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.M).strip()
        start, depth = text.find('{'), 0
        if start < 0:
            return {}
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    return {}
                return obj if isinstance(obj, dict) else {}
        return {}

    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def _current_temperature(self):
        """Same rule generate_response applies, factored out so the batched rollout
        path can compute it once per round instead of duplicating the logic."""
        if getattr(self, 'force_temperature', None) is not None:
            return self.force_temperature
        return 0 if self.mode == 'test' else 0.7

    def generate_response(self, model, messages, role, ep=None):
        # Cost accounting. Every backend funnels through here, so counting once at the
        # entry point covers all of them. A planner that wins by making more calls than
        # the baseline has not won, so these have to be collected during the run --
        # they cannot be reconstructed from the transcript afterwards.
        # `ep`: batched rollouts track n_calls/calls_by_role/prompt_chars per episode
        # rather than on self (which the batch shares across all episodes); defaults to
        # self so the single-episode call sites are unaffected.
        tgt = ep if ep is not None else self
        tgt.n_calls = getattr(tgt, 'n_calls', 0) + 1
        tgt.calls_by_role = getattr(tgt, 'calls_by_role', {})
        tgt.calls_by_role[role] = tgt.calls_by_role.get(role, 0) + 1
        try:
            tgt.prompt_chars = getattr(tgt, 'prompt_chars', 0) + sum(
                len(m.get('content') or '') for m in messages)
        except Exception:
            pass

        temperature = self._current_temperature()
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'qwen':
            output = self._qwen_generate(messages, role, self.args.max_new_tokens,
                                         temperature, n=1)[0]
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, role)
            #print(messages)
            output = query_openai_model(
                api_key=YOUR_API_KEY,
                messages=messages,
                model=getattr(self.args, 'openai_model', 'gpt-3.5-turbo-0613'),
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output

    def _chat_text(self, chat):
        try:
            return self.vicuna_tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.vicuna_tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)

    def _fit_chat_to_budget(self, chat, budget):
        """Drop the OLDEST conversation turns -- never chat[0], the system message,
        which carries the case's facts and instructions -- until the templated prompt
        fits within `budget` tokens.

        Naive left-truncation of the templated string is wrong here: qwen_prompt()
        puts the system message first, so truncating the raw string from the left
        eats into the case facts before it ever touches old dialogue turns, silently
        corrupting the input instead of just shortening it. This trims at the message
        level and always keeps chat[0] intact.
        """
        if not budget or len(chat) <= 2:
            return chat
        tok = self.vicuna_tokenizer
        trimmed = chat
        while len(trimmed) > 2:
            n = len(tok(self._chat_text(trimmed)).input_ids)
            if n <= budget:
                break
            trimmed = [trimmed[0]] + trimmed[2:]  # drop the oldest non-system turn
        return trimmed

    def _qwen_generate(self, messages, role, max_new_tokens, temperature, n=1):
        """Generate with a chat-templated local model.

        The vicuna and llama2 paths hand-assemble prompt strings with family-specific
        separators, which would be malformed for a ChatML model, so the tokenizer's own
        template is applied. Sampling settings otherwise mirror the upstream local paths.
        """
        chat = qwen_prompt(messages, role)
        budget = getattr(self.args, 'qwen_max_input_tokens', 3072)
        chat = self._fit_chat_to_budget(chat, budget)
        text = self._chat_text(chat)
        inputs = self.vicuna_tokenizer([text], return_tensors='pt').to(
            self.vicuna_model.device)
        kw = dict(max_new_tokens=max_new_tokens, num_return_sequences=n,
                  pad_token_id=self.vicuna_tokenizer.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            # greedy; transformers rejects num_return_sequences > 1 without sampling
            kw.update(do_sample=False, num_return_sequences=1)
        with torch.no_grad():
            out = self.vicuna_model.generate(**inputs, **kw)
        plen = inputs['input_ids'].shape[1]
        return [self.vicuna_tokenizer.decode(o[plen:], skip_special_tokens=True).strip()
                for o in out]

    def _qwen_generate_batch(self, items, max_new_tokens, temperature, num_return_sequences=1):
        """Batched sibling of _qwen_generate: one generate() call for many independent
        (messages, role) rows instead of one call per row. This is the whole point of
        batched rollout -- the model forward pass, not the Python bookkeeping around it,
        is what dominates wall-clock time.

        Left-padding is required for a decoder-only model: generate() always appends new
        tokens on the right, so every row's real content has to end at the same column
        for the batch to advance in lockstep. Returns decoded strings in the same order
        as `items` when num_return_sequences == 1 (the ordinary-turn case); with
        num_return_sequences > 1 (the critic judge's multi-sample draw), returns one list
        of that many samples per item instead -- transformers lays its output out as
        num_return_sequences contiguous rows per input row, so reshaping back to
        per-item groups is just a stride split.
        """
        tok = self.vicuna_tokenizer
        # Conversations grow every turn and this is a decoder-only chat template, so an
        # untruncated batch's prefill cost grows with both rollout_batch and turn number
        # at once -- that combination is what blows a 40GB card, not rollout_batch alone.
        # Drop the oldest conversation turns per-row (keeping each row's system message
        # intact -- see _fit_chat_to_budget) once a row exceeds the budget; 0/None
        # disables this and reproduces the old unbounded behaviour.
        budget = getattr(self.args, 'qwen_max_input_tokens', 3072)
        texts = []
        for messages, role in items:
            chat = qwen_prompt(messages, role)
            chat = self._fit_chat_to_budget(chat, budget)
            texts.append(self._chat_text(chat))

        old_side = getattr(tok, 'padding_side', 'right')
        tok.padding_side = 'left'
        try:
            inputs = tok(texts, return_tensors='pt', padding=True).to(
                self.vicuna_model.device)
        finally:
            tok.padding_side = old_side

        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if num_return_sequences > 1:
            kw['num_return_sequences'] = num_return_sequences
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            # greedy; transformers rejects num_return_sequences > 1 without sampling
            kw.update(do_sample=False)
            kw.pop('num_return_sequences', None)
        with torch.no_grad():
            out = self.vicuna_model.generate(**inputs, **kw)
        plen = inputs['input_ids'].shape[1]
        decoded = [tok.decode(o[plen:], skip_special_tokens=True).strip() for o in out]
        if num_return_sequences <= 1:
            return decoded
        return [decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
                for i in range(len(items))]

    def _qwen_critic_reward_batch(self, cases_and_convos):
        """Batched sibling of compute_reward's qwen/critic branch: the ten-sample
        LLM-judge draw, batched across EPISODES (one padded generate() call scores many
        episodes' transcripts at once) rather than just across the ten samples within
        one episode the way the sequential path's --critic_batch already did. This is
        what makes --csa_reward critic usable under batched rollout/eval.

        `cases_and_convos`: list of (case, conversation) pairs. Returns rewards aligned
        by index, using the same reward_dict text-matching as compute_reward.
        """
        n_items = len(cases_and_convos)
        prompts = [CSAMessages(case, 'critic', convo) for case, convo in cases_and_convos]
        samples = [[] for _ in range(n_items)]
        bs = max(1, min(10, getattr(self.args, 'critic_batch', 10)))
        collected = 0
        while collected < 10:
            take = min(bs, 10 - collected)
            groups = self._qwen_generate_batch(
                [(p, 'critic') for p in prompts], 16, 1.1, num_return_sequences=take)
            for i, g in enumerate(groups):
                samples[i].extend(g)
            collected += take

        rewards = []
        for outputs in samples:
            rs = []
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rs.append(self.reward_dict[self.args.data_name][key])
                        break
            rewards.append(sum(rs) / len(rs) if rs else 0)
        return rewards

    # ------------------------------------------------------------ batched csa rollout
    #
    # Everything above operates on `self` as a single episode. Running many episodes
    # side by side so their Qwen calls can share one batched generate() means the
    # per-episode mutable state (conversation, revealed facts, turn cursor, ...) can no
    # longer live on `self` -- self is shared by the whole batch. It moves into a
    # lightweight namespace instead (`_new_csa_episode`), and every helper above that
    # used to read/write `self.<bookkeeping>` now takes an optional `ep=` and reads/
    # writes that instead, defaulting to `self` so the original single-episode call
    # sites (env.step, evaluate()) are byte-for-byte unchanged.
    #
    # Both reward arms are implemented here. The critic arm's reward needs its own
    # 10-sample LLM-judge draw per episode -- a different batching problem from the
    # ordinary turns above, solved by _qwen_critic_reward_batch: it batches that draw
    # across whichever episodes actually need a judge call this micro-tick (rarer than
    # every episode, since --critic_every throttles it), not just across the ten
    # samples within one episode the way the sequential path's --critic_batch did.

    def _new_csa_episode(self, case):
        """One episode's worth of _csa_reset's bookkeeping, as a free-standing object
        instead of attributes on self."""
        from types import SimpleNamespace
        ep = SimpleNamespace()
        ep.case = case
        ep.dm = case['decision_maker']
        ep.order = list(case['interaction_config']['turn_order'])
        ep.order_ptr = 0
        ep.utterances = 0
        ep.utterance_cap = int(case['interaction_config']['turn_cap'])
        dm_slots = sum(1 for i in range(ep.utterance_cap)
                       if ep.order[i % len(ep.order)] == ep.dm)
        ep.max_turn = max(1, min(dm_slots, self.args.max_turn))
        ep.names = {a['agent_id']: a['name'] for a in case['agents']}
        ep.revealed = set()
        ep.reveal_turn = {}
        ep.reveal_elicited = {}
        ep.addressed = set()
        ep.leaks = []
        ep.settlement = {}
        ep.settle_turn = None
        ep.settled_by = None
        ep.last_score = None
        ep.last_score_norm = None
        ep.prev_phi = {'disclosure': 0.0, 'elicitation': 0.0, 'coverage': 0.0}
        ep.last_act = None
        ep.n_calls = 0
        ep.calls_by_role = {}
        ep.prompt_chars = 0
        ep.act_history = []
        ep.conversation = [{"role": "Meeting",
                            "content": "The group convenes to decide: %s"
                                       % case['description']}]
        ep.cur_conver_step = 0
        ep.done = False
        ep.micro_active = False
        return ep

    def reset_batch(self, batch_size):
        """Start `batch_size` independent csa episodes at once. Returns their initial
        states (conversations), aligned by index with self.batch, which step_batch then
        advances."""
        assert self.args.data_name == 'csa', 'batched rollout only implements csa'
        cases = np.random.choice(self.dataset, size=batch_size, replace=True)
        self.batch = [self._new_csa_episode(c) for c in cases]
        return [list(ep.conversation) for ep in self.batch]

    def reset_batch_fixed(self, cases):
        """Same as reset_batch, but for an explicit, ordered list of cases instead of a
        random draw with replacement -- what evaluation needs: every test case scored
        exactly once, not a random sample of them."""
        assert self.args.data_name == 'csa', 'batched rollout only implements csa'
        self.batch = [self._new_csa_episode(c) for c in cases]
        return [list(ep.conversation) for ep in self.batch]

    def step_batch(self, actions):
        """Vectorized _csa_step, plus both _csa_step_verifier and _csa_step_critic,
        across self.batch.

        `actions`: a list aligned with self.batch; None for an index that should be
        skipped (already done). Returns {index: (conversation, reward, done)} for every
        index that was given a non-None action.

        Each call advances every active episode through its own turn-order until its
        chair speaks (mirroring _csa_step's inner while loop), except that -- since all
        three roles are the same Qwen model in the run this was built for -- every
        episode still waiting on a generation this micro-tick is folded into one padded
        batched generate() call, regardless of whether it's an advisor's or the chair's
        turn. Episodes reach their chair turn at different micro-ticks (turn_order and
        utterance_cap vary per case), so the round continues until none are left waiting.

        Reward is computed after that shared loop, branching on --csa_reward exactly as
        _csa_step does: the verifier arm scores every active episode from its executable
        checks (cheap, lexical, always batchable). The critic arm needs its own
        ten-sample LLM-judge draw per episode -- _qwen_critic_reward_batch batches that
        draw across whichever episodes in this round actually need a judge call this
        micro-tick (--critic_every can skip most of them), rather than across the ten
        samples within one episode the way the sequential path's --critic_batch did.
        """
        active = [i for i, a in enumerate(actions) if a is not None]
        for i in active:
            ep = self.batch[i]
            ep.last_act = actions[i]
            ep.act_history.append(actions[i])
            ep.micro_active = True

        while True:
            due = [i for i in active if self.batch[i].micro_active]
            if not due:
                break
            entries = []  # (index, speaker, messages, role_name, is_settling)
            for i in due:
                ep = self.batch[i]
                speaker = ep.order[ep.order_ptr]
                ep.order_ptr = (ep.order_ptr + 1) % len(ep.order)
                ep.utterances += 1
                if speaker == ep.dm:
                    msgs = CSAMessages(ep.case, 'system', ep.conversation, ep.last_act)
                    settling = (ep.last_act == CSA_SETTLING_ACT)
                else:
                    msgs = CSAMessages(ep.case, 'user', ep.conversation, agent_id=speaker)
                    settling = False
                entries.append((i, speaker, msgs, ep.names[speaker], settling))

            temperature = self._current_temperature()
            # A settlement turn needs a much larger token budget (a full JSON object)
            # than an ordinary utterance; generate() takes one max_new_tokens for the
            # whole batch, so settling rows are drawn in their own smaller batch rather
            # than forcing every row to the settlement budget.
            normal = [e for e in entries if not e[4]]
            settle = [e for e in entries if e[4]]
            resp_map = {}
            if normal:
                outs = self._qwen_generate_batch(
                    [(e[2], e[3]) for e in normal], self.args.max_new_tokens, temperature)
                resp_map.update({e[0]: r for e, r in zip(normal, outs)})
            if settle:
                budget = getattr(self.args, 'csa_settlement_max_tokens', 512)
                outs = self._qwen_generate_batch(
                    [(e[2], e[3]) for e in settle], budget, temperature)
                resp_map.update({e[0]: r for e, r in zip(settle, outs)})

            for i, speaker, msgs, role_name, settling in entries:
                ep = self.batch[i]
                resp = resp_map[i]
                ep.n_calls += 1
                ep.calls_by_role[role_name] = ep.calls_by_role.get(role_name, 0) + 1
                ep.prompt_chars += sum(len(m.get('content') or '') for m in msgs)
                if speaker == ep.dm:
                    if not (settling or self._csa_parse_json(resp)):
                        resp = self._csa_postprocess(resp, speaker, ep=ep)
                    ep.conversation.append({"role": ep.names[speaker], "content": resp})
                    print(ep.conversation[-1])
                    if ep.last_act in CSA_ELICITING_ACTS:
                        self._csa_note_addressed(resp, ep=ep)
                    ep.micro_active = False
                else:
                    resp = self._csa_postprocess(resp, speaker, ep=ep)
                    ep.conversation.append({"role": ep.names[speaker], "content": resp})
                    print(ep.conversation[-1])
                    self._csa_note_disclosures(speaker, resp, ep=ep)
                    self._csa_note_leaks(speaker, resp, ep=ep)
                    if ep.utterances >= ep.utterance_cap:
                        ep.micro_active = False

        results = {}
        if getattr(self.args, 'csa_reward', 'critic') == 'verifier':
            for i in active:
                ep = self.batch[i]
                last_turn = (ep.cur_conver_step == ep.max_turn - 1)
                parsed = self._csa_take_settlement(final=last_turn, ep=ep)
                scored = bool(parsed) or last_turn
                if scored:
                    s = self._csa_finalise_score(ep=ep)
                    terminal = self._csa_terminal_reward(s, ep=ep)
                    tau = getattr(self.args, 'csa_done_tau', 0.6)
                    invalid = bool(ep.leaks) and getattr(self.args, 'csa_leak_invalidates', 1)
                    if s['schema_valid'] and not invalid and s['dca'] >= tau:
                        print('--> Goal completed !')
                        ep.cur_conver_step += 1
                        ep.done = True
                        results[i] = (list(ep.conversation), terminal, 1)
                        continue
                    if last_turn:
                        print('--> Maximum number of turns reached !')
                        ep.cur_conver_step += 1
                        ep.done = True
                        results[i] = (list(ep.conversation), terminal, -1)
                        continue
                reward = -0.1 + self._csa_shaping(ep=ep)
                print('--> On-going ! %s' % self._csa_potentials(ep=ep))
                ep.cur_conver_step += 1
                results[i] = (list(ep.conversation), reward, 0)
            return results

        # Arm A: upstream's LLM critic, batched across episodes -- see
        # _qwen_critic_reward_batch. --critic_every throttles which episodes actually
        # need a judge call this round; the rest get the free reward=0.0 the sequential
        # path also gives them, with no generate() call at all.
        every = getattr(self.args, 'critic_every', 1)
        last_turn_of = {}
        need_judge = []
        for i in active:
            ep = self.batch[i]
            last_turn = (ep.cur_conver_step == ep.max_turn - 1)
            last_turn_of[i] = last_turn
            if every > 1 and not last_turn and (ep.cur_conver_step + 1) % every:
                ep._pending_reward = 0.0
            else:
                need_judge.append(i)
        if need_judge:
            items = [(self.batch[i].case, list(self.batch[i].conversation))
                     for i in need_judge]
            rewards = self._qwen_critic_reward_batch(items)
            for i, r in zip(need_judge, rewards):
                self.batch[i]._pending_reward = r

        for i in active:
            ep = self.batch[i]
            last_turn = last_turn_of[i]
            reward = ep._pending_reward
            # Measurement only; never enters `reward` in this arm, matching
            # _csa_step_critic exactly.
            if reward == 1:
                print('--> Goal completed !')
                done = 1
            elif last_turn:
                print('--> Maximum number of turns reached !')
                done = -1
            else:
                print('--> On-going !')
                done = 0
            self._csa_take_settlement(final=bool(done), ep=ep)
            if done:
                self._csa_finalise_score(ep=ep)
            ep.cur_conver_step += 1
            results[i] = (list(ep.conversation), reward, done)
        return results

    def compute_reward(self, model, messages, case):
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'qwen':
            # Same shape as the vicuna critic: ten samples, temperature 1.1, 16 tokens.
            # They are i.i.d., so drawing them in smaller batches is distributionally
            # identical and only lowers peak memory -- the critic dominates the KV cache.
            bs = max(1, min(10, getattr(self.args, 'critic_batch', 10)))
            outputs = []
            while len(outputs) < 10:
                outputs += self._qwen_generate(messages, 'critic', 16, 1.1,
                                               n=min(bs, 10 - len(outputs)))
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_openai_model(
                api_key=YOUR_API_KEY,
                messages=messages,
                model=getattr(self.args, 'openai_model', 'gpt-3.5-turbo-0613'),
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )

        if self.args.data_name in ['esc','cima','csa']:
            rewards = []
            print(outputs)
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
            print(reward)
        elif self.args.data_name == 'cb':
            deals = []
            rewards = []
            print(outputs)
            for output in outputs:
                if 'have not' in output.lower():
                    deals.append(-1)
                elif 'have reached' in output.lower():
                    deals.append(1)
                
                prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
                if len(prices) > 0:
                    deal_price = float(prices[0])
                    reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                    rewards.append(reward)

            if -1 in deals:
                reward = -0.1
            else:
                if len(rewards) == 0:
                    reward = 0
                else:
                    reward = max(set(rewards), key = rewards.count)
            print(reward)

        return reward



def query_openai_model(api_key: str, messages: str, model: str = "gpt-3.5-turbo-0613", max_tokens: int = 128, temperature: float = 0, n: int = 1):
    openai.api_key = api_key
    flag = True
    while flag:
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                stop=None,
                temperature=temperature,
                request_timeout=10,
            )

            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    return output