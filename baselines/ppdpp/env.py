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

    def _csa_take_settlement(self, final):
        """Keep the chair's latest parseable settlement and, on the final turn with none,
        fall back to the extractor. Both reward arms call this, so both are scored alike.
        Returns what this turn's chair message parsed to ({} when nothing)."""
        parsed = self._csa_parse_json(self.conversation[-1]['content'])
        if parsed:
            self.settlement = parsed
            self.settle_turn, self.settled_by = self.cur_conver_step, 'chair'
        if final and not self.settlement:
            self.settlement = self._csa_extract_settlement()
            if self.settlement:
                self.settle_turn, self.settled_by = self.cur_conver_step, 'extractor'
        return parsed

    def _csa_terminal_reward(self, s):
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
        if not s['schema_valid']:
            return -1.0
        if bool(self.leaks) and getattr(self.args, 'csa_leak_invalidates', 1):
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
            r += self.args.csa_w_accept * self._csa_acceptance(self.settlement)[0]
        return max(-1.0, min(1.0, r))

    def _csa_potentials(self):
        """Monotone progress measures. All lexical, so all free."""
        decisive = [x['fact_id'] for x in self.case.get('decisive_facts', [])]
        n = max(1, len(decisive))
        advisors = {a['agent_id'] for a in self.case['agents']} - {self.dm}
        return {
            'disclosure': sum(f in self.revealed for f in decisive) / n,
            'elicitation': sum(self.reveal_elicited.get(f, False) for f in decisive) / n,
            'coverage': len(self.addressed) / max(1, len(advisors)),
        }

    def _csa_shaping(self):
        """Potential-based: gamma*Phi(s') - Phi(s), which provably cannot change the
        optimal policy [Ng et al., 1999]. Only the rate of learning changes."""
        w = {'disclosure': getattr(self.args, 'csa_w_shape_disc', 1.0),
             'elicitation': getattr(self.args, 'csa_w_shape_elic', 0.0),
             'coverage': getattr(self.args, 'csa_w_shape_cover', 0.0)}
        phi = self._csa_potentials()
        total = 0.0
        for k, val in phi.items():
            if w[k]:
                total += w[k] * (self.args.gamma * val - self.prev_phi[k])
        self.prev_phi = phi
        return total

    def _csa_resolve_provenance(self, settlement):
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
        if not isinstance(settlement, dict):
            return settlement
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        prose = [str(v) for v in (settlement.get('decisions') or {}).values()]
        for c in settlement.get('commitments') or []:
            if isinstance(c, dict):
                prose.extend(str(c.get(k, '')) for k in ('type', 'target', 'detail'))
        blob = ' '.join(prose)
        hits = {fid for fid in self.revealed
                if fid in self.case['private_facts']
                and _overlap(self.case['private_facts'][fid]['text'], blob) >= thr}
        if not hits:
            return settlement
        out = dict(settlement)
        for field in ('justification_fact_ids', 'credited_facts'):
            have = [x for x in (out.get(field) or []) if isinstance(x, str)]
            out[field] = have + sorted(hits - set(have))
        return out

    def _csa_finalise_score(self):
        settlement = self.settlement
        if getattr(self.args, 'csa_resolve_provenance', 1):
            settlement = self._csa_resolve_provenance(settlement)
        s = score(self.case, settlement, self.revealed)
        self.last_score = s
        self.last_score_norm = score(self.case, settlement, self.revealed, norm=True)
        s['reveal_turn'] = dict(self.reveal_turn)
        s['reveal_elicited'] = dict(self.reveal_elicited)
        s['turns'] = self.cur_conver_step + 1
        s['max_turn'] = self.max_turn
        s['leaks'] = list(self.leaks)
        s.update(self._csa_potentials())
        return s

    def _csa_note_addressed(self, chair_utterance):
        """Which advisors the chair has directed an eliciting turn at."""
        for aid, name in self.names.items():
            if aid != self.dm and name.split()[-1].lower() in chair_utterance.lower():
                self.addressed.add(aid)

    def _csa_note_disclosures(self, speaker, utterance):
        """Record which of this advisor's private facts the utterance disclosed, and
        whether the chair's immediately preceding turn was an eliciting act.

        Every disclosure-based metric inherits this detector's error, so the threshold
        is calibrated once on a hand-labelled sample, then frozen and reported.
        """
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        for fid, fact in self.case['private_facts'].items():
            if fid in self.revealed or fact['owner'] != speaker:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                self.revealed.add(fid)
                self.reveal_turn[fid] = self.cur_conver_step
                self.reveal_elicited[fid] = self.last_act in CSA_ELICITING_ACTS
                print('--> disclosed %s by %s at turn %d (%s)'
                      % (fid, speaker, self.cur_conver_step,
                         'elicited' if self.reveal_elicited[fid] else 'volunteered'))

    def _csa_note_leaks(self, speaker, utterance):
        """An agent stating a fact it never saw and that nobody had yet disclosed.

        Either the view filter leaked or the model hallucinated the fact into being.
        Both invalidate the episode, and neither would otherwise be noticed.
        """
        thr = getattr(self.args, 'csa_reveal_threshold', 0.35)
        view = self.case['views'].get(speaker, [])
        for fid, fact in self.case['private_facts'].items():
            if fid in view or fid in self.revealed:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                self.leaks.append({'fact': fid, 'by': speaker,
                                   'turn': self.cur_conver_step})
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

    def _csa_postprocess(self, response, speaker):
        """Trim where the model starts speaking for somebody else.

        Upstream strips on the other role's bare name, which is safe for 'Patient' but
        with named colleagues would truncate "Dr. Chen recommends dalbavancin" at the
        name. The speaker label -- name plus colon -- is stripped instead.
        """
        for agent_id, name in self.names.items():
            if agent_id == speaker:
                continue
            response = self.postprocess_response(response, name + ':')
        return self.postprocess_response(response, self.names[speaker] + ':')

    def _csa_extract_settlement(self):
        """Fallback only: used when the chair never emitted parseable JSON."""
        msgs = CSAMessages(self.case, 'settlement', self.conversation)
        budget = getattr(self.args, 'csa_settlement_max_tokens', 512)
        saved, self.args.max_new_tokens = self.args.max_new_tokens, budget
        try:
            raw = self.generate_response(self.args.system, msgs, 'critic')
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

    def generate_response(self, model, messages, role):
        # Cost accounting. Every backend funnels through here, so counting once at the
        # entry point covers all of them. A planner that wins by making more calls than
        # the baseline has not won, so these have to be collected during the run --
        # they cannot be reconstructed from the transcript afterwards.
        self.n_calls = getattr(self, 'n_calls', 0) + 1
        self.calls_by_role = getattr(self, 'calls_by_role', {})
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        try:
            self.prompt_chars = getattr(self, 'prompt_chars', 0) + sum(
                len(m.get('content') or '') for m in messages)
        except Exception:
            pass

        # force_temperature lets a caller iterate the split sequentially (mode='test')
        # while still sampling. Without it, k unprompted rollouts of one scenario are
        # byte-identical and any ranking over them is vacuous.
        if getattr(self, 'force_temperature', None) is not None:
            temperature = self.force_temperature
        elif self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
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

    def _qwen_generate(self, messages, role, max_new_tokens, temperature, n=1):
        """Generate with a chat-templated local model.

        The vicuna and llama2 paths hand-assemble prompt strings with family-specific
        separators, which would be malformed for a ChatML model, so the tokenizer's own
        template is applied. Sampling settings otherwise mirror the upstream local paths.
        """
        chat = qwen_prompt(messages, role)
        kw = {}
        # Qwen3 emits <think> blocks unless thinking is disabled; Qwen2.5 has no such arg.
        try:
            text = self.vicuna_tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.vicuna_tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
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