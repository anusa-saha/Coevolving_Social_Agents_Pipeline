"""The CSA meeting environment, driven by an open-ended strategy instead of an act index.

Parallel to ppdpp_csa/env.py, not a modification of it. What is deliberately identical:
the turn-order walk, the per-case utterance cap, the view filtering, the disclosure
threshold (0.35, frozen), and scoring via ppdpp_csa/verifier.py. What changes:

  * step() takes a raw "tau: sigma" string, not an index into CSAAct
  * step() returns NO reward. EPO assigns credit post-hoc from the finished trajectory,
    so rewards come from prm.py after the episode ends
  * _note_leaks now also runs on CHAIR turns. In ppdpp_csa it runs on advisor turns only
    (env.py:279), which was safe for a four-way classifier and is not safe for a trained
    LLM writing free text into the chair's prompt
  * potential-based shaping is gone: it measures elicited disclosure, which is exactly
    what the verifier PRM now rewards, so keeping both counts it twice
"""
import json
import re

import torch

import compat
import config                                        # noqa: F401  (sys.path shim)
import prompt_epo as pe
from detectors import _overlap
from prm import Trace
from verifier import floor_score, score              # ppdpp_csa, byte-identical

try:
    import nltk
    _SENT = nltk.sent_tokenize
except Exception:                                    # noqa: BLE001
    _SENT = None


class EPOEnv(object):
    def __init__(self, cfg, dataset, mode='train', model=None, tokenizer=None):
        self.cfg = cfg
        self.dataset = dataset
        self.mode = mode
        self.test_num = 0

        if model is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.tokenizer = compat.load_tokenizer(cfg.agent_model)
            self.model = compat.load_causal_lm(cfg.agent_model, cfg.agent_dtype,
                                               cfg.agent_device)

    # ------------------------------------------------------------- episode setup
    def reset(self, case=None):
        """`case` may be forced, which PPDPP's reset cannot do.

        Group-relative advantage needs k rollouts of the SAME scenario, so the caller
        has to be able to pin the case rather than draw one at random.
        """
        if case is not None:
            self.case = case
        elif self.mode == 'train':
            import numpy as np
            self.case = self.dataset[np.random.randint(len(self.dataset))]
        else:
            self.case = self.dataset[self.test_num % len(self.dataset)]
            self.test_num += 1

        c = self.case
        self.dm = c['decision_maker']
        self.order = list(c['interaction_config']['turn_order'])
        self.order_ptr = 0
        self.utterances = 0
        self.utterance_cap = int(c['interaction_config']['turn_cap'])
        dm_slots = sum(1 for i in range(self.utterance_cap)
                       if self.order[i % len(self.order)] == self.dm)
        self.max_turn = max(1, dm_slots)

        self.names = {a['agent_id']: a['name'] for a in c['agents']}
        self.cur_step = 0
        self.revealed = set()
        self.reveal_turn = {}
        self.reveal_elicited = {}
        self.addressed = set()
        self.leaks = []
        self.settlement = {}
        self.settle_turn = None
        self.last_act = None
        self.acts = []
        self.sigmas = []
        self.tag_misses = 0
        self.last_score = None
        self.last_score_norm = None
        self.n_calls = 0
        self.calls_by_role = {}
        self.prompt_chars = 0

        # Same seed turn as ppdpp_csa: shared framing, so it discloses nothing.
        self.conversation = [{'role': 'Meeting',
                              'content': 'The group convenes to decide: %s' % c['description']}]
        return self.conversation

    # ------------------------------------------------------------- the loop
    def step(self, strategy_text):
        """Advance the turn order until the chair has spoken once.

        Returns (conversation, done). No reward: see module docstring.
        """
        tau, sigma, ok = pe.parse_strategy(strategy_text)
        if not ok:
            self.tag_misses += 1
        self.last_act = tau
        self.acts.append(tau)
        self.sigmas.append(sigma)

        while True:
            speaker = self.order[self.order_ptr]
            self.order_ptr = (self.order_ptr + 1) % len(self.order)
            self.utterances += 1

            if speaker == self.dm:
                msgs = pe.chair_messages(self.case, self.conversation, tau, sigma)
                budget = (self.cfg.settlement_max_tokens if tau == pe.CSA_SETTLING_ACT
                          else self.cfg.agent_max_new_tokens)
                resp = self._generate(msgs, self.names[speaker], budget)
                # Trimming at sentence boundaries destroys a JSON object.
                if not (tau == pe.CSA_SETTLING_ACT or self._parse_json(resp)):
                    resp = self._postprocess_for(resp, speaker)
                self.conversation.append({'role': self.names[speaker], 'content': resp})
                if tau in pe.CSA_ELICITING_ACTS:
                    self._note_addressed(resp)
                # THE FIX: the chair is a leak surface once a trained model writes its
                # prompt. Its view provably excludes every private fact (export_csa.py
                # check_invariants), so the detector is correct here unmodified.
                self._note_leaks(speaker, resp)
                break

            msgs = pe.advisor_messages(self.case, self.conversation, speaker)
            resp = self._generate(msgs, self.names[speaker], self.cfg.agent_max_new_tokens)
            resp = self._postprocess_for(resp, speaker)
            self.conversation.append({'role': self.names[speaker], 'content': resp})
            self._note_disclosures(speaker, resp)
            self._note_leaks(speaker, resp)

            if self.utterances >= self.utterance_cap:
                break

        last_turn = self.cur_step == self.max_turn - 1
        parsed = self._parse_json(self.conversation[-1]['content'])
        if parsed:
            self.settlement = parsed
            self.settle_turn = self.cur_step

        done = 0
        if parsed or last_turn:
            if not self.settlement and last_turn:
                self.settlement = self._extract_settlement()
                if self.settlement:
                    self.settle_turn = self.cur_step
            s = self._finalise_score()
            invalid = bool(self.leaks) and self.cfg.leak_invalidates
            if s['schema_valid'] and not invalid and s['dca'] >= self.cfg.done_tau:
                done = 1
            elif last_turn:
                done = -1
        elif last_turn:
            self._finalise_score()
            done = -1

        self.cur_step += 1
        return self.conversation, done

    # ------------------------------------------------------------- scoring
    def _finalise_score(self):
        settlement = self.settlement
        if self.cfg.resolve_provenance:
            settlement = self._resolve_provenance(settlement)
        self.last_score = score(self.case, settlement, self.revealed)
        self.last_score_norm = score(self.case, settlement, self.revealed, norm=True)
        return self.last_score

    def terminal_reward(self):
        """ppdpp_csa's _csa_terminal_reward, unchanged in shape.

        Kept ON TOP of the process rewards, unlike vanilla EPO which uses process
        rewards alone. Reason: without it an invalid or leaking episode returns all
        zeros, which under a group-relative baseline is indistinguishable from an
        episode that simply did nothing. The negative signal has to exist.
        """
        s = self.last_score or self._finalise_score()
        if not s['schema_valid']:
            return -1.0
        if bool(self.leaks) and self.cfg.leak_invalidates:
            return -1.0
        pool, use, close = s['disclosure_rate'], s['dca'], s['close']
        if pool == 0.0 and use == 0.0:
            return -0.5
        a, b, g = self.cfg.w_use, self.cfg.w_pool, self.cfg.w_close
        tot = (a + b + g) or 1.0
        r = 2.0 * ((a * use + b * pool + g * close) / tot) - 1.0
        r -= self.cfg.w_halluc_pen * s['hallucinated_credit']
        return max(-1.0, min(1.0, r))

    def trace(self):
        s = self.last_score or self._finalise_score()
        return Trace(n_turns=self.cur_step,
                     reveal_turn=dict(self.reveal_turn),
                     reveal_elicited=dict(self.reveal_elicited),
                     decisive=self.case.get('decisive_facts') or [],
                     settle_turn=self.settle_turn,
                     dca=s['dca'], schema_valid=s['schema_valid'],
                     leaks=list(self.leaks), acts=list(self.acts),
                     conversation=list(self.conversation), case=self.case)

    def record(self, reward, done, turns):
        """Same shape as ppdpp_csa/run.py evaluate(), so compute_all_metrics.py runs
        on EPO records with no branch."""
        return {
            'dialog': list(self.conversation), 'reward': reward,
            'uid': self.case.get('uid'), 'domain': self.case.get('domain'),
            'num_agents': self.case.get('num_agents'),
            'scenario_type': self.case.get('scenario_type'),
            'settlement': self.settlement, 'score': self.last_score,
            'score_norm': self.last_score_norm, 'floor': floor_score(self.case),
            'revealed': sorted(self.revealed),
            'reveal_elicited': dict(self.reveal_elicited),
            'addressed': sorted(self.addressed), 'leaks': list(self.leaks),
            'done': done, 'turns': turns, 'max_turn': self.max_turn,
            'n_calls': self.n_calls, 'calls_by_role': dict(self.calls_by_role),
            'prompt_chars': self.prompt_chars,
            'act_history': list(self.acts), 'reveal_turn': dict(self.reveal_turn),
            # EPO-only diagnostics
            'strategies': list(self.sigmas), 'tag_misses': self.tag_misses,
        }

    # ------------------------------------------------------------- detectors
    def _note_addressed(self, utterance):
        for aid, name in self.names.items():
            if aid != self.dm and name.split()[-1].lower() in utterance.lower():
                self.addressed.add(aid)

    def _note_disclosures(self, speaker, utterance):
        thr = self.cfg.reveal_threshold
        for fid, fact in self.case['private_facts'].items():
            if fid in self.revealed or fact['owner'] != speaker:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                self.revealed.add(fid)
                self.reveal_turn[fid] = self.cur_step
                self.reveal_elicited[fid] = self.last_act in pe.CSA_ELICITING_ACTS

    def _note_leaks(self, speaker, utterance):
        thr = self.cfg.reveal_threshold
        view = self.case['views'].get(speaker, [])
        for fid, fact in self.case['private_facts'].items():
            if fid in view or fid in self.revealed:
                continue
            if _overlap(fact['text'], utterance) >= thr:
                self.leaks.append({'fact': fid, 'by': speaker, 'turn': self.cur_step})

    def _resolve_provenance(self, settlement):
        if not isinstance(settlement, dict):
            return settlement
        thr = self.cfg.reveal_threshold
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

    # ------------------------------------------------------------- generation
    def _generate(self, messages, role, max_new_tokens):
        self.n_calls += 1
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        self.prompt_chars += sum(len(m.get('content') or '') for m in messages)

        temperature = 0.0 if self.mode == 'test' else self.cfg.agent_temperature
        chat = pe.qwen_prompt(messages, role)
        text = compat.render_chat(self.tokenizer, chat)
        inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()

    def _extract_settlement(self):
        msgs = pe.settlement_messages(self.case, self.conversation)
        return self._parse_json(self._generate(msgs, 'critic',
                                               self.cfg.settlement_max_tokens))

    # ------------------------------------------------------------- text utils
    @staticmethod
    def _parse_json(raw):
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
                except Exception:                    # noqa: BLE001
                    return {}
                return obj if isinstance(obj, dict) else {}
        return {}

    def _postprocess_for(self, response, speaker):
        for aid, name in self.names.items():
            if aid != speaker:
                response = self._trim(response, name + ':')
        return self._trim(response, self.names[speaker] + ':')

    @staticmethod
    def _trim(response, role):
        """ppdpp_csa/env.py:584 postprocess_response, verbatim in behaviour."""
        if role in response:
            response = response.split(role)[0].strip()
        if not response:
            return response
        if _SENT is None:
            return response.strip()
        sents = _SENT(response)
        if len(sents) == 1:
            return response if response[-1] in '.!?:' else response + '.'
        try:
            if sents[-1].strip()[-1] not in '.!?:':
                return ' '.join(sents[:-1]).strip()
            return response.strip()
        except Exception:                            # noqa: BLE001
            return response.strip()
