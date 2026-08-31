"""The CSA meeting, driven by a prompting strategy. No training anywhere.

Sotopia-ToM is an inference-time method, so this environment is simpler than the RL ones:
one frozen model plays every role, the chair's utterance comes from whichever strategy is
active, and nothing has gradients. Turn order, view filtering, the disclosure detector and
the verifier are identical to the other baselines so the arms stay comparable.

The ToM arms cost one extra model call per chair turn -- the analyst call for tom_coach,
the belief update for tom_belief -- and that cost is counted in n_calls like everything
else, because a strategy that wins by making more calls has not won.
"""
import json
import re

import torch

import compat
import detectors_tom as D
import prompts_tom as P
from verifier_tom import floor_score, score

try:
    import nltk
    _SENT = nltk.sent_tokenize
except Exception:                                    # noqa: BLE001
    _SENT = None


class ToMEnv(object):
    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        if model is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.tokenizer = compat.load_tokenizer(cfg.model)
            self.model = compat.load_causal_lm(cfg.model, cfg.dtype, cfg.device)
        self.greedy = True                           # evaluation is deterministic

    # ------------------------------------------------------------- lifecycle
    def reset(self, case, strategy='basic'):
        self.case = case
        self.strategy = strategy
        self.dm = case['decision_maker']
        self.order = list(case['interaction_config']['turn_order'])
        self.ptr = 0
        self.utterances = 0
        self.cap = int(case['interaction_config']['turn_cap'])
        dm_slots = sum(1 for i in range(self.cap)
                       if self.order[i % len(self.order)] == self.dm)
        self.max_turn = max(1, dm_slots)

        self.names = {a['agent_id']: a['name'] for a in case['agents']}
        self.ids = {a['name']: a['agent_id'] for a in case['agents']}
        self.advisors = set(self.names) - {self.dm}

        self.step_i = 0
        self.revealed, self.reveal_turn, self.reveal_elicited = set(), {}, {}
        self.addressed, self.pending, self.cover_credit = set(), {}, {}
        self.leaks = []
        self.settlement, self.settle_turn = {}, None
        self.n_calls, self.calls_by_role, self.prompt_chars = 0, {}, 0
        self.last_score = None
        self.belief = {}                             # tom_belief only
        self.tom_notes = []                          # what was injected, for the record
        self.thinking = []                           # cot only, kept out of the transcript
        self._belief_mark = 1                        # first unconsumed turn index

        self.conversation = [{'role': 'Meeting',
                              'content': 'The group convenes to decide: %s'
                                         % case['description']}]
        return self.conversation

    def is_settling_turn(self):
        return self.step_i >= self.max_turn - 1

    # ------------------------------------------------------------- the loop
    def step(self):
        """Produce the chair's turn under the active strategy, then let advisors reply.

        Returns (conversation, done).
        """
        note = self._tom_note()
        msgs = P.chair_messages(self.case, self.conversation, strategy=self.strategy,
                                settling=self.is_settling_turn(), tom_note=note)
        budget = (self.cfg.settlement_max_tokens if self.is_settling_turn()
                  else self.cfg.max_new_tokens)
        if self.strategy == 'cot' and not self.is_settling_turn():
            budget = budget + self.cfg.cot_extra_tokens
        raw = self._generate(msgs, self.names[self.dm], budget)

        text = raw
        if self.strategy == 'cot':
            text = P.split_thinking(raw)
            if raw != text:
                self.thinking.append(raw[:len(raw) - len(text)].strip()[:400])
        if not (self.is_settling_turn() or self._parse_json(text)):
            text = self._trim_for(text, self.dm)

        eliciting = D.is_eliciting(text, self.case['agents'], self.dm)
        fresh = D.addressed_in(text, self.case['agents'], exclude={self.dm}) - self.addressed
        self.addressed |= fresh
        for aid in fresh:
            self.pending[aid] = self.step_i
        for fid in D.leaks(self.case, self.dm, text, self.revealed,
                           self.cfg.reveal_threshold):
            self.leaks.append({'fact': fid, 'by': self.dm, 'turn': self.step_i})

        self.conversation.append({'role': self.names[self.dm], 'content': text})
        self.utterances += 1
        self.ptr = (self.order.index(self.dm) + 1) % len(self.order)

        parsed = self._parse_json(text)
        if parsed:
            self.settlement = parsed
            self.settle_turn = self.step_i

        while self.utterances < self.cap:
            speaker = self.order[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.order)
            if speaker == self.dm:
                break
            resp = self._generate(P.advisor_messages(self.case, self.conversation, speaker),
                                  self.names[speaker], self.cfg.max_new_tokens)
            resp = self._trim_for(resp, speaker)
            self.conversation.append({'role': self.names[speaker], 'content': resp})
            self.utterances += 1
            self._note(speaker, resp, eliciting)

        last = self.step_i >= self.max_turn - 1
        self.step_i += 1
        if last:
            if not self.settlement:
                self.settlement = self._extract_settlement()
                if self.settlement:
                    self.settle_turn = self.step_i - 1
            self._finalise()
            return self.conversation, -1
        return self.conversation, 0

    # ------------------------------------------------------------- ToM scaffolds
    def _tom_note(self):
        if self.strategy == 'tom_coach':
            note = self._generate(P.analyst_messages(self.case, self.conversation),
                                  'analyst', self.cfg.tom_max_tokens)
            self.tom_notes.append(note[:600])
            return note
        if self.strategy == 'tom_belief':
            new = self.conversation[self._belief_mark:]
            self._belief_mark = len(self.conversation)
            raw = self._generate(P.belief_messages(self.case, self.belief, new),
                                 'belief', self.cfg.tom_max_tokens)
            got = self._parse_json(raw)
            if got:
                # incremental: keep the previous state when an update fails to parse
                self.belief = got
            note = P.render_belief(self.belief)
            self.tom_notes.append(note[:600])
            return note
        return ''

    # ------------------------------------------------------------- detectors
    def _note(self, speaker, utterance, eliciting):
        got = D.disclosures(self.case, speaker, utterance, self.revealed,
                            self.cfg.reveal_threshold)
        for fid in got:
            self.revealed.add(fid)
            self.reveal_turn[fid] = self.step_i
            self.reveal_elicited[fid] = eliciting
        if got and speaker in self.pending:
            self.cover_credit.setdefault(self.pending.pop(speaker), set()).add(speaker)
        for fid in D.leaks(self.case, speaker, utterance, self.revealed,
                           self.cfg.reveal_threshold):
            self.leaks.append({'fact': fid, 'by': speaker, 'turn': self.step_i})

    def _finalise(self):
        self.last_score = score(self.case, self.settlement, self.revealed, resolve=True)
        return self.last_score

    # ------------------------------------------------------------- output
    def dialog(self):
        out = []
        for t in self.conversation:
            spk = 'env' if t['role'] == 'Meeting' else (
                'sys' if t['role'] == self.names[self.dm] else 'usr')
            out.append({'role': t['role'], 'content': t['content'], 'speaker': spk})
        return out

    def record(self, turns):
        s = self.last_score or self._finalise()
        pub = {k: v for k, v in s.items() if k != 'settlement_resolved'}
        n_adv = max(1, len(self.advisors))
        done = 1 if (s['schema_valid'] and not self.leaks
                     and s['dca'] >= self.cfg.done_tau) else -1
        return {'dialog': self.dialog(), 'reward': s['dca'],
                'uid': self.case.get('uid'), 'domain': self.case.get('domain'),
                'num_agents': self.case.get('num_agents'),
                'scenario_type': self.case.get('scenario_type'),
                'strategy': self.strategy,
                'settlement': self.settlement, 'score': pub,
                'floor': {k: v for k, v in floor_score(self.case).items()
                          if k != 'settlement_resolved'},
                'revealed': sorted(self.revealed),
                'reveal_elicited': dict(self.reveal_elicited),
                'reveal_turn': dict(self.reveal_turn),
                'addressed': sorted(self.addressed), 'leaks': list(self.leaks),
                'done': done, 'turns': turns, 'max_turn': self.max_turn,
                'n_calls': self.n_calls, 'calls_by_role': dict(self.calls_by_role),
                'prompt_chars': self.prompt_chars,
                'cover': len(set().union(*self.cover_credit.values())) / n_adv
                         if self.cover_credit else 0.0,
                'tom_notes': list(self.tom_notes), 'thinking': list(self.thinking)}

    # ------------------------------------------------------------- generation
    def _generate(self, messages, speaker, max_new_tokens):
        self.n_calls += 1
        self.calls_by_role[speaker] = self.calls_by_role.get(speaker, 0) + 1
        self.prompt_chars += sum(len(m.get('content') or '') for m in messages)

        text = compat.render_chat(self.tokenizer, P.to_chat(messages, speaker))
        enc = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if self.greedy:
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, temperature=self.cfg.temperature)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        return self.tokenizer.decode(out[0][enc['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()

    def _extract_settlement(self):
        return self._parse_json(
            self._generate(P.settlement_messages(self.case, self.conversation),
                           'extractor', self.cfg.settlement_max_tokens))

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

    def _trim_for(self, response, speaker):
        for aid, name in self.names.items():
            if aid != speaker:
                response = self._cut(response, name + ':')
        return self._cut(response, self.names[speaker] + ':')

    @staticmethod
    def _cut(response, marker):
        if marker in response:
            response = response.split(marker)[0].strip()
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
