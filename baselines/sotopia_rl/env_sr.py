"""The CSA meeting environment for Sotopia-RL.

Differs from the planner baselines in one structural way: the chair's utterance is
supplied by the CALLER, because here the chair IS the policy being trained. There is no
act, no strategy, no conditioning signal -- `step(utterance)` takes text.

Advisors stay frozen. Training them on the same reward collapses the benchmark: an
advisor optimised for the group's outcome learns to state its private fact in its first
utterance, disclosure goes to 1, the hidden profile stops being hidden, and the chair
needs no skill at all.

snapshot()/restore() exist so a candidate utterance can be tried and rolled back. GRPO
needs to score G candidates at the same state, and the only way to know whether a
candidate actually elicited anything is to let an advisor answer it.
"""
import copy
import json
import re

import torch

import compat
import detectors_sr as D
import prompts_sr as P
from verifier_sr import floor_score, score

try:
    import nltk
    _SENT = nltk.sent_tokenize
except Exception:                                    # noqa: BLE001
    _SENT = None


class SREnv(object):
    def __init__(self, cfg, backbone=None):
        """`backbone` is a SharedBackbone. The advisors and the unprompted chair speak
        with its BASE weights -- adapters disabled -- so they are frozen by construction
        rather than by a promise not to touch a second copy."""
        self.cfg = cfg
        if backbone is None:
            from models_sr import SharedBackbone
            backbone = SharedBackbone(cfg)
        self.bb = backbone
        self.tokenizer = backbone.tokenizer
        self.greedy = False

    @property
    def model(self):
        return self.bb.model

    # ------------------------------------------------------------- lifecycle
    def reset(self, case):
        self.case = case
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
        self.addressed, self.pending = set(), {}
        self.cover_credit, self.leaks = {}, []
        self.settlement, self.settle_turn = {}, None
        self.n_calls, self.calls_by_role, self.prompt_chars = 0, {}, 0
        self.last_score = None

        self.conversation = [{'role': 'Meeting',
                              'content': 'The group convenes to decide: %s'
                                         % case['description']}]
        return self.conversation

    def snapshot(self):
        keys = ('ptr', 'utterances', 'step_i', 'revealed', 'reveal_turn',
                'reveal_elicited', 'addressed', 'pending', 'cover_credit', 'leaks',
                'settlement', 'settle_turn', 'conversation', 'n_calls', 'calls_by_role',
                'prompt_chars', 'last_score')
        return {k: copy.deepcopy(getattr(self, k)) for k in keys}

    def restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, copy.deepcopy(v))

    # ------------------------------------------------------------- policy interface
    def is_settling_turn(self):
        """The last chair slot must produce the settlement, or nothing is scoreable."""
        return self.step_i >= self.max_turn - 1

    def chair_prompt(self):
        """Messages for the policy. Chair's view only."""
        return P.chair_messages(self.case, self.conversation,
                                settling=self.is_settling_turn())

    def chair_budget(self):
        return (self.cfg.settlement_max_tokens if self.is_settling_turn()
                else self.cfg.agent_max_new_tokens)

    # ------------------------------------------------------------- the loop
    def step(self, chair_utterance):
        """Commit the chair's utterance, then let advisors speak until the chair's next
        slot. Returns (conversation, done). No reward: credit is assigned post-hoc."""
        text = chair_utterance.strip()
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

        # advisors speak until it is the chair's turn again, or the cap is hit
        while self.utterances < self.cap:
            speaker = self.order[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.order)
            if speaker == self.dm:
                break
            msgs = P.advisor_messages(self.case, self.conversation, speaker)
            resp = self._generate(msgs, self.names[speaker], self.cfg.agent_max_new_tokens)
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

    def peek(self, chair_utterance):
        """Try a candidate: commit it, let one advisor answer, report what it produced,
        then roll everything back. Used to score GRPO candidates without a reward model.

        Returns {'elicited_flips': set, 'disclosed': [fid], 'leaked': bool,
                 'settlement': dict|None}.
        """
        snap = self.snapshot()
        try:
            before = set(self.revealed)
            self.step(chair_utterance)
            got = self.revealed - before
            elicited = {f for f in got if self.reveal_elicited.get(f)}
            flips = set()
            for d in (self.case.get('decisive_facts') or []):
                if d['fact_id'] in elicited:
                    flips |= set(d.get('flips') or [])
            return {'elicited_flips': flips, 'disclosed': sorted(got),
                    'leaked': len(self.leaks) > len(snap['leaks']),
                    'settlement': dict(self.settlement) if self.settlement else None}
        finally:
            self.restore(snap)

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

    # ------------------------------------------------------------- scoring
    def _finalise(self):
        self.last_score = score(self.case, self.settlement, self.revealed, resolve=True)
        return self.last_score

    def episode(self):
        """The dict attribution.attribute() consumes."""
        return {'uid': self.case['uid'], 'dialog': self.dialog(),
                'settlement': self.settlement, 'revealed': sorted(self.revealed),
                'leaks': list(self.leaks)}

    def dialog(self):
        out = []
        for t in self.conversation:
            spk = 'env' if t['role'] == 'Meeting' else (
                'sys' if t['role'] == self.names[self.dm] else 'usr')
            out.append({'role': t['role'], 'content': t['content'], 'speaker': spk})
        return out

    def record(self, reward, done, turns):
        s = self.last_score or self._finalise()
        pub = {k: v for k, v in s.items() if k != 'settlement_resolved'}
        n_adv = max(1, len(self.advisors))
        return {'dialog': self.dialog(), 'reward': reward,
                'uid': self.case.get('uid'), 'domain': self.case.get('domain'),
                'num_agents': self.case.get('num_agents'),
                'scenario_type': self.case.get('scenario_type'),
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
                'cover_credit': {k: sorted(v) for k, v in self.cover_credit.items()},
                'cover': len(set().union(*self.cover_credit.values())) / n_adv
                         if self.cover_credit else 0.0}

    # ------------------------------------------------------------- generation
    def _generate(self, messages, speaker, max_new_tokens, n=1, temperature=None):
        """Always the BASE weights: every utterance the environment produces must be
        independent of whatever the policy adapter currently holds."""
        self.n_calls += n
        self.calls_by_role[speaker] = self.calls_by_role.get(speaker, 0) + n
        self.prompt_chars += sum(len(m.get('content') or '') for m in messages)

        if temperature is None:
            temperature = 0.0 if self.greedy else self.cfg.agent_temperature
        text = compat.render_chat(self.tokenizer, P.to_chat(messages, speaker))
        with self.bb.as_agent():
            outs, _comps, _p = self.bb.generate(
                text, n=n, max_new_tokens=max_new_tokens,
                temperature=temperature, greedy=not temperature)
        return outs if n > 1 else outs[0]

    def chair_say(self, n=1, temperature=None):
        """Sample chair utterance(s) from the FROZEN agent. Used for BC data collection;
        GRPO samples from the trained policy instead."""
        return self._generate(self.chair_prompt(), self.names[self.dm],
                              self.chair_budget(), n=n, temperature=temperature)

    def _extract_settlement(self):
        raw = self._generate(P.settlement_messages(self.case, self.conversation),
                             'extractor', self.cfg.settlement_max_tokens)
        return self._parse_json(raw)

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
        """Cut where the model starts speaking for somebody else.

        Stripping on a bare name would truncate 'Dr. Chen recommends dalbavancin' at the
        name, so the speaker LABEL -- name plus colon -- is what gets stripped.
        """
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
