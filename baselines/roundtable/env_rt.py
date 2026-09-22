"""The round table: N agents talk in turn order, then converge on one settlement.

This is the method-free floor. There is no planner choosing acts, no strategist emitting
instructions, no stall detector, no reasoning scaffold, and nothing is trained. Each agent
sees the transcript and its own view, says one turn, and the table goes round again.

The only design choice is how N agents become ONE answer, and there are three, because
they measure different things:

  chair     the decision_maker writes the settlement, exactly as every other arm does.
            This is the row that belongs in the comparison table.
  vote      every agent proposes a settlement; each field is decided by majority across
            the proposals. Tests whether the table as a whole knows more than its chair.
  converge  the chair proposes, everyone else agrees or returns a correction, and the
            last correction wins. Closest to "they reach a common conclusion", and the
            most expensive.

Records use the same schema as every other arm, so analysis/compute_extended_metrics.py
picks this arm up with no changes.
"""
import collections
import json
import re

import _detectors as D
from _verifier import floor_score, score

import prompts_rt as P

_JSON = re.compile(r'\{.*\}', re.S)


class RoundTableEnv(object):
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.backend = backend

    # ------------------------------------------------------------- lifecycle
    def reset(self, case):
        self.case = case
        self.dm = case['decision_maker']
        self.order = list(case['interaction_config']['turn_order'])
        self.names = {a['agent_id']: a['name'] for a in case['agents']}
        self.ids = {a['name']: a['agent_id'] for a in case['agents']}
        self.advisors = set(self.names) - {self.dm}

        self.conversation = []
        self.step_i = 0
        self.n_calls = 0
        self.calls_by_role = {}
        self.prompt_chars = 0

        self.revealed, self.reveal_turn, self.reveal_elicited = set(), {}, {}
        self.addressed, self.pending, self.cover_credit = set(), {}, {}
        self.leaks = []
        self.settlement = None
        self.settle_turn = -1
        self.settled_by = None
        self._extracted = False                  # the extraction fallback ran
        self.proposals = {}
        self.agreed = {}
        self.last_score = None
        # Chair slots, so max_turn means the same thing here as in the other arms.
        cap = int(case['interaction_config']['turn_cap'])
        self.max_turn = max(1, sum(1 for i in range(cap)
                                   if self.order[i % len(self.order)] == self.dm))

    # ------------------------------------------------------------- detectors
    def _note(self, speaker, utterance, eliciting):
        """Record disclosures, leaks and who was addressed. Identical bookkeeping to the
        other arms -- it is the measurement, not the method."""
        for fid in D.disclosures(self.case, speaker, utterance, self.revealed,
                                 self.cfg.reveal_threshold):
            self.revealed.add(fid)
            self.reveal_turn[fid] = self.step_i
            self.reveal_elicited[fid] = bool(eliciting)
        for fid in D.leaks(self.case, speaker, utterance, self.revealed,
                           self.cfg.reveal_threshold):
            self.leaks.append({'fact': fid, 'by': speaker, 'turn': self.step_i})
        if speaker in self.pending:
            self.cover_credit.setdefault(self.pending.pop(speaker), set()).add(speaker)

    def _speak(self, agent_id, settling=False, proposal=None, tokens=None):
        msgs = P.agent_messages(self.case, self.conversation, agent_id,
                                settling=settling, proposal=proposal)
        self.prompt_chars += sum(len(m.get('content') or '') for m in msgs)
        self.calls_by_role[agent_id] = self.calls_by_role.get(agent_id, 0) + 1
        self.n_calls += 1
        text = self.backend(msgs, self.names[agent_id],
                            tokens or self.cfg.max_new_tokens,
                            temperature=self.cfg.temperature)
        return (text or '').strip()

    # ------------------------------------------------------------- the loop
    def run(self, case):
        self.reset(case)
        for _ in range(self.cfg.rounds):
            for agent_id in self.order[:len(self.names)]:
                text = self._speak(agent_id)
                if not text:
                    continue
                eliciting = D.is_eliciting(text, self.case['agents'], agent_id)
                fresh = D.addressed_in(text, self.case['agents'],
                                       exclude={agent_id}) - self.addressed
                self.addressed |= fresh
                for aid in fresh:
                    self.pending[aid] = self.step_i
                self.conversation.append({'role': self.names[agent_id],
                                          'content': text,
                                          'speaker': 'sys' if agent_id == self.dm
                                                     else 'usr'})
                self._note(agent_id, text, eliciting)
                if agent_id == self.dm:
                    self.step_i = min(self.step_i + 1, self.max_turn - 1)

        self.settlement = self._converge()
        self.settle_turn = self.step_i
        if self.settlement:
            self.settled_by = 'extractor' if self._extracted else self.cfg.decide
        self._finalise()
        return self.record()

    # ------------------------------------------------------------- consensus
    def _parse(self, text):
        m = _JSON.search(text or '')
        if not m:
            return None
        try:
            got = json.loads(m.group(0))
        except Exception:                            # noqa: BLE001
            return None
        return got if isinstance(got, dict) else None

    def _converge(self):
        mode = self.cfg.decide
        if mode == 'chair':
            return self._settle_chair()
        if mode == 'vote':
            return self._settle_vote()
        if mode == 'converge':
            return self._settle_converge()
        raise ValueError('unknown --decide %r; choose from %s' % (mode, P.DECIDERS))

    def _settle_chair(self):
        raw = self._speak(self.dm, settling=True, tokens=self.cfg.settlement_tokens)
        self.conversation.append({'role': self.names[self.dm], 'content': raw,
                                  'speaker': 'sys'})
        return self._parse(raw) or self._extract()

    def _settle_vote(self):
        """Every agent proposes; each decision field goes to the majority.

        Ties break towards the chair's own proposal, so the arm degrades to `chair`
        rather than to an arbitrary agent when the table is split.
        """
        for agent_id in self.order[:len(self.names)]:
            raw = self._speak(agent_id, settling=True, tokens=self.cfg.settlement_tokens)
            got = self._parse(raw)
            if got:
                self.proposals[agent_id] = got

        if not self.proposals:
            return self._extract()
        chair = self.proposals.get(self.dm, {})
        fields = (self.case['settlement_schema'].get('decisions') or {})
        decisions = {}
        for field in fields:
            votes = collections.Counter()
            for aid, prop in self.proposals.items():
                v = (prop.get('decisions') or {}).get(field)
                if v not in (None, ''):
                    votes[json.dumps(v, sort_keys=True)] += 1
            if not votes:
                continue
            top = max(votes.values())
            tied = [k for k, c in votes.items() if c == top]
            pick = tied[0]
            if len(tied) > 1:
                own = (chair.get('decisions') or {}).get(field)
                if own not in (None, ''):
                    key = json.dumps(own, sort_keys=True)
                    if key in tied:
                        pick = key
            decisions[field] = json.loads(pick)

        # Fact lists are a UNION, not a vote: a fact any agent relied on was in fact
        # relied on, and majority-voting them would discard genuine provenance.
        merged = {'decisions': decisions}
        for key in ('credited_facts', 'justification_fact_ids', 'revealed'):
            seen = []
            for prop in self.proposals.values():
                for f in (prop.get(key) or []):
                    if f not in seen:
                        seen.append(f)
            if seen:
                merged[key] = seen
        commits = []
        for prop in self.proposals.values():
            for c in (prop.get('commitments') or []):
                if c not in commits:
                    commits.append(c)
        if commits:
            merged['commitments'] = commits
        return merged

    def _settle_converge(self):
        """Chair proposes; the others agree or correct. The last correction wins."""
        raw = self._speak(self.dm, settling=True, tokens=self.cfg.settlement_tokens)
        self.conversation.append({'role': self.names[self.dm], 'content': raw,
                                  'speaker': 'sys'})
        current = self._parse(raw) or self._extract()
        if not current:
            return None

        for _ in range(max(1, self.cfg.converge_rounds)):
            changed = False
            for agent_id in self.order[:len(self.names)]:
                if agent_id == self.dm:
                    continue
                reply = self._speak(agent_id, proposal=current,
                                    tokens=self.cfg.settlement_tokens)
                if reply.strip().upper().startswith('AGREE'):
                    self.agreed[agent_id] = True
                    continue
                got = self._parse(reply)
                if got:
                    self.agreed[agent_id] = False
                    current = got
                    changed = True
            if not changed:
                break                                # everyone agreed; stop early
        return current

    def _extract(self):
        """Fallback when nobody emitted parseable JSON."""
        self._extracted = True
        msgs = P.settlement_messages(self.case, self.conversation)
        self.prompt_chars += sum(len(m.get('content') or '') for m in msgs)
        self.calls_by_role['extract'] = self.calls_by_role.get('extract', 0) + 1
        self.n_calls += 1
        raw = self.backend(msgs, 'extract', self.cfg.settlement_tokens, temperature=0.0)
        return self._parse(raw)

    # ------------------------------------------------------------- scoring
    def _finalise(self):
        self.last_score = score(self.case, self.settlement or {}, self.revealed)
        return self.last_score

    def record(self):
        s = self.last_score or self._finalise()
        pub = {k: v for k, v in s.items() if k != 'settlement_resolved'}
        n_adv = max(1, len(self.advisors))
        done = 1 if (s['schema_valid'] and not self.leaks
                     and s['dca'] >= self.cfg.done_tau) else -1
        return {'dialog': list(self.conversation), 'reward': s['dca'],
                'uid': self.case.get('uid'), 'domain': self.case.get('domain'),
                'num_agents': self.case.get('num_agents'),
                'scenario_type': self.case.get('scenario_type'),
                'decide': self.cfg.decide, 'backend': self.cfg.backend,
                'settlement': self.settlement, 'score': pub,
                'settle_turn': self.settle_turn, 'settled_by': self.settled_by,
                'floor': {k: v for k, v in floor_score(self.case).items()
                          if k != 'settlement_resolved'},
                'revealed': sorted(self.revealed),
                'reveal_elicited': dict(self.reveal_elicited),
                'reveal_turn': dict(self.reveal_turn),
                'addressed': sorted(self.addressed), 'leaks': list(self.leaks),
                'done': done, 'turns': len(self.conversation),
                'max_turn': self.max_turn,
                'n_calls': self.n_calls, 'calls_by_role': dict(self.calls_by_role),
                'prompt_chars': self.prompt_chars,
                'n_proposals': len(self.proposals),
                'agreed': dict(self.agreed),
                'cover': len(set().union(*self.cover_credit.values())) / n_adv
                         if self.cover_credit else 0.0}
