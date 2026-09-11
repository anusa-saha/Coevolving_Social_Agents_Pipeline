"""The CSA meeting with Omega's stall detection and fast/slow mode switching.

Omega detects deadlock with the expert rating itself: a `goal_c` (current) and `goal_p`
(predicted) score, each 0-10, sampled five times and averaged, feeding three threshold
intervals that pick native / simple / negotiation strategy. Rating starts from the sixth
turn, where the paper reports deadlocks typically begin. (The paper states the mechanism
and the intervals but not their numeric cut-points; do not quote a specific threshold
from here.)

CSA does not need a judge: the thing that should be happening is decisive facts reaching
the chair, and whether that happened is a set membership test.

    stalled  <=>  step >= stall_after  and  (revealed & decisive) has not grown
                  for `stall_patience` chair turns

Deterministic, zero calls, no threshold to tune. Once stalled, the chair runs the three
scaffolding stages before speaking, and stays in slow mode for the rest of the episode --
matching Omega, whose agent does not revert once the negotiation strategy is selected.

Three calls, four protocol steps: stages 1-3 are internal reasoning, and the fourth step
is the chair's actual utterance rather than a fourth scaffolding call.

Only the final utterance enters the transcript. The stage outputs are recorded on the
episode for inspection but never shown to the advisors and never used as SFT labels.
"""
import json
import re

import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import detectors as D
import prompts_om as P
from csa_core.verifier import floor_score, score

try:
    import nltk
    _SENT = nltk.sent_tokenize
except Exception:                                    # noqa: BLE001
    _SENT = None


class OmegaEnv(object):
    def __init__(self, cfg, expert):
        self.cfg = cfg
        self.expert = expert

    # ------------------------------------------------------------- lifecycle
    def reset(self, case, opponent=None, force_mode=None):
        """`opponent` is an advisor agent_id put into withholding mode, or None.
        `force_mode` pins 'fast' or 'slow' for the probe; None means adaptive."""
        self.case = case
        self.opponent = opponent
        self.force_mode = force_mode
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
        self.leaks, self.opponent_leaks = [], []
        self.settlement, self.settle_turn = {}, None
        self.n_calls, self.calls_by_role, self.prompt_chars = 0, {}, 0
        self.last_score = None

        self.mode = force_mode or 'fast'
        self.modes = []                              # per chair turn, Omega's `difference`
        self.plans = []                              # slow-mode scaffolding, kept aside
        self.stalled_at = None
        self._decisive_seen = 0
        self._flat_for = 0

        self.conversation = [{'role': 'Meeting',
                              'content': 'The group convenes to decide: %s'
                                         % case['description']}]
        return self.conversation

    def is_settling_turn(self):
        return self.step_i >= self.max_turn - 1

    # ------------------------------------------------------------- stall
    def decisive_pooled(self):
        dec = {d['fact_id'] for d in (self.case.get('decisive_facts') or [])}
        return len(self.revealed & dec)

    def _update_stall(self):
        """Computable deadlock: decisive pooling has not advanced for N chair turns."""
        now = self.decisive_pooled()
        self._flat_for = 0 if now > self._decisive_seen else self._flat_for + 1
        self._decisive_seen = max(self._decisive_seen, now)
        if self.force_mode:
            return
        if (self.mode == 'fast' and self.step_i >= self.cfg.stall_after
                and self._flat_for >= self.cfg.stall_patience):
            self.mode = 'slow'                       # Omega does not revert once hard
            self.stalled_at = self.step_i

    # ------------------------------------------------------------- slow mode
    def _make_plan(self):
        """Three scaffolding calls. Their output never enters the transcript."""
        c, conv = self.case, self.conversation
        s1 = self._gen(P.stage_messages(c, conv, 1), 'planner', self.cfg.stage_tokens)
        s2 = self._gen(P.stage_messages(c, conv, 2, unsupported=s1), 'planner',
                       self.cfg.stage_tokens)
        s3 = self._gen(P.stage_messages(c, conv, 3, unsupported=s1, holders=s2), 'planner',
                       self.cfg.stage_tokens)
        plan = P.format_plan(s1, s2, s3)
        self.plans.append({'turn': self.step_i, 'unsupported': s1[:600],
                           'holders': s2[:600], 'question': s3[:300]})
        return plan

    # ------------------------------------------------------------- the loop
    def step(self):
        self._update_stall()
        plan = self._make_plan() if self.mode == 'slow' else ''
        self.modes.append(self.mode)

        msgs = P.chair_messages(self.case, self.conversation,
                                settling=self.is_settling_turn(), plan=plan)
        budget = (self.cfg.settlement_tokens if self.is_settling_turn()
                  else self.cfg.max_new_tokens)
        text = self._gen(msgs, self.names[self.dm], budget,
                         temperature=self.cfg.temperature)
        if not (self.is_settling_turn() or self._parse_json(text)):
            text = self._trim_for(text, self.dm)

        eliciting = D.is_eliciting(text, self.case['agents'], self.dm)
        fresh = D.addressed_in(text, self.case['agents'], exclude={self.dm}) - self.addressed
        self.addressed |= fresh
        for aid in fresh:
            self.pending[aid] = self.step_i
        self._note_leak(self.dm, text)

        self.conversation.append({'role': self.names[self.dm], 'content': text})
        self.utterances += 1
        self.ptr = (self.order.index(self.dm) + 1) % len(self.order)

        parsed = self._parse_json(text)
        if parsed:
            self.settlement, self.settle_turn = parsed, self.step_i

        while self.utterances < self.cap:
            speaker = self.order[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.order)
            if speaker == self.dm:
                break
            resp = self._gen(P.advisor_messages(self.case, self.conversation, speaker,
                                                opponent=self.opponent),
                             self.names[speaker], self.cfg.max_new_tokens,
                             temperature=self.cfg.temperature)
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
        self._note_leak(speaker, utterance)

    def _note_leak(self, speaker, utterance):
        """A designated opponent is PLAYING a role, so its leaks are recorded separately
        and never invalidate the episode. Without this the adversarial arm would poison
        its own reward: scripted evasion that happens to echo an unseen fact would trip
        the integrity gate and zero an otherwise valid episode."""
        found = D.leaks(self.case, speaker, utterance, self.revealed,
                        self.cfg.reveal_threshold)
        bucket = (self.opponent_leaks if speaker == self.opponent else self.leaks)
        for fid in found:
            bucket.append({'fact': fid, 'by': speaker, 'turn': self.step_i})

    # ------------------------------------------------------------- scoring
    def ceiling(self):
        """Highest dca reachable given who is withholding.

        With an opponent silenced, the checks its facts control may be unreachable, so a
        raw score mixes policy skill with which advisor was muted. Reporting dca/ceiling
        keeps the number interpretable -- and it is computable, because decisive_facts
        names both the owner and the checks each fact flips.
        """
        flips = {d['fact_id']: set(d.get('flips') or [])
                 for d in (self.case.get('decisive_facts') or [])}
        phi = set().union(*flips.values()) if flips else set()
        if not phi or not self.opponent:
            return 1.0
        lost = set()
        for fid, f in self.case['private_facts'].items():
            if f['owner'] == self.opponent and fid in flips:
                lost |= flips[fid]
        return len(phi - lost) / len(phi)

    def _finalise(self):
        self.last_score = score(self.case, self.settlement, self.revealed, resolve=True)
        return self.last_score

    def episode(self):
        s = self.last_score or self._finalise()
        pub = {k: v for k, v in s.items() if k != 'settlement_resolved'}
        n_adv = max(1, len(self.advisors))
        ceil = self.ceiling()
        return {
            'uid': self.case['uid'], 'domain': self.case.get('domain'),
            'num_agents': self.case.get('num_agents'),
            'scenario_type': self.case.get('scenario_type'),
            'dialog': self.dialog(), 'settlement': self.settlement,
            'score': pub, 'reward': s['dca'],
            'floor': {k: v for k, v in floor_score(self.case).items()
                      if k != 'settlement_resolved'},
            'ceiling': ceil,
            'dca_norm': (s['dca'] / ceil) if ceil > 0 else 0.0,
            'revealed': sorted(self.revealed),
            'reveal_elicited': dict(self.reveal_elicited),
            'reveal_turn': dict(self.reveal_turn),
            'addressed': sorted(self.addressed),
            'leaks': list(self.leaks), 'opponent_leaks': list(self.opponent_leaks),
            'opponent': self.opponent,
            'modes': list(self.modes), 'stalled_at': self.stalled_at,
            'plans': list(self.plans),
            'done': 1 if (s['schema_valid'] and not self.leaks
                          and s['dca'] >= self.cfg.done_tau) else -1,
            'turns': self.step_i, 'max_turn': self.max_turn,
            'n_calls': self.n_calls, 'calls_by_role': dict(self.calls_by_role),
            'prompt_chars': self.prompt_chars,
            'cover': len(set().union(*self.cover_credit.values())) / n_adv
                     if self.cover_credit else 0.0,
        }

    def dialog(self):
        out = []
        for t in self.conversation:
            spk = 'env' if t['role'] == 'Meeting' else (
                'sys' if t['role'] == self.names[self.dm] else 'usr')
            out.append({'role': t['role'], 'content': t['content'], 'speaker': spk})
        return out

    # ------------------------------------------------------------- generation
    def _gen(self, messages, speaker, max_new_tokens, temperature=None):
        self.n_calls += 1
        self.calls_by_role[speaker] = self.calls_by_role.get(speaker, 0) + 1
        self.prompt_chars += sum(len(m.get('content') or '') for m in messages)
        return self.expert(messages, speaker, max_new_tokens, temperature)

    def _extract_settlement(self):
        return self._parse_json(
            self._gen(P.settlement_messages(self.case, self.conversation),
                      'extractor', self.cfg.settlement_tokens))

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
