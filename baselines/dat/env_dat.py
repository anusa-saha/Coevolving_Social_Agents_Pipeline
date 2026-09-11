"""The CSA meeting, with the chair's generation steered by dialogue action tokens.

Structurally this is the Sotopia-ToM environment: one frozen model plays every role, the
turn order walk, the view filtering, the disclosure threshold (0.35, frozen) and the
verifier are all the shared instrument. Two things are DAT's own:

  * before the chair speaks, the planner reads the last-token hidden state of the chair's
    own prompt and emits L prefix embeddings that are prepended to it. Nothing in the
    prompt text changes -- see prompts_dat.py.
  * the environment keeps the (state, action) pair for every chair turn, because the
    replay buffer stage 2 trains on is exactly that sequence plus the per-turn reward.

Three arms share this class, and they differ only in what the prefix is:

    unsteered   no prefix at all. The control, and the row DAT has to beat.
    selfclone   prefix from pi_phi alone (the RL head contributes zero). The paper's
                "w/ self-clone" row: it should land on top of `unsteered`, and if it
                does not, stage 1 has changed behaviour instead of preserving it.
    dat         prefix from pi_phi + pi_phi_rl. The trained arm.

Keeping all three in one class is not tidiness -- it is what makes them comparable. A
separate unsteered path would be free to differ in a detail nobody remembered to hold
fixed.
"""
import json
import re

import torch

import paths  # noqa: F401  -- puts the repo root and ppdpp/ on sys.path
from csa_core import detectors as D
from csa_core.verifier import floor_score, score
import prompts_dat as P
import reward_dat as R
from steering import SteeredLM

try:
    import nltk
    _SENT = nltk.sent_tokenize
except Exception:                                    # noqa: BLE001
    _SENT = None

ARMS = ('unsteered', 'selfclone', 'dat')


class DATEnv(object):
    def __init__(self, cfg, lm=None, planner=None):
        self.cfg = cfg
        self.lm = lm if lm is not None else SteeredLM(cfg)
        self.planner = planner
        self.reward_fn = R.build(cfg)
        self.greedy = True                           # evaluation is deterministic
        self.collect_clone_pairs = False             # stage 1 corpus collection

    # ------------------------------------------------------------- lifecycle
    def reset(self, case, arm='dat'):
        assert arm in ARMS, arm
        if arm != 'unsteered' and self.planner is None:
            raise SystemExit('arm %r needs a planner; pass one to DATEnv' % arm)
        self.case = case
        self.arm = arm
        self.dm = case['decision_maker']
        self.order = list(case['interaction_config']['turn_order'])
        self.ptr = 0
        self.utterances = 0
        self.cap = int(case['interaction_config']['turn_cap'])
        dm_slots = sum(1 for i in range(self.cap)
                       if self.order[i % len(self.order)] == self.dm)
        self.max_turn = max(1, dm_slots)

        self.names = {a['agent_id']: a['name'] for a in case['agents']}
        self.advisors = set(self.names) - {self.dm}

        self.step_i = 0
        self.revealed, self.reveal_turn, self.reveal_elicited = set(), {}, {}
        self.addressed, self.pending, self.cover_credit = set(), {}, {}
        self.leaks = []
        self.settlement, self.settle_turn = {}, None
        self.n_calls, self.calls_by_role, self.prompt_chars = 0, {}, 0
        self.planner_forwards = 0
        self.last_score = self.last_score_norm = None

        # what stage 2 reads back out
        self.states, self.actions, self.residuals = [], [], []
        self.derived_acts = []
        self.clone_pairs = []

        self.conversation = [{'role': 'Meeting',
                              'content': 'The group convenes to decide: %s'
                                         % case['description']}]
        return self.conversation

    def is_settling_turn(self):
        return self.step_i >= self.max_turn - 1

    # ------------------------------------------------------------- the loop
    def step(self, residual=None, sigma=0.0, generator=None):
        """One chair turn plus the advisor replies that follow it.

        `residual` overrides pi_phi_rl for this turn and `sigma` adds exploration noise
        on top -- that pair is how collect_buffer.py perturbs the self-cloned action, and
        it is the only way an action enters the buffer.

        Returns (conversation, done).
        """
        settling = self.is_settling_turn()
        msgs = P.chair_messages(self.case, self.conversation, settling=settling)
        budget = (self.cfg.settlement_max_tokens if settling
                  else self.cfg.max_new_tokens)

        ids = self.lm.encode(msgs, self.names[self.dm])
        prefix, action, res = None, None, None
        if self.arm != 'unsteered':
            s = self.lm.state(ids)
            self.planner_forwards += 1
            res = self._residual_for(s, residual, sigma, generator)
            # No gradient anywhere in a rollout: the planner is optimised offline, from
            # the buffer, and building a graph here would only hold activations alive.
            with torch.no_grad():
                action = self.planner.action(s.to(self._pdev()), residual=res)
                prefix = self.planner.prefix(action)
            self.states.append(s.cpu())
            self.actions.append(action.detach().cpu())
            self.residuals.append(res.detach().cpu())

        raw = self._generate(ids, msgs, budget, prefix, self.names[self.dm])
        text = raw if (settling or self._parse_json(raw)) else self._trim_for(raw, self.dm)
        if self.collect_clone_pairs and not settling:
            # The clone target is what the UNSTEERED policy actually emitted, before
            # trimming: Equation 5 fits the model's own distribution, not our clean-up of
            # it. Settling turns are excluded -- a 512-token JSON would dominate the loss.
            self.clone_pairs.append({'prompt': self.lm.render(msgs, self.names[self.dm]),
                                     'target': raw, 'uid': self.case.get('uid'),
                                     'turn': self.step_i})

        eliciting = D.is_eliciting(text, self.case['agents'], self.dm)
        self.derived_acts.append('decide' if settling else
                                 ('elicit' if eliciting else 'other'))
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
            a_msgs = P.advisor_messages(self.case, self.conversation, speaker)
            a_ids = self.lm.encode(a_msgs, self.names[speaker])
            resp = self._trim_for(
                self._generate(a_ids, a_msgs, self.cfg.max_new_tokens, None,
                               self.names[speaker]), speaker)
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

    def _residual_for(self, s, residual, sigma, generator):
        """pi_phi_rl(s) for the `dat` arm, exactly zero for `selfclone`.

        The zero is what makes `selfclone` the paper's "w/ self-clone" row rather than an
        untrained-RL row, and it is also what collect_buffer.py perturbs around: the
        buffer has to hold actions near the self-cloned policy, not near whatever the
        RL head currently is.
        """
        dev = self._pdev()
        if residual is None:
            if self.arm == 'selfclone':
                residual = torch.zeros(self.planner.action_dim, device=dev)
            else:
                with torch.no_grad():
                    residual = self.planner.rl_action(s.to(dev))
        residual = torch.as_tensor(residual, dtype=torch.float32, device=dev)
        if sigma:
            noise = torch.randn(residual.shape, generator=generator,
                                dtype=torch.float32)
            residual = residual + sigma * noise.to(dev)
        return residual

    def _pdev(self):
        return next(self.planner.parameters()).device

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
        self.last_score = score(self.case, self.settlement, self.revealed,
                                resolve=self.cfg.resolve_provenance)
        self.last_score_norm = score(self.case, self.settlement, self.revealed,
                                     norm=True, resolve=self.cfg.resolve_provenance)
        return self.last_score

    # ------------------------------------------------------------- rewards
    def trace(self):
        s = self.last_score or self._finalise()
        return R.Trace(n_turns=self.step_i,
                       reveal_turn=dict(self.reveal_turn),
                       reveal_elicited=dict(self.reveal_elicited),
                       decisive=self.case.get('decisive_facts') or [],
                       settle_turn=self.settle_turn,
                       dca=s['dca'], schema_valid=s['schema_valid'],
                       leaks=list(self.leaks), acts=list(self.derived_acts),
                       conversation=list(self.conversation), case=self.case)

    def terminal_reward(self):
        return R.terminal_reward(self.cfg, self.last_score or self._finalise(),
                                 self.leaks)

    def turn_rewards(self, w_outcome=None):
        """r_1..r_T, with the episode outcome folded into the last turn."""
        r = self.reward_fn(self.trace())
        w = self.cfg.w_outcome if w_outcome is None else w_outcome
        if w and r:
            r[-1] += w * self.terminal_reward()
        return r

    def transitions(self, w_outcome=None):
        """(s, u, r, s', done) per chair turn, for the offline buffer.

        s' at the last turn is a zero vector with done=1: the episode ends when the turn
        cap is reached, so there is no successor state to bootstrap from and the critic
        must not invent one.
        """
        if not self.states:
            return []
        r = self.turn_rewards(w_outcome)
        out, T = [], len(self.states)
        for t in range(T):
            s2 = self.states[t + 1] if t + 1 < T else torch.zeros_like(self.states[t])
            out.append({'s': self.states[t].numpy(),
                        'u': self.residuals[t].numpy(),
                        'r': float(r[t]) if t < len(r) else 0.0,
                        's2': s2.numpy(),
                        'done': 1.0 if t == T - 1 else 0.0})
        return out

    # ------------------------------------------------------------- output
    def dialog(self):
        out = []
        for t in self.conversation:
            spk = 'env' if t['role'] == 'Meeting' else (
                'sys' if t['role'] == self.names[self.dm] else 'usr')
            out.append({'role': t['role'], 'content': t['content'], 'speaker': spk})
        return out

    def record(self, turns):
        """Same schema as every other arm's record, so compute_all_metrics.py and
        analysis/compute_extended_metrics.py read a DAT record with no branch.

        `act_history` is deliberately ABSENT. DAT's action is a vector in R^64: there is
        no discrete act it chose, and putting the detector's guess in that field would
        put a measurement next to PPDPP's and EPO's planner decisions and invite them to
        be read as the same thing. The guess is kept, under `derived_acts`, where it
        reads as what it is.
        """
        s = self.last_score or self._finalise()
        pub = {k: v for k, v in s.items() if k != 'settlement_resolved'}
        pubn = {k: v for k, v in (self.last_score_norm or {}).items()
                if k != 'settlement_resolved'}
        n_adv = max(1, len(self.advisors))
        done = 1 if (s['schema_valid'] and not self.leaks
                     and s['dca'] >= self.cfg.done_tau) else -1
        acts = [a.tolist() for a in self.actions]
        return {
            'dialog': self.dialog(), 'reward': s['dca'],
            'uid': self.case.get('uid'), 'domain': self.case.get('domain'),
            'num_agents': self.case.get('num_agents'),
            'scenario_type': self.case.get('scenario_type'),
            'arm': self.arm,
            'settlement': self.settlement, 'score': pub, 'score_norm': pubn,
            'floor': {k: v for k, v in floor_score(self.case).items()
                      if k != 'settlement_resolved'},
            'revealed': sorted(self.revealed),
            'reveal_elicited': dict(self.reveal_elicited),
            'reveal_turn': dict(self.reveal_turn),
            'addressed': sorted(self.addressed), 'leaks': list(self.leaks),
            'done': done, 'turns': turns, 'max_turn': self.max_turn,
            'n_calls': self.n_calls, 'calls_by_role': dict(self.calls_by_role),
            'prompt_chars': self.prompt_chars,
            'cover': (len(set().union(*self.cover_credit.values())) / n_adv
                      if self.cover_credit else 0.0),
            # DAT diagnostics
            'derived_acts': list(self.derived_acts),
            'planner_forwards': self.planner_forwards,
            'turn_rewards': self.turn_rewards(),
            'terminal_reward': self.terminal_reward(),
            'action_norms': [float(torch.linalg.vector_norm(a)) for a in self.actions],
            'residual_norms': [float(torch.linalg.vector_norm(u))
                               for u in self.residuals],
            'action_cos': _mean_consecutive_cos(self.actions),
            'actions': acts if getattr(self.cfg, 'store_actions', False) else None,
        }

    # ------------------------------------------------------------- generation
    def _generate(self, input_ids, messages, max_new_tokens, prefix, role):
        self.n_calls += 1
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        self.prompt_chars += sum(len(m.get('content') or '') for m in messages)
        return self.lm.generate(input_ids, max_new_tokens, prefix=prefix,
                                greedy=self.greedy, temperature=self.cfg.temperature)

    def _extract_settlement(self):
        msgs = P.settlement_messages(self.case, self.conversation)
        ids = self.lm.encode(msgs, 'extractor')
        return self._parse_json(
            self._generate(ids, msgs, self.cfg.settlement_max_tokens, None,
                           'extractor'))

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


def _mean_consecutive_cos(actions):
    """How much the planner's action moves between turns.

    A planner that has collapsed onto one constant vector still steers -- it just steers
    identically everywhere, which is a static prefix rather than a policy. That failure
    is invisible in the outcome metrics and obvious here.

    None, never float('nan'), when there is nothing to compare -- a one-turn episode, or
    the unsteered arm, which has no actions at all. Records are stored as repr() and read
    back with ast.literal_eval, and a bare `nan` is not a literal: it raises, the loader
    skips the block, and the whole arm silently disappears from the metrics tables.
    """
    if len(actions) < 2:
        return None
    cos = torch.nn.functional.cosine_similarity
    vals = [float(cos(a.unsqueeze(0), b.unsqueeze(0)))
            for a, b in zip(actions, actions[1:])]
    return sum(vals) / len(vals)
