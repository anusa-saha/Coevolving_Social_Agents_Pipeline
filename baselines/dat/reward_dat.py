"""The reward signal. Torch-free, model-free, deterministic.

DAT's reward is whatever the environment supplies -- Sotopia's prompted GPT-4 in the
social-capability experiment, a fine-tuned judge's Yes/No logit gap in the red-teaming
one. Its limitations section names this as the real constraint: "our work builds on the
assumption that we have access to cheap, stable reward signals".

CSA removes the assumption. `decisive_facts` states, per private fact, exactly which
executable checks it flips, and the environment already records which turn each fact
surfaced on and whether the chair drew it out. So criticality is a lookup and the reward
is byte-for-byte reproducible with no judge model anywhere in the arm.

This is deliberately the SAME rule EPO's verifier PRM applies (epo/prm.py, VerifierPRM):
two RL arms supervised by different reward functions are not comparable, and the whole
point of the repo is that a difference between two rows is a difference between two
methods. selftest.py imports EPO's implementation and asserts the two agree on random
traces; if they ever drift, that check fails loudly instead of the numbers diverging
quietly.

What DAT adds on top of the per-turn signal is the terminal outcome, with the same shape
as EPO's `terminal_reward`. TD3+BC is an offline actor-critic over ~1200 transitions: an
episode that returns all zeros contributes a Q-target of zero and is indistinguishable
from one that actively wasted its turns, so the negative branch has to exist.
"""


class Trace(object):
    """Everything the reward needs from one episode. Built by DATEnv.trace().

    Field names match epo/prm.py's Trace exactly, so EPO's VerifierPRM can score a DAT
    trace unchanged -- which is what makes the agreement check in selftest.py possible.
    """

    def __init__(self, n_turns, reveal_turn, reveal_elicited, decisive, settle_turn,
                 dca, schema_valid, leaks, acts, conversation, case):
        self.n_turns = n_turns
        self.reveal_turn = reveal_turn
        self.reveal_elicited = reveal_elicited
        self.decisive = decisive
        self.settle_turn = settle_turn
        self.dca = dca
        self.schema_valid = schema_valid
        self.leaks = leaks
        self.acts = acts
        self.conversation = conversation
        self.case = case

    @property
    def valid(self):
        return bool(self.schema_valid) and not self.leaks


class VerifierReward(object):
    """r_t from the dataset's own decisive-fact structure.

    binary  -- r_t in {0, 1}, the shape EPO's judge produces
    graded  -- r_t weighted by how many checks the disclosed fact flips, and the settle
               term continuous in dca rather than thresholded

    `graded` is the default here where EPO defaults to `binary`, for a reason specific to
    this arm: TD3+BC bootstraps a Q function from ~1200 transitions, and a signal that is
    zero on almost every one of them gives the critic nothing to fit. EPO's REINFORCE
    tolerates that better because it only needs the advantage's sign.
    """

    def __init__(self, mode='graded', done_tau=0.6, lam=1.0, leak_invalidates=True):
        assert mode in ('binary', 'graded')
        self.mode = mode
        self.done_tau = done_tau
        self.lam = lam
        self.leak_invalidates = leak_invalidates

    def __call__(self, tr):
        r = [0.0] * tr.n_turns
        flips = {d['fact_id']: list(d.get('flips') or []) for d in (tr.decisive or [])}
        universe = set()
        for f in flips.values():
            universe.update(f)
        n_phi = max(1, len(universe))

        gated = tr.schema_valid and not (tr.leaks and self.leak_invalidates)

        for fid, turn in (tr.reveal_turn or {}).items():
            if fid not in flips:
                continue                             # disclosed, but not decisive
            if not tr.reveal_elicited.get(fid):
                continue                             # volunteered: the chair earned it not
            if not (0 <= turn < tr.n_turns):
                continue
            if self.mode == 'binary':
                r[turn] = 1.0
            else:
                r[turn] += len(flips[fid]) / n_phi

        if tr.settle_turn is not None and 0 <= tr.settle_turn < tr.n_turns and gated:
            if self.mode == 'binary':
                if tr.dca >= self.done_tau:
                    r[tr.settle_turn] = 1.0
            else:
                r[tr.settle_turn] += self.lam * float(tr.dca)
        return r


def terminal_reward(cfg, score, leaks):
    """The episode's outcome, in [-1, 1]. Same shape as EPO's env.terminal_reward().

    Kept separate from the per-turn signal so `--w_outcome 0` reproduces a process-only
    reward, which is what vanilla DAT's per-round judge would give.
    """
    if not score['schema_valid']:
        return -1.0
    if bool(leaks) and cfg.leak_invalidates:
        return -1.0
    pool, use, close = score['disclosure_rate'], score['dca'], score['close']
    if pool == 0.0 and use == 0.0:
        return -0.5
    a, b, g = cfg.w_use, cfg.w_pool, cfg.w_close
    tot = (a + b + g) or 1.0
    r = 2.0 * ((a * use + b * pool + g * close) / tot) - 1.0
    r -= cfg.w_halluc_pen * score['hallucinated_credit']
    return max(-1.0, min(1.0, r))


def build(cfg):
    return VerifierReward(mode=cfg.reward_mode, done_tau=cfg.done_tau,
                          leak_invalidates=cfg.leak_invalidates)
