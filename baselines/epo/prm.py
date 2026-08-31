"""Process reward models. Torch-free.

EPO's process reward is a frozen GPT-4o reading the finished trajectory and naming the
turns whose strategies were critical (r_t = 1 for those, 0 elsewhere). That costs a call
per episode, drifts with the judge, and on SOTOPIA there is no way to check it.

CSA ships `decisive_facts`, which states for every private fact exactly which checks it
flips. Criticality is therefore a lookup, not a judgement, and the environment already
records `reveal_turn` and `reveal_elicited` per turn. VerifierPRM computes what
JudgePRM guesses: zero model calls, byte-for-byte reproducible.

Both are shipped. JudgePRM reproduces EPO faithfully; running both on the same
trajectories gives an agreement statistic no existing EPO environment can produce.
"""
import json
import os
import re


class Trace(object):
    """Everything a PRM needs from one episode. Built by EPOEnv.trace()."""

    def __init__(self, n_turns, reveal_turn, reveal_elicited, decisive,
                 settle_turn, dca, schema_valid, leaks, acts, conversation, case):
        self.n_turns = n_turns                  # number of strategist actions, T
        self.reveal_turn = reveal_turn          # fact_id -> turn index
        self.reveal_elicited = reveal_elicited  # fact_id -> bool
        self.decisive = decisive                # [{'fact_id':..., 'flips':[...]}, ...]
        self.settle_turn = settle_turn          # turn index or None
        self.dca = dca
        self.schema_valid = schema_valid
        self.leaks = leaks
        self.acts = acts                        # ['ask', 'followup', ...]
        self.conversation = conversation
        self.case = case

    @property
    def valid(self):
        return bool(self.schema_valid) and not self.leaks


class VerifierPRM(object):
    """Deterministic. r_t from the dataset's own decisive-fact structure.

    binary  -- EPO-faithful: r_t in {0, 1}.
    graded  -- r_t weighted by how many checks the disclosed fact flips, and the settle
               term continuous in dca rather than thresholded. Strictly more informative
               and free; matters most at small episode budgets, where an all-zero episode
               contributes no gradient at all.
    """

    def __init__(self, mode='binary', done_tau=0.6, lam=1.0, leak_invalidates=True):
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
                continue                        # disclosed, but not decisive
            if not tr.reveal_elicited.get(fid):
                continue                        # volunteered: the chair earned nothing
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


JUDGE_SYSTEM = (
    'You review a meeting transcript and identify which of the chair\'s strategies were '
    'critical to the outcome. Answer with a JSON list of turn numbers and nothing else, '
    'for example: [1, 3]')

JUDGE_TEMPLATE = """Goal: {goal}

The chair issued these strategies, one per numbered turn:
{acts}

Transcript:
{dialog}

Outcome score (0-1, higher is better): {score:.3f}

Which of the numbered turns above carried a strategy that was CRITICAL to reaching the
outcome? A turn is critical if removing it would plausibly have changed the result.
Answer with a JSON list of turn numbers only."""


class JudgePRM(object):
    """EPO-faithful: an LLM marks the critical turns, post-hoc, from the whole episode.

    Reads OPENROUTER_API_KEY (or OPENAI_API_KEY) from the environment. Never put a key
    in this file.
    """

    def __init__(self, model=None, base_url=None, max_retries=4, timeout=60):
        import config
        self.model = model or config.Defaults.or_model
        self.base_url = base_url or config.Defaults.or_base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self.n_calls = 0

    def _client_lazy(self):
        if self._client is None:
            key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if not key:
                raise SystemExit('OPENROUTER_API_KEY is not set; JudgePRM needs it. '
                                 'Set it in your shell, never in a file.')
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout)
        return self._client

    def __call__(self, tr):
        r = [0.0] * tr.n_turns
        if tr.n_turns == 0:
            return r
        acts = '\n'.join('  %d. %s' % (i + 1, a) for i, a in enumerate(tr.acts))
        dialog = '\n'.join('%s: %s' % (t['role'], t['content'][:400])
                           for t in tr.conversation)[-6000:]
        prompt = JUDGE_TEMPLATE.format(goal=tr.case.get('description', ''), acts=acts,
                                       dialog=dialog, score=float(tr.dca))
        raw = ''
        for _ in range(self.max_retries):
            try:
                self.n_calls += 1
                resp = self._client_lazy().chat.completions.create(
                    model=self.model, temperature=0, max_tokens=64,
                    messages=[{'role': 'system', 'content': JUDGE_SYSTEM},
                              {'role': 'user', 'content': prompt}])
                raw = (resp.choices[0].message.content or '').strip()
                break
            except Exception as e:                  # noqa: BLE001
                raw, err = '', e
        for n in _parse_indices(raw):
            if 1 <= n <= tr.n_turns:
                r[n - 1] = 1.0
        return r


def _parse_indices(text):
    try:
        obj = json.loads(re.search(r'\[.*?\]', text, re.S).group(0))
        return [int(x) for x in obj if isinstance(x, (int, float, str)) and str(x).strip().isdigit()]
    except Exception:                               # noqa: BLE001
        return [int(x) for x in re.findall(r'\b(\d{1,2})\b', text or '')]


def build(name, cfg):
    if name == 'verifier':
        return VerifierPRM(mode=cfg.prm_mode, done_tau=cfg.done_tau,
                           leak_invalidates=cfg.leak_invalidates)
    if name == 'judge':
        return JudgePRM(model=cfg.or_model, base_url=cfg.or_base_url)
    raise ValueError('unknown prm %r' % name)


# ------------------------------------------------------------------ agreement
def agreement(rs_a, rs_b):
    """Cohen's kappa between two PRMs over aligned per-turn binary labels.

    This is the measurement vanilla EPO cannot make: with no executable ground truth,
    there is nothing to check the judge against.
    """
    a = [1 if x > 0 else 0 for ep in rs_a for x in ep]
    b = [1 if x > 0 else 0 for ep in rs_b for x in ep]
    n = len(a)
    if n == 0 or len(b) != n:
        return float('nan')
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
