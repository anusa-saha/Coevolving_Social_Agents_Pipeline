"""Utterance-level reward attribution -- the core of Sotopia-RL, made deterministic.

The paper's reward is  r_t = G . A(a_t, tau)  (Eq. 3): an episode-level score G times an
LLM attributor's judgement of how much turn t contributed, then per-dimension normalised
and averaged (Eq. 4). Both factors are GPT-4o guesses, and on SOTOPIA there is nothing to
check them against.

On CSA both are computable. `decisive_facts` states, for every private fact, exactly
which checks it controls, and the environment records which turn surfaced it. Criticality
is a lookup, not a judgement: zero model calls, byte-for-byte reproducible.

Dimensions (the CSA analogue of REL / KNO / GOAL):

    POOL   did private information surface at all          <- KNO, knowledge seeking
    USE    did surfaced information land in passing checks <- GOAL
    COVER  did the chair draw out the people who had it    <- REL, but see below

COVER is deliberately NOT "advisors whose surname was mentioned". Measured that way it
saturates on the opening turn, where the chair greets everyone by name, and rewards
politeness rather than elicitation. Here an advisor counts only once it has been
addressed AND has subsequently disclosed something.

Three details that are easy to get wrong and that change the numbers:

  * A_pool uses the UNION of flips across facts credited to a turn, not the sum. The
    flips lists overlap, so summing lets a single turn score above 1.
  * Provenance resolution is applied before scoring. Without it this arm runs a harsher
    provenance rule than the other two and the comparison breaks.
  * Eliciting-ness is read from the utterance (detectors_sr.is_eliciting), not from a
    planner's act label -- there is no planner here.
"""
import statistics

from detectors_sr import (REVEAL_THRESHOLD, addressed_in, disclosures, is_eliciting,
                          leaks as leak_check)
from verifier_sr import flipped_checks, score

DIMS = ('pool', 'use', 'cover')


def walk_episode(case, dialog, threshold=REVEAL_THRESHOLD, acts=None):
    """Replay a transcript and recover per-turn bookkeeping.

    `dialog` entries need 'role', 'content' and 'speaker' in {'sys','usr','env'}.
    `acts` optionally supplies a per-chair-turn act label (from an annotated corpus); when
    absent, eliciting-ness comes from the lexical detector.
    """
    dm = case['decision_maker']
    advisors = {a['agent_id'] for a in case['agents']} - {dm}
    id_of = {a['name']: a['agent_id'] for a in case['agents']}

    step = -1
    revealed, reveal_turn, reveal_elicited = set(), {}, {}
    addressed, addressed_turn = set(), {}
    cover_credit = {}                    # turn -> advisors that turn "unlocked"
    all_leaks = []
    last_eliciting = False
    pending = {}                         # advisor -> turn it was addressed on
    settle_turn = None

    for t in dialog:
        spk = t.get('speaker')
        if spk == 'sys':
            step += 1
            act = (acts or {}).get(step)
            last_eliciting = (act in ('ask', 'followup')) if act is not None \
                else is_eliciting(t['content'], case['agents'], dm)
            fresh = addressed_in(t['content'], case['agents'], exclude={dm}) - addressed
            addressed |= fresh
            for aid in fresh:
                addressed_turn[aid] = step
                pending[aid] = step
            for fid in leak_check(case, dm, t['content'], revealed, threshold):
                all_leaks.append({'fact': fid, 'by': dm, 'turn': step})
            if _looks_like_settlement(t['content']):
                settle_turn = step
        elif spk == 'usr':
            aid = id_of.get(t['role'])
            if aid is None:
                continue
            got = disclosures(case, aid, t['content'], revealed, threshold)
            for fid in got:
                revealed.add(fid)
                reveal_turn[fid] = step
                reveal_elicited[fid] = last_eliciting
            if got and aid in pending:
                # COVER pays the turn that ASKED, once the asking produced something
                cover_credit.setdefault(pending.pop(aid), set()).add(aid)
            for fid in leak_check(case, aid, t['content'], revealed, threshold):
                all_leaks.append({'fact': fid, 'by': aid, 'turn': step})

    n = max(1, step + 1)
    return {'n_turns': n, 'revealed': revealed, 'reveal_turn': reveal_turn,
            'reveal_elicited': reveal_elicited, 'addressed': addressed,
            'cover_credit': cover_credit, 'leaks': all_leaks,
            'settle_turn': settle_turn if settle_turn is not None else n - 1,
            'advisors': advisors}


def _looks_like_settlement(text):
    return '{' in text and '}' in text


def attribute(case, dialog, settlement, acts=None, threshold=REVEAL_THRESHOLD,
              gate_on_integrity=True):
    """-> (raw {dim: [r_t]}, diagnostics). Eq. 3 with G and A both computed."""
    w = walk_episode(case, dialog, threshold, acts)
    n = w['n_turns']
    s = score(case, settlement or {}, w['revealed'], resolve=True)

    checks = dict(s['content'])
    checks.update(s['provenance'])
    phi = flipped_checks(case) or ['_']
    passed = {c for c in phi if checks.get(c)}
    flips = {d['fact_id']: set(d.get('flips') or [])
             for d in (case.get('decisive_facts') or [])}

    n_adv = max(1, len(w['advisors']))
    G = {'pool': s['disclosure_rate'],
         'use': s['dca'],
         'cover': len(set().union(*w['cover_credit'].values())) / n_adv
                  if w['cover_credit'] else 0.0}

    A = {d: [0.0] * n for d in DIMS}
    credited = set()
    per_turn_facts = {}
    for fid, turn in w['reveal_turn'].items():
        if not w['reveal_elicited'].get(fid) or not (0 <= turn < n):
            continue
        per_turn_facts.setdefault(turn, set()).add(fid)
    for turn, fids in per_turn_facts.items():
        # UNION, not sum: flips lists overlap
        union = set().union(*(flips.get(f, set()) for f in fids))
        A['pool'][turn] += len(union) / len(phi)
        hit = passed & union
        A['use'][turn] += len(hit) / len(phi)
        credited |= hit
    # checks that landed with no elicited fact behind them: the settlement's own work
    st = w['settle_turn']
    if 0 <= st < n:
        A['use'][st] += len(passed - credited) / len(phi)
    for turn, aids in w['cover_credit'].items():
        if 0 <= turn < n:
            A['cover'][turn] += len(aids) / n_adv

    gate = bool(s['schema_valid']) and not (gate_on_integrity and w['leaks'])
    raw = {d: [(G[d] * a if gate else 0.0) for a in A[d]] for d in DIMS}

    diag = {'n_turns': n, 'G': G, 'gate': gate, 'settle_turn': st,
            'revealed': sorted(w['revealed']),
            'elicited': sorted(f for f, e in w['reveal_elicited'].items() if e),
            'leaks': w['leaks'], 'addressed': sorted(w['addressed']),
            'dca': s['dca'], 'cbar': s['cbar'], 'pbar': s['pbar'],
            'close': s['close'], 'schema_valid': s['schema_valid'],
            'hallucinated_credit': s['hallucinated_credit']}
    return raw, diag


# ------------------------------------------------------------------ Eq. 4
class Normaliser:
    """Dataset-level min-max per dimension, then an equal-weight average.

    The paper writes min_k / max_k without binding k. If k ranges over the TURNS OF ONE
    EPISODE then G_d, being constant across turns, cancels exactly:

        (G.A_t - G.min A) / (G.max A - G.min A)  =  (A_t - min A) / (max A - min A)

    and the episode score stops affecting the labels at all. Ranging k over the dataset
    keeps G_d, which on CSA is the verifier's own judgement and the most trustworthy
    signal available. Dataset-level it is; `within_episode=True` reproduces the other
    reading for the ablation.
    """

    def __init__(self, weights=None, within_episode=False):
        self.weights = weights or {d: 1.0 for d in DIMS}
        self.within_episode = within_episode
        self.lo = {d: 0.0 for d in DIMS}
        self.hi = {d: 1.0 for d in DIMS}

    def fit(self, all_raw):
        """all_raw: iterable of per-episode {dim: [r_t]}."""
        for d in DIMS:
            vals = [x for raw in all_raw for x in raw[d]]
            self.lo[d] = min(vals) if vals else 0.0
            self.hi[d] = max(vals) if vals else 1.0
        return self

    def apply(self, raw):
        n = len(raw[DIMS[0]])
        out = {}
        for d in DIMS:
            if self.within_episode:
                lo, hi = min(raw[d]), max(raw[d])
            else:
                lo, hi = self.lo[d], self.hi[d]
            span = hi - lo
            out[d] = [((x - lo) / span if span > 0 else 0.0) for x in raw[d]]
        tot = sum(self.weights.values()) or 1.0
        return [sum(self.weights[d] * out[d][i] for d in DIMS) / tot for i in range(n)]

    def state(self):
        return {'lo': self.lo, 'hi': self.hi, 'weights': self.weights,
                'within_episode': self.within_episode}

    @classmethod
    def from_state(cls, st):
        o = cls(weights=st.get('weights'), within_episode=st.get('within_episode', False))
        o.lo, o.hi = st['lo'], st['hi']
        return o


def describe(values):
    if not values:
        return {}
    nz = [v for v in values if v > 1e-9]
    return {'n': len(values), 'non_zero': len(nz),
            'frac_non_zero': round(len(nz) / len(values), 4),
            'mean': round(sum(values) / len(values), 4),
            'median': round(statistics.median(values), 4),
            'std': round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            'max': round(max(values), 4)}
