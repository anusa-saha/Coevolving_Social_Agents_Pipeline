"""DAT's own metrics, on top of the CSA outcome metrics every arm reports.

Three groups.

CSA outcome -- dca, disclosure, cbar/pbar, schema validity, cost. Identical definitions to
every other arm; they come out of csa_core.verifier and are only averaged here.

Steering -- what the planner actually did. An arm whose action vector is the same at every
turn of every scenario is a constant prefix, not a policy, and its outcome numbers can
look fine while the mechanism the paper describes is absent. `action_cos` (turn-to-turn
cosine) and `action_norm_sd` are what make that visible.

Language health -- distinct-1/2 and utterance length on the chair's turns. DAT's opening
argument is that RL over language degrades the language distribution, and that freezing
the LM and steering with a prefix avoids it. That claim is about the TEXT, so the arm
reports it locally rather than waiting for the cross-arm script: if the steered chair's
distinct-2 collapses against the unsteered chair's, DAT has bought its reward the way the
paper says it should not be possible to.

analysis/compute_extended_metrics.py computes the same language numbers across all arms;
these are here so `python run_dat.py --compare` answers the question without a second
script, and the two agree by construction (both call csa_core.detectors.content_tokens).
"""
import statistics as st
from math import comb

import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core.detectors import content_tokens


def _sc(rec):
    return rec.get('score') or {}


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float('nan')


# ------------------------------------------------------------------ steering
def chair_turns(rec):
    return [t['content'] for t in (rec.get('dialog') or [])
            if t.get('speaker') == 'sys']


def distinct_n(texts, n):
    total, uniq = 0, set()
    for t in texts:
        tok = content_tokens(t)
        for i in range(len(tok) - n + 1):
            uniq.add(tuple(tok[i:i + n]))
            total += 1
    return (len(uniq) / total) if total else float('nan')


def steering(recs):
    """What the planner emitted, aggregated. Empty for the unsteered arm, by design."""
    norms = [x for r in recs for x in (r.get('action_norms') or [])]
    res = [x for r in recs for x in (r.get('residual_norms') or [])]
    cos = [r['action_cos'] for r in recs
           if isinstance(r.get('action_cos'), (int, float))
           and r['action_cos'] == r['action_cos']]      # None on unsteered/one-turn
    return {
        'n_actions': len(norms),
        'action_norm_mean': st.mean(norms) if norms else float('nan'),
        'action_norm_sd': st.pstdev(norms) if len(norms) > 1 else 0.0,
        'residual_norm_mean': st.mean(res) if res else float('nan'),
        'action_cos_mean': st.mean(cos) if cos else float('nan'),
        'planner_forwards': _avg([r.get('planner_forwards') for r in recs]),
    }


def language(recs):
    chair = [c for r in recs for c in chair_turns(r) if c.strip()]
    lens = [len(c.split()) for c in chair]
    return {
        'n_chair_turns': len(chair),
        'distinct_1': distinct_n(chair, 1),
        'distinct_2': distinct_n(chair, 2),
        'utterance_len_mean': st.mean(lens) if lens else float('nan'),
        'utterance_len_sd': st.pstdev(lens) if len(lens) > 1 else 0.0,
    }


def elicit_rate(rec):
    """Share of the chair's turns the detector read as an actual request to a named
    participant. DAT has no acts to count, so this is the closest thing to an act
    distribution the arm can honestly report."""
    acts = rec.get('derived_acts') or []
    if not acts:
        return None
    return sum(1 for a in acts if a == 'elicit') / len(acts)


# ------------------------------------------------------------------ summary
def summarise(recs):
    if not recs:
        return {}
    out = {
        'n': len(recs),
        'arm': recs[0].get('arm'),
        'SR': _avg([1.0 if r.get('done') == 1 else 0.0 for r in recs]),
        'dca': _avg([_sc(r).get('dca') for r in recs]),
        'disclosure_rate': _avg([_sc(r).get('disclosure_rate') for r in recs]),
        'any_reveal': sum(1 for r in recs if r.get('revealed')),
        'cbar': _avg([_sc(r).get('cbar') for r in recs]),
        'pbar': _avg([_sc(r).get('pbar') for r in recs]),
        'close': _avg([_sc(r).get('close') for r in recs]),
        'joint': _avg([1.0 if _sc(r).get('joint') else 0.0 for r in recs]),
        'schema_valid': _avg([1.0 if _sc(r).get('schema_valid') else 0.0 for r in recs]),
        'hallucinated_credit': _avg([_sc(r).get('hallucinated_credit') for r in recs]),
        'cover': _avg([r.get('cover') for r in recs]),
        'leaks': sum(1 for r in recs if r.get('leaks')),
        'elicit_rate': _avg([elicit_rate(r) for r in recs]),
        'turn_reward_mean': _avg([sum(r.get('turn_rewards') or [0.0]) for r in recs]),
        'terminal_reward': _avg([r.get('terminal_reward') for r in recs]),
        'turns': _avg([r.get('turns') for r in recs]),
        'n_calls': _avg([r.get('n_calls') for r in recs]),
        'prompt_chars': _avg([r.get('prompt_chars') for r in recs]),
    }
    out.update(steering(recs))
    out.update(language(recs))
    return out


# ------------------------------------------------------------------ paired
_GETTERS = {
    'elicit_rate': elicit_rate,
    'SR': lambda r: 1.0 if r.get('done') == 1 else 0.0,
    'distinct_2': lambda r: distinct_n(chair_turns(r), 2),
}


def _get(rec, key):
    fn = _GETTERS.get(key)
    if fn is not None:
        return fn(rec)
    if key in rec:
        return rec.get(key)
    return _sc(rec).get(key)


def paired(a_recs, b_recs, key):
    """Wins / ties / losses per scenario, plus a two-sided sign test.

    The three arms see identical scenarios by construction, so pairing is the only way to
    see an effect this size: DAT's own table reports a +0.35 move on a 0-4 scale with
    confidence intervals that overlap, and it is the paired comparison that carries it.
    """
    A = {r['uid']: r for r in a_recs}
    B = {r['uid']: r for r in b_recs}
    w = l = t = 0
    deltas = []
    for u in sorted(set(A) & set(B)):
        x, y = _get(A[u], key), _get(B[u], key)
        if x is None or y is None or x != x or y != y:
            continue
        deltas.append(x - y)
        if x > y:
            w += 1
        elif x < y:
            l += 1
        else:
            t += 1
    n = w + l
    p = float('nan')
    if n:
        k = min(w, l)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return {'win': w, 'tie': t, 'loss': l, 'p': p,
            'mean_delta': st.mean(deltas) if deltas else float('nan'),
            'n_paired': len(deltas)}


def steering_effect(steered_recs, base_recs, keys=('dca', 'disclosure_rate', 'cbar',
                                                   'elicit_rate', 'distinct_2')):
    """The arm's headline: what the prefix changed, per scenario, against the control."""
    return {k: paired(steered_recs, base_recs, k) for k in keys}
