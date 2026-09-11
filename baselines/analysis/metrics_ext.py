"""Section E (numeric / categorical prediction) and headline statistics.

Imported by compute_extended_metrics.py. Pure stdlib, no GPU, no judge.

A note on section E and CSA. The catalogue's numeric block was designed for a task that
predicts numbers. CSA settlements are overwhelmingly categorical -- a lane, a drug, a
patient, a time -- and the dataset ships no gold settlement at all. What it does ship is
content checks of the form

    decisions['medication_approved'] == 'dalbavancin'
    decisions['lane'] in ['Lane 2', 'lane 2']

which NAME the gold answer. Recovering it from there is the only route to a
predicted-against-gold comparison, and it is exact where it applies. Measured on the
corpus: 355 of 789 decision fields yield a gold, but only 20 of those parse as numeric.

So PE@K, MAE, RMSE, MAPE, MPE, R2, Pearson, Spearman and Kendall are all computed here
and all reported WITH THEIR n. At n in the single digits they are not evidence; printing
them beside the sample size is more honest than omitting them, because it lets the reader
see exactly why they cannot be leaned on.
"""
import collections
import math
import random
import re
import statistics as st

_EQ = re.compile(r"decisions\[['\"]([^'\"]+)['\"]\]\s*==\s*['\"]([^'\"]+)['\"]")
_IN = re.compile(r"decisions\[['\"]([^'\"]+)['\"]\]\s*in\s*\[([^\]]+)\]")
_LIT = re.compile(r"['\"]([^'\"]+)['\"]")
_NUM = re.compile(r"^\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*"
                  r"(%|kg|mg|g|km|m|hours?|hrs?|h|minutes?|mins?|liters?|litres?|l)?\s*$",
                  re.I)


# ------------------------------------------------------------------ gold recovery
def gold_values(case):
    """field -> {'kind': 'exact'|'set', 'gold': ...} recovered from the content checks."""
    out = {}
    for expr in (case.get('content_checks') or {}).values():
        for m in _EQ.finditer(expr):
            out.setdefault(m.group(1), {'kind': 'exact', 'gold': m.group(2)})
        for m in _IN.finditer(expr):
            alts = _LIT.findall(m.group(2))
            if alts:
                out.setdefault(m.group(1), {'kind': 'set', 'gold': alts})
    return out


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    m = _NUM.match(v.replace(',', ''))
    return float(m.group(1)) if m else None


def canon(v):
    return re.sub(r'\s+', ' ', str(v)).strip().lower().strip(' .;:')


def _decisions(rec):
    d = ((rec.get('settlement') or {}).get('decisions') or {})
    return d if isinstance(d, dict) else {}


def numeric_pairs(recs, cases):
    out = []
    for r in recs:
        case = cases.get(r.get('uid'))
        dec = _decisions(r)
        if not case:
            continue
        for field, g in gold_values(case).items():
            if g['kind'] != 'exact':
                continue
            gv, pv = as_number(g['gold']), as_number(dec.get(field))
            if gv is not None and pv is not None:
                out.append((gv, pv, r.get('uid'), field))
    return out


def categorical_pairs(recs, cases):
    out = []
    for r in recs:
        case = cases.get(r.get('uid'))
        dec = _decisions(r)
        if not case:
            continue
        for field, g in gold_values(case).items():
            if field not in dec:
                continue
            pred = canon(dec[field])
            ok = (pred == canon(g['gold'])) if g['kind'] == 'exact' \
                else any(pred == canon(a) for a in g['gold'])
            out.append((field, ok, pred))
    return out


# ------------------------------------------------------------------ correlations
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rk[order[k]] = r
        i = j + 1
    return rk


def pearson(a, b):
    if len(a) < 2:
        return float('nan')
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float('nan')


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def kendall(a, b):
    n = len(a)
    if n < 2:
        return float('nan')
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (c + d) if (c + d) else float('nan')


# ------------------------------------------------------------------ section E
def section_E(recs, cases, tol_delta=1.0):
    num = numeric_pairs(recs, cases)
    cat = categorical_pairs(recs, cases)
    out = {'n_numeric_pairs': len(num), 'n_categorical_pairs': len(cat),
           'tolerance_delta': tol_delta,
           'note': 'numeric metrics are unreliable below roughly n=30; the n is printed '
                   'beside them for exactly that reason'}

    if cat:
        out['exact_match_accuracy'] = sum(1 for _f, ok, _p in cat if ok) / len(cat)
        wrong = collections.Counter(f for f, ok, _p in cat if not ok)
        seen = collections.Counter(f for f, _ok, _p in cat)
        out['field_error_rate'] = {f: wrong[f] / seen[f] for f in sorted(seen)}
        out['worst_fields'] = dict(wrong.most_common(8))

    if num:
        g = [x[0] for x in num]
        p = [x[1] for x in num]
        err = [pi - gi for gi, pi in zip(g, p)]
        ape = [abs(e) / abs(gi) for e, gi in zip(err, g) if gi]
        pe = [(pi - gi) / abs(gi) for gi, pi in zip(g, p) if gi]
        mg = st.mean(g)
        sstot = sum((x - mg) ** 2 for x in g)
        out.update({
            'MAE': st.mean([abs(e) for e in err]),
            'RMSE': math.sqrt(st.mean([e * e for e in err])),
            'MAPE': st.mean(ape) if ape else float('nan'),
            'MPE': st.mean(pe) if pe else float('nan'),
            'tolerance_accuracy': sum(1 for e in err if abs(e) <= tol_delta) / len(err),
            'R2': (1 - sum(e * e for e in err) / sstot) if sstot else float('nan'),
            'pearson_r': pearson(g, p),
            'spearman_rho': spearman(g, p),
            'kendall_tau': kendall(g, p),
        })
        for K in (10, 20, 30):
            out['PE@%d' % K] = (sum(1 for a in ape if a <= K / 100.0) / len(ape)
                                if ape else float('nan'))
    return out


# ------------------------------------------------------------------ headline stats
def bootstrap_ci(vals, iters=10000, alpha=0.05, seed=0):
    """Percentile bootstrap of the mean.

    At n=42 every point estimate in this project is uncertain enough that quoting it bare
    is misleading. This is the cheapest fix available and needs no assumptions.
    """
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if len(vals) < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iters))
    return (means[int((alpha / 2) * iters)],
            means[min(iters - 1, int((1 - alpha / 2) * iters))])


def cliffs_delta(a, b):
    """Non-parametric effect size, paired-friendly and ordinal.

    A sign test says the direction is certain; it says nothing about magnitude. Bands
    (Romano et al.): |d| < .147 negligible, < .33 small, < .474 medium, else large.
    """
    a = [x for x in a if isinstance(x, (int, float))]
    b = [x for x in b if isinstance(x, (int, float))]
    if not a or not b:
        return float('nan')
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def delta_label(d):
    if d != d:
        return '-'
    ad = abs(d)
    return ('negligible' if ad < 0.147 else 'small' if ad < 0.33
            else 'medium' if ad < 0.474 else 'large')


def per_scenario(recs, key):
    out = {}
    for r in recs:
        s = r.get('score') or {}
        v = s.get(key, r.get(key))
        if isinstance(v, (int, float)):
            out[r.get('uid')] = v
    return out


def paired_sign(a_recs, b_recs, key):
    """Wins / ties / losses per scenario, two-sided sign test, and Cliff's delta."""
    pa, pb = per_scenario(a_recs, key), per_scenario(b_recs, key)
    common = sorted(set(pa) & set(pb))
    w = sum(1 for u in common if pa[u] > pb[u])
    l = sum(1 for u in common if pa[u] < pb[u])
    t = len(common) - w - l
    n = w + l
    if n:
        k = min(w, l)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    else:
        p = float('nan')
    d = cliffs_delta([pa[u] for u in common], [pb[u] for u in common])
    return {'win': w, 'tie': t, 'loss': l, 'p': p, 'cliffs_delta': d,
            'effect': delta_label(d), 'n': len(common)}


def win_matrix(arms, key='dca'):
    """Full N x N paired win rate. One table shows the whole ordering, and exposes
    non-transitivity if any exists."""
    labels = sorted(arms)
    mat = {}
    for a in labels:
        pa = per_scenario(arms[a], key)
        row = {}
        for b in labels:
            if a == b:
                row[b] = None
                continue
            pb = per_scenario(arms[b], key)
            common = set(pa) & set(pb)
            w = sum(1 for u in common if pa[u] > pb[u])
            l = sum(1 for u in common if pa[u] < pb[u])
            row[b] = (w / (w + l)) if (w + l) else float('nan')
        mat[a] = row
    return mat


def _floor_of(rec, key='dca'):
    """The floor score for one episode.

    Records store `floor` as the whole score dict evaluated under a do-nothing policy,
    so the scalar has to be pulled out of it. Older records may store a bare number.
    """
    f = rec.get('floor')
    if isinstance(f, dict):
        v = f.get(key)
    else:
        v = f
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _ceiling_of(rec, key='dca'):
    """The ceiling, or 1.0 when the record does not carry one.

    Most records store ceiling=None. Treating that as 1.0 says "a perfect score was
    reachable", which is the right default for a benchmark whose checks are all
    satisfiable in principle -- and it is stated rather than silently assumed.
    """
    c = rec.get('ceiling')
    if isinstance(c, dict):
        c = c.get(key)
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else 1.0


def normalised_gain(recs, key='dca'):
    """Mean over scenarios of (score - floor) / (ceiling - floor).

    Normalised per scenario and then averaged. A ratio of means would let one easy
    scenario with a high floor distort the whole column, and would not be comparable
    across arms that happen to see different difficulty mixes.
    """
    vals = []
    for r in recs:
        v = (r.get('score') or {}).get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        lo, hi = _floor_of(r, key), _ceiling_of(r, key)
        if hi - lo > 1e-9:
            vals.append((float(v) - lo) / (hi - lo))
    return st.mean(vals) if vals else float('nan')


def headline(recs):
    """The row a results table actually needs: outcome, interval, cost, cost-adjusted
    outcome, and the floor-normalised score."""
    dca = [(r.get('score') or {}).get('dca') for r in recs]
    dis = [(r.get('score') or {}).get('disclosure_rate') for r in recs]
    dca_v = [x for x in dca if isinstance(x, (int, float))]
    dis_v = [x for x in dis if isinstance(x, (int, float))]
    calls = [r.get('n_calls') for r in recs if isinstance(r.get('n_calls'), (int, float))]
    chars = [r.get('prompt_chars') for r in recs
             if isinstance(r.get('prompt_chars'), (int, float))]
    floors = [_floor_of(r) for r in recs]
    ceils = [_ceiling_of(r) for r in recs]

    md = st.mean(dca_v) if dca_v else float('nan')
    mc = st.mean(calls) if calls else float('nan')
    mt = (st.mean(chars) / 4.0) if chars else float('nan')   # ~4 chars per token
    mceil = st.mean(ceils) if ceils else 1.0
    mfloor = st.mean(floors) if floors else 0.0
    ok = lambda x: isinstance(x, float) and x == x          # noqa: E731
    return {
        'n': len(recs),
        'dca': md, 'dca_ci': bootstrap_ci(dca_v),
        'disclosure': st.mean(dis_v) if dis_v else float('nan'),
        'disclosure_ci': bootstrap_ci(dis_v),
        'calls': mc,
        'dca_per_100_calls': (md / mc * 100) if ok(md) and ok(mc) and mc else float('nan'),
        'dca_per_10k_tokens': (md / mt * 10000) if ok(md) and ok(mt) and mt
                              else float('nan'),
        'ceiling': mceil,
        'floor': mfloor,
        # Per-scenario (score - floor) / (ceiling - floor), then averaged. The floor is
        # NOT zero everywhere any more: 3 of the 550 scenarios (all in domains added
        # beyond the original three) are partly solvable with no disclosure at all.
        'normalised_gain': normalised_gain(recs),
    }


# ================================================================== section H
# Stratified breakdowns, dispersion, subscores and multiplicity control.
#
# Everything above reports one number per arm over 42 scenarios. That hides three things
# a reviewer will ask about immediately:
#   * whether a gain is real or comes from one easy domain,
#   * whether the interval on the DIFFERENCE excludes zero (separate per-arm CIs can
#     overlap while the paired difference is decisive -- the arms see the same scenarios),
#   * whether 16 paired tests produced a "significant" result by chance.
# Section H covers those.

SUBSCORES = ('dca', 'cbar', 'pbar', 'content', 'provenance', 'close', 'joint',
             'schema_valid', 'disclosure_rate', 'hallucinated_credit')


def _sv(rec, key):
    v = (rec.get('score') or {}).get(key, rec.get(key))
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None


def strat(recs, field, key='dca'):
    """mean of `key` within each level of a record field, plus the level's n."""
    buckets = collections.defaultdict(list)
    for r in recs:
        v = _sv(r, key)
        if v is not None and r.get(field) is not None:
            buckets[r[field]].append(v)
    return {str(k): {'n': len(v), 'mean': st.mean(v)}
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def dispersion(recs, key='dca'):
    """Spread of the per-scenario score. A mean of 0.35 built from 42 middling episodes
    is a different result from one built from 15 perfect and 27 zero episodes."""
    v = sorted(x for x in (_sv(r, key) for r in recs) if x is not None)
    if not v:
        return {}
    n = len(v)
    q = lambda p: v[min(n - 1, int(p * n))]              # noqa: E731
    return {'n': n, 'mean': st.mean(v), 'sd': st.pstdev(v) if n > 1 else 0.0,
            'min': v[0], 'q25': q(0.25), 'median': st.median(v), 'q75': q(0.75),
            'max': v[-1], 'iqr': q(0.75) - q(0.25),
            'frac_zero': sum(1 for x in v if x <= 0.0) / n,
            'frac_one': sum(1 for x in v if x >= 1.0) / n}


def subscore_table(recs):
    """Mean + bootstrap CI for every score component the verifier emits."""
    out = {}
    for k in SUBSCORES:
        v = [x for x in (_sv(r, k) for r in recs) if x is not None]
        if v:
            out[k] = {'n': len(v), 'mean': st.mean(v), 'ci': bootstrap_ci(v)}
    return out


def paired_delta(a_recs, b_recs, key='dca', iters=10000, seed=0):
    """Bootstrap CI on the PAIRED difference (a - b), resampling scenarios.

    This is the interval to quote for a comparison. Two arms scored on the same 42
    scenarios share their difficulty, so the paired difference has far less variance than
    either arm's own mean -- and its CI can exclude zero when the marginal CIs overlap.
    """
    pa, pb = per_scenario(a_recs, key), per_scenario(b_recs, key)
    common = sorted(set(pa) & set(pb))
    d = [pa[u] - pb[u] for u in common]
    if len(d) < 2:
        return {'n': len(d)}
    lo, hi = bootstrap_ci(d, iters=iters, seed=seed)
    return {'n': len(d), 'mean_delta': st.mean(d), 'ci': (lo, hi),
            'excludes_zero': (lo > 0) or (hi < 0)}


def holm(pvals):
    """Holm-Bonferroni step-down over a family of tests. `pvals` is {name: p}.

    With 8 arms x 2 metrics the family is 16 tests, so an uncorrected p just under .05 is
    not by itself evidence. Holm controls the family-wise error rate and, unlike plain
    Bonferroni, loses no power to do it.
    """
    items = sorted(((p, k) for k, p in pvals.items() if p == p))
    m = len(items)
    out, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        run = max(run, adj)                # enforce monotonicity down the ladder
        out[k] = run
    for k, p in pvals.items():
        if p != p:
            out[k] = float('nan')
    return out


def disclosure_vs_accuracy(recs):
    """Does telling the room more actually buy a better decision, scenario by scenario?"""
    pairs = [(_sv(r, 'disclosure_rate'), _sv(r, 'dca')) for r in recs]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return {'n': len(pairs)}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    return {'n': len(pairs), 'pearson_r': pearson(x, y), 'spearman_rho': spearman(x, y),
            'kendall_tau': kendall(x, y)}


def difficulty_tertiles(recs, base_recs, key='dca'):
    """Split scenarios into easy / medium / hard by the BASELINE's score on them, then
    report the arm within each third. Shows where an arm's gain actually comes from."""
    pb = per_scenario(base_recs, key)
    pa = per_scenario(recs, key)
    common = sorted(set(pa) & set(pb), key=lambda u: pb[u])
    if len(common) < 6:
        return {}
    third = len(common) // 3
    names = ('hard', 'medium', 'easy')
    out = {}
    for i, nm in enumerate(names):
        seg = common[i * third:] if i == 2 else common[i * third:(i + 1) * third]
        out[nm] = {'n': len(seg), 'arm': st.mean([pa[u] for u in seg]),
                   'base': st.mean([pb[u] for u in seg])}
    return out


def section_H(recs):
    return {
        'by_domain': strat(recs, 'domain'),
        'by_num_agents': strat(recs, 'num_agents'),
        'by_scenario_type': strat(recs, 'scenario_type'),
        'macro_by_domain': (st.mean([v['mean'] for v in strat(recs, 'domain').values()])
                            if strat(recs, 'domain') else float('nan')),
        'dispersion_dca': dispersion(recs, 'dca'),
        'dispersion_disclosure': dispersion(recs, 'disclosure_rate'),
        'subscores': subscore_table(recs),
        'disclosure_vs_accuracy': disclosure_vs_accuracy(recs),
    }
