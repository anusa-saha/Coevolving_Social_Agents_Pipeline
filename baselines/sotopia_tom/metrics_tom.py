"""Sotopia-ToM's InfoMgmt metric, rebuilt from CSA's executable signals.

The paper scores four dimensions with an LLM judge and combines them as a geometric mean:

    InfoMgmt = [ DA . IA . (1 - CPV) . EFF ]^(1/4)

Three of the four have exact CSA analogues that need no judge. The fourth does not exist
here at all, and is reported as absent rather than approximated:

    DA   disclosure alignment   -> decisive facts actually pooled          (computed)
    IA   inquiry alignment      -> disclosures that FOLLOWED an eliciting  (computed)
                                   turn addressed at a named participant
    EFF  efficiency             -> how early the pooling happened, against  (computed)
                                   the scenario's own turn budget
    CPV  critical privacy viol. -> NO ANALOGUE. CSA is a single public      (absent)
                                   meeting with no private channel, and no
                                   fact that must be withheld: every private
                                   fact SHOULD be pooled.

So the composite reported here is the three-way geometric mean

    InfoMgmt3 = [ DA . IA . EFF ]^(1/3)

and it is NOT comparable in level to the paper's four-way number. Dropping a factor that
is usually close to 1 raises the mean, so InfoMgmt3 will read higher than InfoMgmt would
on the same behaviour. Use it to rank the five arms against each other, never to compare
against the paper's table.

`leaks` is deliberately not substituted for CPV. A leak here is an agent stating a fact it
was never shown -- fabrication, not inappropriate disclosure. It is reported separately.
"""
import statistics


def _geo(values):
    """Geometric mean. Any zero factor sends the whole thing to zero, which is the point
    of the paper's design: one failed dimension should not be averaged away."""
    vals = [max(0.0, float(v)) for v in values]
    if not vals or any(v == 0.0 for v in vals):
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= v
    return prod ** (1.0 / len(vals))


def disclosure_alignment(rec):
    """DA: share of decisive facts that reached the meeting."""
    return float((rec.get('score') or {}).get('disclosure_rate') or 0.0)


def inquiry_alignment(rec):
    """IA: of the facts that surfaced, the share drawn out by a deliberate request.

    The paper excludes passive receipt, which is exactly what `reveal_elicited` records:
    a disclosure counts only if the chair's preceding turn named a participant and asked
    something. An episode where nothing surfaced scores 0 -- there was no inquiry to
    align.
    """
    el = rec.get('reveal_elicited') or {}
    if not el:
        return 0.0
    return sum(1 for v in el.values() if v) / len(el)


def efficiency(rec):
    """EFF: how early pooling happened, normalised by the scenario's own budget.

    The paper uses the median round at which desired information is acquired, and calls
    it robust to outliers. Median over the disclosure turns, mapped so that surfacing on
    the first chair turn scores 1 and surfacing on the last scores near 0.
    """
    turns = [t for t in (rec.get('reveal_turn') or {}).values()
             if isinstance(t, (int, float))]
    cap = max(1, int(rec.get('max_turn') or 1))
    if not turns:
        return 0.0
    med = statistics.median(turns)
    return max(0.0, min(1.0, 1.0 - (med / cap)))


def infomgmt(rec):
    da, ia, eff = disclosure_alignment(rec), inquiry_alignment(rec), efficiency(rec)
    return {'DA': da, 'IA': ia, 'EFF': eff, 'InfoMgmt3': _geo([da, ia, eff])}


def summarise(recs):
    """Per-arm summary: the ToM metrics plus the CSA outcome metrics they sit beside."""
    if not recs:
        return {}

    def avg(fn):
        v = [fn(r) for r in recs if fn(r) is not None]
        return sum(v) / len(v) if v else float('nan')

    sc = lambda r: (r.get('score') or {})            # noqa: E731
    per = [infomgmt(r) for r in recs]

    # The composite is averaged over EPISODES, not computed from the averaged dimensions.
    # Doing it the other way lets a scenario that scored zero on one dimension be
    # rescued by the others, which is exactly what the geometric mean exists to prevent.
    return {
        'n': len(recs),
        'DA': sum(p['DA'] for p in per) / len(per),
        'IA': sum(p['IA'] for p in per) / len(per),
        'EFF': sum(p['EFF'] for p in per) / len(per),
        'InfoMgmt3': sum(p['InfoMgmt3'] for p in per) / len(per),
        'CPV': None,                                 # no private channel on CSA
        'SR': avg(lambda r: 1.0 if r.get('done') == 1 else 0.0),
        'dca': avg(lambda r: sc(r).get('dca')),
        'disclosure_rate': avg(lambda r: sc(r).get('disclosure_rate')),
        'any_reveal': sum(1 for r in recs if r.get('revealed')),
        'cbar': avg(lambda r: sc(r).get('cbar')),
        'pbar': avg(lambda r: sc(r).get('pbar')),
        'schema_valid': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
        'cover': avg(lambda r: r.get('cover')),
        'leaks': sum(1 for r in recs if r.get('leaks')),
        'turns': avg(lambda r: r.get('turns')),
        'n_calls': avg(lambda r: r.get('n_calls')),
        'prompt_chars': avg(lambda r: r.get('prompt_chars')),
    }


def paired(a_recs, b_recs, key):
    """Wins / ties / losses per scenario plus a two-sided sign test.

    Five arms over 42 scenarios with small expected effects: paired testing is the only
    way to see anything, and every arm runs the same scenarios by construction.
    """
    from math import comb
    fn = {'DA': disclosure_alignment, 'IA': inquiry_alignment, 'EFF': efficiency,
          'InfoMgmt3': lambda r: infomgmt(r)['InfoMgmt3']}.get(key)
    if fn is None:
        fn = lambda r: (r.get('score') or {}).get(key)   # noqa: E731
    A = {r['uid']: r for r in a_recs}
    B = {r['uid']: r for r in b_recs}
    w = l = t = 0
    for u in sorted(set(A) & set(B)):
        x, y = fn(A[u]), fn(B[u])
        if x is None or y is None:
            continue
        if x > y:
            w += 1
        elif x < y:
            l += 1
        else:
            t += 1
    n = w + l
    if n == 0:
        return {'win': w, 'tie': t, 'loss': l, 'p': float('nan')}
    k = min(w, l)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return {'win': w, 'tie': t, 'loss': l, 'p': p}
