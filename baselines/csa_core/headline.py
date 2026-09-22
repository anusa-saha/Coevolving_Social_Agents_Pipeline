"""The headline metrics, defined once for every arm.

eval.py -- the co-evolved system's held-out evaluation -- reports a fixed set of numbers.
Those are the headline here too, so a baseline row and a co-evolved row read the same way:

    CHECKS      content / provenance / both passed, checks_frac, success
    BEHAVIOUR   reveals, decisive facts revealed, settled rate, turns to settle, turns used
    BOTTLENECK  unsettled | settled with an incomplete record | settled with a complete one
    COMPARISON  paired table and paired bootstrap, eval.py's procedure unchanged

Two of eval.py's blocks measure machinery the baselines do not have. They are replaced by
the nearest thing every arm already records, under their own names, rather than faked:

    ROUTING  -> ADDRESSING  there is no router. Of the advisors the chair named, the share
                            holding a decisive fact (precision), and the share of decisive
                            holders it named (recall).
    INSIGHT  -> GUARDRAIL   there is no insight head. `leaks` -- a speaker stating a private
                            fact it was never shown -- is the same failure, knowledge the
                            speaker should not have, measured the way every arm records it.

Definitions that differ from a literal reading, stated rather than left to be noticed:

  * success      every content AND provenance check passes. Not the verifier's `joint`,
                 which is False on a scenario that ships no provenance checks at all.
  * settled      a settlement exists to score, whether the chair wrote it or the
                 extraction fallback recovered it. `settled_by` separates the two.
  * turns_used   chair utterances in the transcript, the settling one included. The arms'
                 own `turns` fields count different things.

Computed from an episode record plus its case, so it runs on every arm's record shape --
including archived records that predate this module, where `settled_by` reads
'unrecorded' and turns_to_settle is empty. Pure stdlib.
"""
import collections
import random

N_BOOT = 2000                    # eval.py's paired bootstrap resamples

# One row of scenarios.csv / rollouts.csv, in this order.
METRIC_COLS = (
    'uid', 'domain', 'num_agents',
    'checks_passed', 'checks_total', 'checks_frac', 'success',
    'content_passed', 'content_total', 'prov_passed', 'prov_total',
    'reveals', 'decisive_revealed', 'decisive_total',
    'settled', 'settled_by', 'turns_to_settle', 'turns_used', 'max_turn',
    'address_precision', 'address_recall', 'leaks', 'chair_leaks',
    'n_calls', 'dca', 'disclosure_rate',
)

# (key, higher is better). The paired comparison runs over exactly these.
METRICS = (
    ('checks_frac', True), ('success', True), ('checks_passed', True),
    ('content_passed', True), ('prov_passed', True),
    ('reveals', True), ('decisive_revealed', True), ('settled', True),
    ('turns_to_settle', False), ('turns_used', False),
    ('address_precision', True), ('address_recall', True), ('leaks', False),
)

_LABELS = ('uid', 'domain', 'num_agents', 'settled_by')


# ------------------------------------------------------------------ per episode
def chair_name(case):
    return next((a['name'] for a in case.get('agents') or ()
                 if a['agent_id'] == case.get('decision_maker')), None)


def speaker_of(case, turn):
    """'sys' (chair), 'usr' (advisor) or 'env'.

    Some arms tag turns with `speaker`. PPDPP and EPO store only role and content, so the
    chair is recovered by matching the role against the decision maker's name.
    """
    if turn.get('speaker'):
        return turn['speaker']
    if turn.get('role') == 'Meeting':
        return 'env'
    return 'sys' if turn.get('role') == chair_name(case) else 'usr'


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def episode_metrics(case, rec):
    """eval.py's per-scenario summary, from one episode record and its case."""
    s = rec.get('score') or {}
    content, prov = s.get('content'), s.get('provenance')
    scored = isinstance(content, dict) and isinstance(prov, dict)
    if scored:
        c_tot, p_tot = len(content), len(prov)
        c_ok = sum(1 for v in content.values() if v)
        p_ok = sum(1 for v in prov.values() if v)
    else:
        # No per-check results on the record: the totals are still known, the passes are
        # not, and they are left empty rather than guessed.
        c_tot = len(case.get('content_checks') or {})
        p_tot = len(case.get('provenance_checks') or {})
        c_ok = p_ok = None
    total = c_tot + p_tot
    passed = (c_ok + p_ok) if scored else None

    facts = case.get('private_facts') or {}
    decisive = {d['fact_id'] for d in case.get('decisive_facts') or ()}
    revealed = set(rec.get('revealed') or ())
    holders = {facts[f]['owner'] for f in decisive if f in facts}
    addressed = set(rec.get('addressed') or ())
    named = addressed & holders

    settlement = rec.get('settlement')
    settled = isinstance(settlement, dict) and bool(settlement)
    at = _int(rec.get('settle_turn'))
    dialog = [t for t in rec.get('dialog') or () if isinstance(t, dict)]
    leaks = [x for x in rec.get('leaks') or () if isinstance(x, dict)]

    return {
        'uid': rec.get('uid') or case.get('uid'),
        'domain': rec.get('domain') or case.get('domain'),
        'num_agents': rec.get('num_agents') or case.get('num_agents'),
        'checks_passed': passed,
        'checks_total': total,
        'checks_frac': (passed / total) if scored and total else None,
        'success': int(passed == total) if scored and total else None,
        'content_passed': c_ok, 'content_total': c_tot,
        'prov_passed': p_ok, 'prov_total': p_tot,
        'reveals': len(revealed),
        'decisive_revealed': len(revealed & decisive),
        'decisive_total': len(decisive),
        'settled': int(settled),
        'settled_by': (rec.get('settled_by') or 'unrecorded') if settled else '',
        'turns_to_settle': (at + 1) if settled and at is not None and at >= 0 else None,
        'turns_used': (sum(1 for t in dialog if speaker_of(case, t) == 'sys')
                       if dialog else rec.get('turns')),
        'max_turn': rec.get('max_turn'),
        'address_precision': (len(named) / len(addressed)) if addressed else None,
        'address_recall': (len(named) / len(holders)) if holders else None,
        'leaks': len(leaks),
        'chair_leaks': sum(1 for x in leaks if x.get('by') == case.get('decision_maker')),
        'n_calls': rec.get('n_calls'),
        'dca': s.get('dca'),
        'disclosure_rate': s.get('disclosure_rate'),
    }


def rows_from_records(recs, cases):
    """episode_metrics over every record whose uid is in `cases` (uid -> case)."""
    return [episode_metrics(cases[r['uid']], r) for r in recs if r.get('uid') in cases]


# ------------------------------------------------------------------ aggregation
def num(v):
    """A metric value as a float, or None. Also accepts what csv.DictReader returns."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v) if v == v else None
    if isinstance(v, str) and v.strip():
        try:
            f = float(v)
        except ValueError:
            return None
        return f if f == f else None
    return None


def mean(rows, key):
    """eval.py's _mean: blanks, None and NaN are skipped, not counted as zero."""
    v = [x for x in (num(r.get(key)) for r in rows) if x is not None]
    return sum(v) / len(v) if v else float('nan')


def summary(rows):
    rows = list(rows)
    out = {'n': len(rows)}
    for k in METRIC_COLS:
        if k not in _LABELS:
            out[k] = mean(rows, k)
    settled = [r for r in rows if num(r.get('settled'))]
    complete = [r for r in settled
                if num(r.get('decisive_revealed')) == num(r.get('decisive_total'))]
    done = {id(r) for r in complete}
    incomplete = [r for r in settled if id(r) not in done]
    unsettled = [r for r in rows if not num(r.get('settled'))]
    out['settled_by'] = dict(collections.Counter(r.get('settled_by') or 'unrecorded'
                                                 for r in settled))
    out['leak_episodes'] = sum(1 for r in rows if (num(r.get('leaks')) or 0) > 0)
    out['bottleneck'] = {
        'unsettled': {'n': len(unsettled), 'checks_frac': mean(unsettled, 'checks_frac')},
        'settled_incomplete': {'n': len(incomplete),
                               'checks_frac': mean(incomplete, 'checks_frac')},
        'settled_complete': {'n': len(complete),
                             'checks_frac': mean(complete, 'checks_frac'),
                             'success': mean(complete, 'success')},
    }
    return out


def summary_lines(tag, rows):
    """eval.py's SUMMARY block, line for line where the metric exists here."""
    s = summary(rows)
    out = ['', '-' * 78, 'SUMMARY  %s   (n=%d)' % (tag, s['n']), '-' * 78]
    if not s['n']:
        return out
    b = s['bottleneck']
    by = '  '.join('%s %d' % kv for kv in sorted(s['settled_by'].items())) or '-'
    out += [
        '  CHECKS',
        '    content checks passed (mean)   : %.3f / %.3f'
        % (s['content_passed'], s['content_total']),
        '    provenance checks passed (mean): %.3f / %.3f'
        % (s['prov_passed'], s['prov_total']),
        '    BOTH: checks passed (mean)     : %.3f / %.3f'
        % (s['checks_passed'], s['checks_total']),
        '    checks_frac (mean fraction)    : %.4f' % s['checks_frac'],
        '    success (ALL checks pass)      : %.4f' % s['success'],
        '  BEHAVIOUR',
        '    reveals (mean)                 : %.3f' % s['reveals'],
        '    decisive facts revealed        : %.3f / %.3f'
        % (s['decisive_revealed'], s['decisive_total']),
        '    settled rate                   : %.4f' % s['settled'],
        '    settled by                     : %s' % by,
        '    turns to settle (mean, settled): %.2f' % s['turns_to_settle'],
        '    turns used (mean, chair turns) : %.2f' % s['turns_used'],
        '  ADDRESSING   (stands in for ROUTING: there is no router here)',
        '    address precision              : %.4f' % s['address_precision'],
        '    address recall                 : %.4f' % s['address_recall'],
        '  GUARDRAIL    (stands in for INSIGHT: there is no insight head here)',
        '    leaks per episode (mean)       : %.4f   <- GUARDRAIL' % s['leaks'],
        '    episodes with a leak           : %d / %d' % (s['leak_episodes'], s['n']),
        '  BOTTLENECK DECOMPOSITION',
        '    unsettled episodes             : n=%-3d checks_frac=%.4f'
        % (b['unsettled']['n'], b['unsettled']['checks_frac']),
        '    settled, record INCOMPLETE     : n=%-3d checks_frac=%.4f'
        % (b['settled_incomplete']['n'], b['settled_incomplete']['checks_frac']),
        '    settled, record COMPLETE       : n=%-3d checks_frac=%.4f  success=%.4f'
        % (b['settled_complete']['n'], b['settled_complete']['checks_frac'],
           b['settled_complete']['success']),
        '    (COMPLETE = every decisive fact was revealed. The last row is the ceiling',
        '     pooling alone can reach; beating it takes better settlement text)',
    ]
    return out


# ------------------------------------------------------------------ comparison
def paired_bootstrap(base, arm, n_boot, seed=0):
    """eval.py's procedure, unchanged: bootstrap over the PAIRED per-scenario differences.
    Both arms run the same scenarios, so pairing removes the between-scenario variance
    that dominates this task."""
    rng = random.Random(seed)
    d = [a - b for a, b in zip(arm, base)]
    n = len(d)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    obs = sum(d) / n
    means = []
    for _ in range(n_boot):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    # two-sided p: how often does a mean-centred resample reach |obs|?
    hits = sum(1 for m in means if abs(m - obs) >= abs(obs))
    return obs, lo, hi, hits / n_boot


def compare_lines(rows_by_tag, baseline, n_boot=N_BOOT):
    """eval.py's comparison: paired table, paired bootstrap per metric, guardrail verdict.

    `rows_by_tag` maps an arm label to its episode_metrics rows. Only scenarios every arm
    ran are compared, so every number in the table is over the same set.
    """
    rows = {t: {r['uid']: r for r in rs if r.get('uid') is not None}
            for t, rs in rows_by_tag.items() if rs}
    if baseline not in rows or len(rows) < 2:
        return ['', '[compare] need at least two arms with episodes, including %r'
                % baseline]
    uids = set(rows[baseline])
    for t in rows:
        uids &= set(rows[t])
    uids = sorted(uids)
    other = [t for t in rows_by_tag if t != baseline and t in rows]
    w = max(12, max(len(t) for t in rows) + 2)

    out = ['', '=' * 92,
           'PAIRED COMPARISON   baseline = %s   n paired = %d' % (baseline, len(uids)),
           '=' * 92,
           '  %-26s' % 'metric' + ''.join('%*s' % (w, t) for t in [baseline] + other),
           '  ' + '-' * (26 + w * (1 + len(other)))]
    for key, higher in METRICS:
        vals = {}
        for t in [baseline] + other:
            v = [x for x in (num(rows[t][u].get(key)) for u in uids) if x is not None]
            vals[t] = (sum(v) / len(v)) if v else None
        if vals[baseline] is None:
            continue
        line = '  %-26s%*.4f' % (key, w, vals[baseline])
        for t in other:
            if vals[t] is None:
                line += '%*s' % (w, '-')
                continue
            better = vals[t] > vals[baseline] if higher else vals[t] < vals[baseline]
            line += '%*.4f%s' % (w - 1, vals[t], '*' if better else ' ')
        out.append(line)
    out.append('')
    out.append('  * = better than baseline')

    out += ['', '=' * 92,
            'PAIRED BOOTSTRAP vs %s   (%d resamples, 95%% CI)' % (baseline, n_boot),
            '=' * 92,
            '  %-16s%-24s%10s%20s%10s' % ('arm', 'metric', 'delta', '95% CI', 'p'),
            '  ' + '-' * 80]
    for t in other:
        for key, _higher in METRICS:
            pairs = [(num(rows[baseline][u].get(key)), num(rows[t][u].get(key)))
                     for u in uids]
            pairs = [(b, a) for b, a in pairs if b is not None and a is not None]
            if not pairs:
                continue
            obs, lo, hi, p = paired_bootstrap([b for b, _ in pairs],
                                              [a for _, a in pairs], n_boot)
            out.append('  %-16s%-24s%+10.4f   [%+.4f, %+.4f]%10.4f%s'
                       % (t[:15], key, obs, lo, hi, p, '  *' if p < 0.05 else ''))
    out += ['', '  * = p < 0.05. A CI that straddles zero means the arms are not '
                'distinguishable',
            '    at n=%d, whatever the point estimate says.' % len(uids)]

    bl = mean([rows[baseline][u] for u in uids], 'leaks')
    out += ['', '=' * 92,
            'GUARDRAIL   leaks (a speaker stating a private fact it was never shown)',
            '=' * 92]
    for t in other:
        al = mean([rows[t][u] for u in uids], 'leaks')
        if al != al or bl != bl:
            verdict = 'n/a'
        elif al <= bl * 1.5 + 0.01:
            verdict = 'OK'
        else:
            verdict = 'SUSPECT - the gain may rest on fabricated facts'
        out.append('  %-12s %.4f  vs  %s %.4f   -> %s' % (t, al, baseline, bl, verdict))
    out += ['  A checks_frac gain that comes with a large leak rise is fabrication, not',
            '  pooling. Reject that checkpoint rather than reporting it.']
    return out
