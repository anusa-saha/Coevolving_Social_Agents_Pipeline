"""Sections F and G of the metric catalogue, across every arm that has records.

Complements ppdpp_csa/compute_all_metrics.py (sections A-D, H, I) rather than replacing
it. Runs on CPU: everything here is derived from the stored transcripts and training
logs, so it is cheap and safe to re-run while training continues.

    python compute_extended_metrics.py
    python compute_extended_metrics.py --out extended.json

The HEADLINE is eval.py's metric set (csa_core.headline): checks passed, checks_frac,
success, reveals, settled, turns, leaks and the bottleneck decomposition, with eval.py's
paired bootstrap against the baseline. dca and disclosure are reported beside it.

WHAT IS DEDUCIBLE WITHOUT A MODEL, AND WHAT IS NOT
--------------------------------------------------
Computed here:
  F  distinct-1/2/3, utterance length, role-adherence violations, view-leakage rate
  G  act distribution, act entropy, act bigram matrix, return variance, policy loss,
     gradient norm, potential trajectory, sample efficiency, seed spread

Needs a GPU, so deliberately not attempted:
  F  perplexity steered vs unsteered (two forward passes per utterance)
  F  Sdiv -- embedding self-similarity, power-penalised (needs an embedding model)
  G  KL to the SFT policy (needs both policies resident)

Needs a judge or annotators, so out of scope for a deterministic script:
  F  G-Eval, human rubric, Srel

Not applicable to CSA, and reported as such rather than faked:
  E  the whole numeric-prediction block. Only 20 of 789 decision fields carry a NUMERIC
     gold value; the rest are categorical (a lane, a drug, a time). MAE / RMSE / MAPE /
     MPE / R2 / Pearson / Spearman / Kendall / PE@K / tolerance accuracy all need numeric
     pairs, and 45% of fields have a CATEGORICAL gold that cbar already scores exactly.
  E  expected calibration error -- CSA settlements carry no confidence to calibrate.
  F  BLEU / ROUGE -- no reference justifications exist to compare against.
"""
import argparse
import ast
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from csa_core import data_csa                        # noqa: E402
from csa_core.detectors import content_tokens, overlap   # noqa: E402

from csa_core import headline as H                   # noqa: E402
import metrics_ext as MX                             # noqa: E402

CASES = data_csa.case_index()


# ------------------------------------------------------------------ loading
def load_records(path):
    out = []
    for blk in open(path, encoding='utf-8').read().split('\n\n'):
        blk = blk.strip()
        if blk:
            try:
                out.append(ast.literal_eval(blk))
            except Exception:                        # noqa: BLE001
                pass
    return out


def load_jsonl(path):
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:                        # noqa: BLE001
                pass
    return out


def records_root():
    """Where evaluation records live.

    They are results, so they are not in the repo. CSA_ARTIFACTS_DIR wins; otherwise the
    sibling csa-artifacts/ the reorganisation created; otherwise the repo itself, which
    is what you get once you have re-run an arm and its logs/ is populated again.
    """
    env = os.environ.get('CSA_ARTIFACTS_DIR')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    sib = os.path.join(os.path.dirname(HERE), 'csa-artifacts')
    return os.path.abspath(sib) if os.path.isdir(sib) else HERE


def discover():
    """(arm label -> records). Every arm writes the same record schema.

    Searched in both the artifacts archive and the repo, so an arm you have just re-run
    is picked up from its own logs/ without moving anything.
    """
    arms = {}
    roots = list(dict.fromkeys([records_root(), HERE]))
    pats = [
        ('PPDPP', 'ppdpp/tmp/csa/eval_result/Record-epoch-*.txt',
         lambda f: 'PPDPP ' + re.search(r'epoch-(\d+)', f).group(1)),
        ('EPO', 'epo/logs/Record-*seed1-ep*.txt',
         lambda f: 'EPO ' + re.search(r'-ep(\d+)\.txt$', f).group(1)),
        ('SR', 'sotopia_rl/logs/Record-*.txt',
         lambda f: 'SR ' + re.sub(r'^Record-|-\w+\.txt$', '', os.path.basename(f))),
        ('ToM', 'sotopia_tom/logs/Record-tom-*.txt',
         lambda f: 'ToM ' + re.sub(r'^Record-tom-|-\w+\.txt$', '', os.path.basename(f))),
        ('OMEGA', 'sotopia_omega/logs/Record-*.txt',
         lambda f: 'OM ' + re.sub(r'^Record-|-\w+\.txt$', '', os.path.basename(f))),
        ('RT', 'roundtable/logs/Record-rt-*.txt',
         lambda f: 'RT ' + '/'.join(os.path.basename(f)[:-4].split('-')[2:4])),
    ]
    for _fam, pat, namer in pats:
        hits = []
        for root in roots:
            hits += sorted(glob.glob(os.path.join(root, pat)))
        for f in hits:
            try:
                label = namer(f)
            except Exception:                        # noqa: BLE001
                continue
            recs = load_records(f)
            if recs:
                arms[label] = recs
    return arms


# ------------------------------------------------------------------ F: language
def speaker_of(rec, turn):
    """Which role a turn belongs to: 'sys' (chair), 'usr' (advisor) or 'env'.

    sotopia_rl / _tom / _omega write a `speaker` key. PPDPP and EPO store
    `self.conversation` directly, which has only role and content -- so for those the
    chair has to be recovered by matching the role name against the case's
    decision_maker. Assuming the key exists silently produced zero chair turns for two
    of the four arms.
    """
    if turn.get('speaker'):
        return turn['speaker']
    case = CASES.get(rec.get('uid'))
    if not case:
        return 'usr'
    if turn.get('role') == 'Meeting':
        return 'env'
    chair = next((a['name'] for a in case['agents']
                  if a['agent_id'] == case['decision_maker']), None)
    return 'sys' if turn.get('role') == chair else 'usr'


def chair_turns(rec):
    return [t['content'] for t in (rec.get('dialog') or [])
            if speaker_of(rec, t) == 'sys']


def distinct_n(texts, n):
    """Distinct-n: unique n-grams over total n-grams. Parroting shows up as a low value
    and needs no judge to detect."""
    total, uniq = 0, set()
    for t in texts:
        tok = content_tokens(t)
        for i in range(len(tok) - n + 1):
            g = tuple(tok[i:i + n])
            uniq.add(g)
            total += 1
    return (len(uniq) / total) if total else float('nan')


def role_adherence(rec):
    """An agent speaking as, or addressing, itself.

    Cheap proxy for the role confusion visible in the self-play corpus: a turn whose
    opening names its own speaker. Not the full definition (that would need to detect an
    agent speaking another's lines) but it is the part that is lexically decidable.
    """
    bad = tot = 0
    for t in (rec.get('dialog') or []):
        if speaker_of(rec, t) not in ('sys', 'usr'):
            continue
        tot += 1
        surname = (t.get('role') or '').split()[-1:] or ['']
        if surname[0] and surname[0].lower() in t['content'][:60].lower():
            bad += 1
    return bad, tot


def view_leakage(rec):
    """A speaker stating a fact its view never contained.

    This is `leaks` as the environment records it. It should be exactly zero for a
    correct prompt construction, so it doubles as a test of our own view filtering.
    """
    return len(rec.get('leaks') or [])


def section_F(recs):
    chair = [c for r in recs for c in chair_turns(r)]
    lens = [len(c.split()) for c in chair if c.strip()]
    ra = [role_adherence(r) for r in recs]
    bad, tot = sum(a for a, _b in ra), sum(b for _a, b in ra)
    return {
        'n_chair_turns': len(chair),
        'distinct_1': distinct_n(chair, 1),
        'distinct_2': distinct_n(chair, 2),
        'distinct_3': distinct_n(chair, 3),
        'utterance_len_mean': st.mean(lens) if lens else float('nan'),
        'utterance_len_median': st.median(lens) if lens else float('nan'),
        'utterance_len_sd': st.pstdev(lens) if len(lens) > 1 else 0.0,
        'role_adherence_violations': bad,
        'role_adherence_rate': (bad / tot) if tot else float('nan'),
        'view_leakage_episodes': sum(1 for r in recs if view_leakage(r)),
        'view_leakage_total': sum(view_leakage(r) for r in recs),
    }


# ------------------------------------------------------------------ G: policy
def entropy(counter):
    n = sum(counter.values())
    if not n:
        return float('nan')
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def section_G(recs):
    acts = collections.Counter(a for r in recs for a in (r.get('act_history') or []))
    bigrams = collections.Counter()
    for r in recs:
        h = r.get('act_history') or []
        for a, b in zip(h, h[1:]):
            bigrams['%s->%s' % (a, b)] += 1
    out = {
        'act_distribution': dict(acts),
        'act_entropy_bits': entropy(acts) if acts else None,
        'act_entropy_max_bits': math.log2(len(acts)) if len(acts) > 1 else 0.0,
        'act_bigrams': dict(bigrams.most_common(12)) or None,
        'n_acts_used': len(acts) or None,
    }
    rew = [r.get('reward') for r in recs if isinstance(r.get('reward'), (int, float))]
    out['return_mean'] = st.mean(rew) if rew else float('nan')
    out['return_variance'] = st.pvariance(rew) if len(rew) > 1 else 0.0
    # sample efficiency needs a success signal that actually moves; joint success is 0
    # for every arm on this benchmark, so it is reported as undefined rather than 0.
    js = [1.0 if r.get('done') == 1 else 0.0 for r in recs]
    out['success_rate'] = st.mean(js) if js else float('nan')
    return out


def training_health():
    """Policy loss, gradient norm and potential trajectory, from the training logs."""
    out = {}
    for label, pat, keys in (
            ('EPO', 'epo/logs/*-history.jsonl', ('loss', 'grad_norm')),
            ('SR', 'sotopia_rl/logs/*-history.jsonl', ('loss', 'grad_norm'))):
        for f in glob.glob(os.path.join(HERE, pat)):
            rows = load_jsonl(f)
            if not rows:
                continue
            d = {}
            for k in keys:
                v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                if v:
                    d[k + '_mean'] = st.mean(v)
                    d[k + '_sd'] = st.pstdev(v) if len(v) > 1 else 0.0
            sc = [s for r in rows for s in (r.get('scores') or [])]
            if sc:
                d['group_score_mean'] = st.mean(sc)
                d['group_score_var'] = st.pvariance(sc) if len(sc) > 1 else 0.0
            col = [r for r in rows if r.get('collapsed')]
            if rows and 'collapsed' in rows[0]:
                d['collapsed_frac'] = len(col) / len(rows)
            d['updates'] = len(rows)
            out['%s %s' % (label, os.path.basename(f))] = d
    return out


def potential_trajectory():
    """Phi(s_t) against t, parsed from the PPDPP run log.

    The potentials are printed per step by the verifier arm and never written to the
    records, so this is the only place they survive. Shows whether shaping ever fired.
    """
    out = {}
    pat = re.compile(r"'disclosure': ([\d.]+), 'elicitation': ([\d.]+), 'coverage': ([\d.]+)")
    for f in glob.glob(os.path.join(HERE, 'ppdpp/ppdpp_csa/logs/rl-arm*.log')):
        d, e, c = [], [], []
        try:
            for line in open(f, encoding='utf-8', errors='replace'):
                m = pat.search(line)
                if m:
                    d.append(float(m.group(1)))
                    e.append(float(m.group(2)))
                    c.append(float(m.group(3)))
        except Exception:                            # noqa: BLE001
            continue
        if d:
            out[os.path.basename(f)] = {
                'steps_logged': len(d),
                'disclosure_mean': st.mean(d),
                'disclosure_zero_frac': sum(1 for x in d if x == 0.0) / len(d),
                'elicitation_mean': st.mean(e),
                'coverage_mean': st.mean(c),
                'shaping_fired': any(x > 0 for x in d),
            }
    return out


def seed_spread(arms):
    """Std of the final success rate across seeds. One seed means this is undefined, and
    saying so is more useful than printing 0.0."""
    by_seed = collections.defaultdict(list)
    for label, recs in arms.items():
        m = re.search(r'seed(\d+)', label)
        by_seed[m.group(1) if m else 'unknown'].append(
            st.mean([1.0 if r.get('done') == 1 else 0.0 for r in recs]))
    if len(by_seed) < 2:
        return {'seeds': list(by_seed), 'note': 'one seed only; spread is undefined'}
    finals = [v[-1] for v in by_seed.values()]
    return {'seeds': list(by_seed), 'sd_final_success': st.pstdev(finals)}


# ------------------------------------------------------------------ report
def fmt(x, d=4):
    if x is None:
        return '-'
    if isinstance(x, float):
        return '-' if x != x else ('%.' + str(d) + 'f') % x
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(HERE, 'extended_metrics.json'))
    cli = p.parse_args()

    arms = discover()
    if not arms:
        raise SystemExit('no record files found under %s' % HERE)

    res = {}
    rows_by_arm = {}
    for label, recs in arms.items():
        rows_by_arm[label] = H.rows_from_records(recs, CASES)
        res[label] = {'n_episodes': len(recs),
                      'E': MX.section_E(recs, CASES),
                      'F': section_F(recs),
                      'G': section_G(recs),
                      'H': MX.section_H(recs),
                      'headline': H.summary(rows_by_arm[label]),
                      'dca_disclosure': MX.headline(recs)}

    print('=' * 108)
    print('F. LANGUAGE AND JUSTIFICATION QUALITY   (chair turns only)')
    print('=' * 108)
    hdr = ('arm', 'turns', 'dist-1', 'dist-2', 'dist-3', 'len mean', 'len sd',
           'role viol', 'leaks')
    print('%-14s %6s %8s %8s %8s %9s %8s %10s %7s' % hdr)
    for label in sorted(res):
        f = res[label]['F']
        print('%-14s %6d %8s %8s %8s %9.1f %8.1f %10s %7d'
              % (label, f['n_chair_turns'], fmt(f['distinct_1'], 3),
                 fmt(f['distinct_2'], 3), fmt(f['distinct_3'], 3),
                 f['utterance_len_mean'], f['utterance_len_sd'],
                 '%d (%.1f%%)' % (f['role_adherence_violations'],
                                  100 * f['role_adherence_rate']),
                 f['view_leakage_total']))

    print()
    print('=' * 108)
    print('G. POLICY DIAGNOSTICS')
    print('=' * 108)
    print('%-14s %6s %10s %8s %10s %12s %10s' % ('arm', 'acts', 'entropy', 'max', 'SR',
                                                 'return var', 'return mean'))
    for label in sorted(res):
        g = res[label]['G']
        print('%-14s %6s %10s %8s %10s %12s %10s'
              % (label, fmt(g['n_acts_used']), fmt(g['act_entropy_bits'], 3),
                 fmt(g['act_entropy_max_bits'], 2), fmt(g['success_rate'], 3),
                 fmt(g['return_variance'], 4), fmt(g['return_mean'], 4)))

    print('\nact distribution and top bigrams')
    for label in sorted(res):
        g = res[label]['G']
        if g['act_distribution']:
            print('  %-12s %s' % (label, g['act_distribution']))
            if g['act_bigrams']:
                print('  %-12s %s' % ('', dict(list(g['act_bigrams'].items())[:5])))

    th = training_health()
    if th:
        print('\ntraining health (from *-history.jsonl)')
        for k, v in sorted(th.items()):
            print('  %s' % k)
            for kk, vv in sorted(v.items()):
                print('      %-22s %s' % (kk, fmt(vv)))
        res['_training_health'] = th

    pt = potential_trajectory()
    if pt:
        print('\npotential trajectory Phi(s_t)  (PPDPP verifier arm only; never stored '
              'in records)')
        for k, v in sorted(pt.items()):
            print('  %s' % k)
            for kk, vv in sorted(v.items()):
                print('      %-22s %s' % (kk, fmt(vv)))
        res['_potential_trajectory'] = pt

    ss = seed_spread(arms)
    res['_seed_spread'] = ss
    print('\nseed spread: %s' % ss)

    base = next((k for k in sorted(arms) if k.startswith('PPDPP')), sorted(arms)[0])

    print()
    print('=' * 108)
    print("HEADLINE   eval.py's metrics (bootstrap 95% CI, 10k resamples)")
    print('=' * 108)
    print('%-14s %5s %22s %22s %8s %8s %9s %8s %9s %7s'
          % ('arm', 'n', 'checks_frac [95% CI]', 'success [95% CI]', 'content', 'prov',
             'decisive', 'settled', 't_settle', 'leaks'))
    for label in sorted(arms):
        rows, h = rows_by_arm[label], res[label]['headline']
        cf = MX.bootstrap_ci([H.num(r['checks_frac']) for r in rows])
        sc = MX.bootstrap_ci([H.num(r['success']) for r in rows])
        h['checks_frac_ci'], h['success_ci'] = cf, sc
        print('%-14s %5d %22s %22s %8s %8s %9s %8s %9s %7s'
              % (label, h['n'],
                 '%s [%s, %s]' % (fmt(h['checks_frac'], 3), fmt(cf[0], 3), fmt(cf[1], 3)),
                 '%s [%s, %s]' % (fmt(h['success'], 3), fmt(sc[0], 3), fmt(sc[1], 3)),
                 fmt(h['content_passed'], 2), fmt(h['prov_passed'], 2),
                 fmt(h['decisive_revealed'], 2), fmt(h['settled'], 3),
                 fmt(h['turns_to_settle'], 2), fmt(h['leaks'], 3)))
    for label in sorted(arms):
        for line in H.summary_lines(label, rows_by_arm[label]):
            print(line)
    for line in H.compare_lines({l: rows_by_arm[l] for l in sorted(arms)}, base):
        print(line)

    print()
    print('=' * 108)
    print('DCA AND DISCLOSURE   (bootstrap 95% CI, 10k resamples; gain normalised per '
          'scenario)')
    print('=' * 108)
    print('%-14s %5s %22s %22s %7s %11s %10s'
          % ('arm', 'n', 'dca [95% CI]', 'disclosure [95% CI]', 'calls',
             'dca/100call', 'norm gain'))
    for label in sorted(arms):
        h = res[label]['dca_disclosure']
        print('%-14s %5d %22s %22s %7s %11s %10s'
              % (label, h['n'],
                 '%s [%s, %s]' % (fmt(h['dca'], 3), fmt(h['dca_ci'][0], 3),
                                  fmt(h['dca_ci'][1], 3)),
                 '%s [%s, %s]' % (fmt(h['disclosure'], 3), fmt(h['disclosure_ci'][0], 3),
                                  fmt(h['disclosure_ci'][1], 3)),
                 fmt(h['calls'], 1), fmt(h['dca_per_100_calls'], 3),
                 fmt(h['normalised_gain'], 3)))

    print('\npaired vs %s  (sign test + Cliff\'s delta)' % base)
    print('%-14s %-16s %5s %5s %5s %9s %8s %-12s'
          % ('arm', 'metric', 'win', 'tie', 'loss', 'p', 'delta', 'effect'))
    for label in sorted(arms):
        if label == base:
            continue
        for key in ('dca', 'disclosure_rate'):
            r = MX.paired_sign(arms[label], arms[base], key)
            print('%-14s %-16s %5d %5d %5d %9s %8s %-12s'
                  % (label, key, r['win'], r['tie'], r['loss'],
                     ('<0.001' if r['p'] < 0.001 else fmt(r['p'], 3)),
                     fmt(r['cliffs_delta'], 3), r['effect']))
    res['_paired'] = {l: {k: MX.paired_sign(arms[l], arms[base], k)
                          for k in ('dca', 'disclosure_rate')}
                      for l in arms if l != base}

    fam = {'%s|%s' % (l, k): res['_paired'][l][k]['p']
           for l in res['_paired'] for k in res['_paired'][l]}
    adj = MX.holm(fam)
    res['_holm'] = adj
    print()
    print('paired difference vs %s: bootstrap CI on the DELTA, Holm-corrected p' % base)
    print('  %d tests in this family. The delta CI is the interval to quote -- both arms'
          % len(fam))
    print('  see the same 42 scenarios, so the paired difference is far tighter than')
    print('  either arm own mean interval, and can exclude zero when those overlap.')
    print('%-14s %-16s %11s %22s %9s %9s %-4s'
          % ('arm', 'metric', 'mean delta', 'delta 95% CI', 'p', 'p (Holm)', 'sig'))
    for label in sorted(arms):
        if label == base:
            continue
        for key in ('dca', 'disclosure_rate'):
            d = MX.paired_delta(arms[label], arms[base], key)
            pa = adj.get('%s|%s' % (label, key), float('nan'))
            praw = res['_paired'][label][key]['p']
            ci = d.get('ci', (float('nan'), float('nan')))
            print('%-14s %-16s %11s %22s %9s %9s %-4s'
                  % (label, key, fmt(d.get('mean_delta'), 3),
                     '[%s, %s]' % (fmt(ci[0], 3), fmt(ci[1], 3)),
                     ('<0.001' if praw < 0.001 else fmt(praw, 3)),
                     ('<0.001' if pa < 0.001 else fmt(pa, 3)),
                     ('yes' if pa == pa and pa < 0.05 else 'no')))
            res['_paired'][label][key]['delta'] = d
            res['_paired'][label][key]['p_holm'] = pa

    wm = MX.win_matrix(arms, 'dca')
    res['_win_matrix_dca'] = wm
    labels = sorted(wm)
    print('\nhead-to-head win rate on dca  (row beats column, per scenario)')
    print('%-14s %s' % ('', ' '.join('%9s' % l[:9] for l in labels)))
    for a in labels:
        print('%-14s %s' % (a, ' '.join(
            '%9s' % ('  -' if wm[a][b] is None else fmt(wm[a][b], 2)) for b in labels)))

    print()
    print('=' * 108)
    print('E. PREDICTION AGAINST RECOVERED GOLD')
    print('=' * 108)
    print('gold answers are recovered from content checks of the form')
    print("  decisions['x'] == 'v'   /   decisions['x'] in [...]")
    print('CSA ships no gold settlement, so this is the only route to a predicted-vs-gold')
    print('comparison. 355 of 789 decision fields yield a gold; only 20 parse as numeric.')
    print()
    print('%-14s %7s %10s %9s %9s %9s %8s %8s %8s'
          % ('arm', 'cat n', 'exact acc', 'num n', 'PE@10', 'PE@20', 'PE@30', 'MAPE',
             'MAE'))
    for label in sorted(arms):
        e = res[label]['E']
        print('%-14s %7d %10s %9d %9s %9s %8s %8s %8s'
              % (label, e.get('n_categorical_pairs', 0),
                 fmt(e.get('exact_match_accuracy'), 3), e.get('n_numeric_pairs', 0),
                 fmt(e.get('PE@10'), 3), fmt(e.get('PE@20'), 3), fmt(e.get('PE@30'), 3),
                 fmt(e.get('MAPE'), 3), fmt(e.get('MAE'), 3)))
    any_num = max((res[l]['E'].get('n_numeric_pairs', 0) for l in arms), default=0)
    if any_num < 30:
        print()
        print('  NOTE: at n=%d numeric pairs the PE@K / MAPE / MAE / R2 / correlation'
              % any_num)
        print('  figures are descriptive only. CSA settlements are categorical, so the')
        print('  exact-match column is the meaningful one and cbar already scores it.')

    print('\n' + '=' * 108)
    print()
    print('=' * 108)
    print('H. STRATIFIED BREAKDOWN, DISPERSION AND SUBSCORES')
    print('=' * 108)

    doms = sorted({d for l in arms for d in res[l]['H']['by_domain']})
    print('dca by domain -- a gain that lives in one domain is not a general gain')
    print('%-14s %s %10s' % ('arm', ' '.join('%18s' % d[:18] for d in doms), 'macro avg'))
    for label in sorted(arms):
        bd = res[label]['H']['by_domain']
        print('%-14s %s %10s' % (label, ' '.join(
            '%18s' % (('%s  n=%d' % (fmt(bd[d]['mean'], 3), bd[d]['n'])) if d in bd
                      else '-') for d in doms),
            fmt(res[label]['H']['macro_by_domain'], 3)))

    nags = sorted({a for l in arms for a in res[l]['H']['by_num_agents']})
    print()
    print('dca by table size')
    print('%-14s %s' % ('arm', ' '.join('%16s' % ('%s agents' % a) for a in nags)))
    for label in sorted(arms):
        ba = res[label]['H']['by_num_agents']
        print('%-14s %s' % (label, ' '.join(
            '%16s' % (('%s n=%d' % (fmt(ba[a]['mean'], 3), ba[a]['n'])) if a in ba else '-')
            for a in nags)))

    print()
    print('per-scenario spread of dca -- the mean alone hides the shape')
    print('%-14s %6s %6s %6s %6s %6s %6s %6s %9s %9s'
          % ('arm', 'sd', 'min', 'q25', 'med', 'q75', 'max', 'iqr', 'frac at 0',
             'frac at 1'))
    for label in sorted(arms):
        d = res[label]['H']['dispersion_dca']
        if not d:
            continue
        print('%-14s %6s %6s %6s %6s %6s %6s %6s %9s %9s'
              % (label, fmt(d['sd'], 3), fmt(d['min'], 2), fmt(d['q25'], 2),
                 fmt(d['median'], 2), fmt(d['q75'], 2), fmt(d['max'], 2),
                 fmt(d['iqr'], 2), fmt(d['frac_zero'], 3), fmt(d['frac_one'], 3)))

    keys = [k for k in MX.SUBSCORES
            if any(k in res[l]['H']['subscores'] for l in arms)]
    print()
    print('verifier subscores, mean [95% CI]')
    for label in sorted(arms):
        sub = res[label]['H']['subscores']
        print('  %s' % label)
        for k in keys:
            if k in sub:
                print('    %-20s %8s  [%s, %s]   n=%d'
                      % (k, fmt(sub[k]['mean'], 3), fmt(sub[k]['ci'][0], 3),
                         fmt(sub[k]['ci'][1], 3), sub[k]['n']))

    print()
    print('does disclosure buy accuracy? correlation across the scenarios')
    print('%-14s %6s %10s %10s %10s' % ('arm', 'n', 'pearson', 'spearman', 'kendall'))
    for label in sorted(arms):
        c = res[label]['H']['disclosure_vs_accuracy']
        if c.get('n', 0) < 3:
            continue
        print('%-14s %6d %10s %10s %10s'
              % (label, c['n'], fmt(c['pearson_r'], 3), fmt(c['spearman_rho'], 3),
                 fmt(c['kendall_tau'], 3)))

    print()
    print('dca by scenario difficulty, tertiles set by %s (arm vs base)' % base)
    print('%-14s %s' % ('arm', ' '.join('%24s' % t for t in ('hard', 'medium', 'easy'))))
    for label in sorted(arms):
        if label == base:
            continue
        t = MX.difficulty_tertiles(arms[label], arms[base])
        if not t:
            continue
        print('%-14s %s' % (label, ' '.join(
            '%24s' % ('%s vs %s  n=%d' % (fmt(t[k]['arm'], 3), fmt(t[k]['base'], 3),
                                          t[k]['n'])) for k in ('hard', 'medium', 'easy'))))

    print()
    print('=' * 108)
    print('OMITTED, AND WHY')
    print('=' * 108)
    for line in (
        'E  numeric block is COMPUTED above, but on very few pairs: only 20 of 789',
        '     decision fields carry a numeric gold. Read it with its n.',
        'E  expected calibration error -- settlements carry no confidence to calibrate.',
        'E  bin confusion matrix -- no bins are defined for categorical settlements.',
        'F  G-Eval, human rubric, Srel -- need a judge model or annotators.',
        'F  BLEU / ROUGE -- no reference justifications exist.',
        'F  Sdiv, perplexity steered vs unsteered -- need a GPU forward pass.',
        'G  KL to the SFT policy -- needs both policies resident on a GPU.',
        'G  sample efficiency (episodes to 90% of final joint success) -- joint success',
        '     is 0.0 for every arm, so the target is undefined.',
    ):
        print('  ' + line)

    with open(cli.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=str)
    print('\nwrote %s' % cli.out)


if __name__ == '__main__':
    main()
