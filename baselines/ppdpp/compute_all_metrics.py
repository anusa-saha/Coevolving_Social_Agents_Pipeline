"""Compute the metric catalogue from RL episode records.

Structured the way the catalogue asks to be reported: a headline set of six, a main table
of roughly fifteen, and everything else relegated to an appendix block. Nothing here
re-runs a model -- every number comes from the records written during evaluation, so this
is cheap to re-run and safe to run while training continues.

Sections F (G-Eval, BLEU/ROUGE, Sdiv, perplexity) and J (Wasserstein, KDE) are NOT
computed: they need reference justifications, a judge model, or a second distribution to
compare against, none of which exist yet. They are listed as omitted rather than silently
dropped.

Every metric is individually guarded. A malformed record or a missing field degrades one
number, not the whole report.

    python compute_all_metrics.py --records tmp/csa/eval_result --out logs/catalogue.json
"""
import argparse
import ast
import collections
import glob
import json
import math
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load_records(path):
    """Records are written as repr(dict) separated by blank lines. Utterances contain
    blank lines too, so splitting on '\\n\\n' alone corrupts them -- fragments are
    rejoined until they parse."""
    text = open(path, encoding='utf-8', errors='replace').read()
    parts, buf, out = text.split('\n\n'), '', []
    for p in parts:
        buf = (buf + '\n\n' + p) if buf else p
        s = buf.strip()
        if not s.startswith('{'):
            buf = ''
            continue
        try:
            out.append(ast.literal_eval(s))
            buf = ''
        except Exception:
            continue
    return out


def mean(xs, d=float('nan')):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else d


def pct(x):
    return 'n/a' if x != x else '%.4f' % x


def entropy(counter):
    n = sum(counter.values())
    if not n:
        return float('nan')
    return -sum((v / n) * math.log2(v / n) for v in counter.values() if v)


def gini(counter):
    v = sorted(counter.values())
    n = len(v)
    if n == 0 or sum(v) == 0:
        return float('nan')
    cum = sum((2 * i - n - 1) * x for i, x in enumerate(v, 1))
    return cum / (n * sum(v))


def safe(fn, default=float('nan')):
    try:
        r = fn()
        return default if r is None else r
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', default='tmp/csa/eval_result')
    p.add_argument('--glob', default='Record-epoch-*.txt')
    p.add_argument('--epoch', default='', help='only this epoch; default = latest')
    p.add_argument('--out', default='logs/catalogue.json')
    cli = p.parse_args()

    files = sorted(glob.glob(os.path.join(cli.records, cli.glob)))
    if not files:
        raise SystemExit('no record files under %s' % cli.records)

    def epoch_of(f):
        m = re.search(r'Record-epoch-(\d+)-', os.path.basename(f))
        return int(m.group(1)) if m else -1

    by_epoch = collections.defaultdict(list)
    for f in files:
        by_epoch[epoch_of(f)] += load_records(f)
    epochs = sorted(by_epoch)
    target = int(cli.epoch) if cli.epoch else epochs[-1]
    R = by_epoch[target]
    print('record files : %d, epochs present: %s' % (len(files), epochs))
    print('scoring epoch: %d  (%d episodes)\n' % (target, len(R)))
    if not R:
        raise SystemExit('no parseable records for epoch %s' % target)

    res = {'epoch': target, 'n_episodes': len(R), 'epochs_present': epochs}

    # Which facts are decisive, and which checks they flip, live in the CASE, not in the
    # score dict. Joining on uid recovers them without re-running anything.
    cases = {}
    try:
        from utils import load_dataset
        for sp in ('train', 'valid', 'test'):
            for c in load_dataset('csa')[sp]:
                cases[c['uid']] = c
    except Exception as e:
        # utils.py imports torch at module level, so on a box without a GPU stack the
        # whole of section B used to be silently dropped -- which contradicts this
        # script's own promise that it re-runs no model. The split files are plain
        # repr(dict) per line, so read them directly instead.
        print('note: utils.load_dataset unavailable (%s); reading splits directly' % e)
        for sp in ('train', 'valid', 'test'):
            path = next((p for p in ('./data/csa-%s.txt' % sp, '../data/csa-%s.txt' % sp,
                                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'data', 'csa-%s.txt' % sp))
                         if os.path.exists(p)), None)
            if not path:
                continue
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            c = ast.literal_eval(line)
                            cases[c['uid']] = c
                        except Exception:            # noqa: BLE001
                            pass
        if not cases:
            print('WARNING: no cases loaded; section B will be empty')
        else:
            print('loaded %d cases without torch' % len(cases))

    def case_of(r):
        return cases.get(r.get('uid'), {})

    def decisive_of(r):
        return [d['fact_id'] for d in (case_of(r).get('decisive_facts') or [])]

    def flips_of(r):
        return {c for d in (case_of(r).get('decisive_facts') or [])
                for c in (d.get('flips') or [])}

    def owner_of(r, fid):
        return ((case_of(r).get('private_facts') or {}).get(fid) or {}).get('owner')

    # floor_score returns a full score dict, not a scalar
    def floor_val(r, key):
        f = r.get('floor')
        return f.get(key) if isinstance(f, dict) else (f if key == 'joint' else None)

    # ---------------------------------------------------------------- helpers
    def sc(r):
        return r.get('score') or {}

    def checks(r, key):
        d = sc(r).get(key)
        return d if isinstance(d, dict) else {}

    def allpass(d):
        return bool(d) and all(bool(v) for v in d.values())

    # ================================================== A. TASK OUTCOME
    # score() already computes these correctly; prefer them and only fall back to
    # recomputing from the raw check dicts when an older record lacks the field.
    joint = [1.0 if sc(r).get('joint', allpass(checks(r, 'content'))
                              and allpass(checks(r, 'provenance'))) else 0.0 for r in R]
    settle = [1.0 if sc(r).get('all_content', allpass(checks(r, 'content'))) else 0.0
              for r in R]
    prov = [1.0 if sc(r).get('all_prov', allpass(checks(r, 'provenance'))) else 0.0
            for r in R]

    c_all = [v for r in R for v in checks(r, 'content').values()]
    p_all = [v for r in R for v in checks(r, 'provenance').values()]
    c_micro = mean([1.0 if v else 0.0 for v in c_all])
    p_micro = mean([1.0 if v else 0.0 for v in p_all])
    c_macro = mean([sc(r).get('cbar') for r in R])
    p_macro = mean([sc(r).get('pbar') for r in R])

    # per-check pass rate: identifies unreachable checks (dataset artefacts)
    per_check = collections.defaultdict(list)
    for r in R:
        for k, v in list(checks(r, 'content').items()) + list(checks(r, 'provenance').items()):
            per_check[k].append(1.0 if v else 0.0)
    per_check_rate = {k: mean(v) for k, v in per_check.items()}
    unreachable = sorted([k for k, v in per_check_rate.items() if v == 0.0])

    # DCA: only checks that a decisive fact flips
    dca_num, dca_den, nd_num, nd_den = 0, 0, 0, 0
    for r in R:
        flips = flips_of(r)
        both = dict(checks(r, 'content')); both.update(checks(r, 'provenance'))
        for k, v in both.items():
            if k in flips:
                dca_den += 1; dca_num += 1 if v else 0
            else:
                nd_den += 1; nd_num += 1 if v else 0
    dca = dca_num / dca_den if dca_den else float('nan')
    nondecisive = nd_num / nd_den if nd_den else float('nan')

    floor_dca = mean([floor_val(r, 'dca') for r in R])
    floor_c = mean([floor_val(r, 'cbar') for r in R])
    floor_joint = mean([1.0 if floor_val(r, 'joint') else 0.0 for r in R])
    floor_mean = floor_dca          # NDG normalises against the DECISIVE floor
    # NDG against the shared-context floor
    ndg = safe(lambda: (dca - floor_dca) / (1 - floor_dca)
               if floor_dca == floor_dca and floor_dca < 1 else float('nan'))
    ng_all = safe(lambda: (c_micro - floor_c) / (1 - floor_c)
                  if floor_c == floor_c and floor_c < 1 else float('nan'))

    res['A_task_outcome'] = {
        'joint_success_rate': mean(joint), 'settlement_success_rate': mean(settle),
        'provenance_success_rate': mean(prov),
        'C_micro': c_micro, 'C_macro': c_macro, 'P_micro': p_micro, 'P_macro': p_macro,
        'DCA': dca, 'non_decisive_accuracy': nondecisive,
        'NDG': ndg, 'normalised_gain_all_checks': ng_all,
        'floor_dca': floor_dca, 'floor_cbar': floor_c, 'floor_joint': floor_joint,
        'per_check_pass_rate': per_check_rate,
        'unreachable_checks': unreachable,
        'score_distribution': collections.Counter(
            round(sc(r).get('cbar') or 0, 1) for r in R),
    }

    # commitments
    comm_present, comm_wf = [], []
    for r in R:
        st = r.get('settlement') or {}
        cs = st.get('commitments') or []
        comm_present.append(1.0 if cs else 0.0)
        if cs:
            ok = sum(1 for c in cs if isinstance(c, dict)
                     and c.get('owner') and c.get('action') and c.get('deadline'))
            comm_wf.append(ok / len(cs))
    res['A_task_outcome']['commitment_coverage'] = mean(comm_present)
    res['A_task_outcome']['commitment_wellformedness'] = mean(comm_wf)

    # ================================================== B. INFORMATION POOLING
    disc, credit, complete, over, elicited = [], [], [], [], []
    prec_l, rec_l, silent, halluc, contain, premature = [], [], [], [], [], []
    contributors = collections.Counter()
    latency, survival = [], []
    for r in R:
        s = sc(r)
        decisive = decisive_of(r)
        revealed = set(r.get('revealed') or [])
        st = r.get('settlement') or {}
        credited = set(st.get('credited_facts') or [])
        just = set(st.get('justification_fact_ids') or [])
        dset = set(decisive)
        if dset:
            hit = dset & revealed
            disc.append(len(hit) / len(dset))
            complete.append(1.0 if dset <= revealed else 0.0)
            credit.append(len(dset & revealed & credited & just) / len(dset)
                          if (credited or just) else 0.0)
            premature.append(1.0 if (r.get('done') == 1 and not dset <= revealed) else 0.0)
            silent.append(len(dset - revealed) / len(dset))
        if revealed:
            prec_l.append(len(dset & revealed) / len(revealed))
            over.append(len([f for f in revealed if f not in dset]) / len(revealed))
        rec_l.append(len(dset & revealed) / len(dset) if dset else float('nan'))
        el = r.get('reveal_elicited') or {}
        if dset:
            elicited.append(mean([1.0 if el.get(f) else 0.0 for f in dset if f in revealed],
                                 float('nan')))
        for f in dset & revealed:
            o = owner_of(r, f)
            if o:
                contributors[o] += 1
        halluc.append(1.0 if ((credited | just) - revealed) else 0.0)
        contain.append(1.0 if (just <= credited and credited <= revealed) else 0.0)
        rt = r.get('reveal_turn') or {}
        cap = r.get('max_turn') or 1
        for f in dset & revealed:
            if f in rt:
                latency.append(rt[f] / max(1, cap))
        if rt:
            survival.append(mean([1 - (rt.get(f, cap) / max(1, cap)) for f in dset])
                            if dset else float('nan'))

    dprec, drec = mean(prec_l), mean([x for x in rec_l if x == x])
    res['B_pooling'] = {
        'decisive_disclosure_rate': mean(disc),
        'decisive_credit_rate': mean(credit),
        'pooling_completeness': mean(complete),
        'disclosure_precision': dprec, 'disclosure_recall': drec,
        'disclosure_f1': safe(lambda: 2 * dprec * drec / (dprec + drec)),
        'over_disclosure_rate': mean(over),
        'elicited_fraction': mean([x for x in elicited if x == x]),
        'silent_holder_rate': mean(silent),
        'hallucinated_credit_rate': mean(halluc),
        'citation_containment': mean(contain),
        'premature_closure_rate': mean(premature),
        'disclosure_latency_norm': mean(latency),
        'survival_auc': mean([x for x in survival if x == x]),
    }

    # act-conditional yield: P(new decisive fact | act)
    yield_num, yield_den = collections.Counter(), collections.Counter()
    for r in R:
        acts = r.get('act_history') or []
        for a in acts:
            yield_den[a] += 1
        rt = r.get('reveal_turn') or {}
        dset = set(decisive_of(r))
        for f, t in rt.items():
            if f in dset and 0 <= t - 1 < len(acts):
                yield_num[acts[t - 1]] += 1
    res['B_pooling']['act_conditional_yield'] = {
        a: (yield_num[a] / yield_den[a]) for a in yield_den}

    # contributor balance
    res['B_pooling']['contributor_entropy'] = safe(lambda: entropy(contributors))
    res['B_pooling']['contributor_gini'] = safe(lambda: gini(contributors))

    # ================================================== C. EFFICIENCY AND COST
    turns = [r.get('turns') for r in R if r.get('turns')]
    caps = [r.get('max_turn') or 1 for r in R]
    succ_turns = [r.get('turns') for r, j in zip(R, joint) if j and r.get('turns')]
    calls = [r.get('n_calls') for r in R if r.get('n_calls')]
    pchars = [r.get('prompt_chars') for r in R if r.get('prompt_chars')]
    res['C_efficiency'] = {
        'avg_T': mean(turns),
        'avg_T_norm_by_cap': mean([t / max(1, c) for t, c in zip(turns, caps)]),
        'avg_T_given_success': mean(succ_turns),
        'timeout_rate': mean([1.0 if r.get('done') == -1 else 0.0 for r in R]),
        'calls_per_episode': mean(calls),
        'calls_by_role': dict(sum((collections.Counter(r.get('calls_by_role') or {})
                                   for r in R), collections.Counter())),
        'prompt_chars_per_episode': mean(pchars),
        'approx_prompt_tokens_per_episode': safe(lambda: mean(pchars) / 4.0),
        'joint_success_per_1k_prompt_tokens': safe(
            lambda: mean(joint) / (mean(pchars) / 4000.0)),
    }

    # ================================================== D. FORMAT ROBUSTNESS
    parsed = [1.0 if (r.get('settlement') or {}) else 0.0 for r in R]
    keys_seen = collections.Counter()
    typemis = []
    for r in R:
        st = r.get('settlement') or {}
        for k in st:
            keys_seen[k] += 1
        dec = st.get('decisions') or {}
        if isinstance(dec, dict) and dec:
            typemis.append(mean([1.0 if isinstance(v, str) and
                                 re.fullmatch(r'-?\d+(\.\d+)?', v.strip() or 'x') else 0.0
                                 for v in dec.values()]))
    strict = mean([sc(r).get('cbar') for r in R])
    norm = mean([(r.get('score_norm') or {}).get('cbar') for r in R])
    res['D_format'] = {
        'json_parse_rate': mean(parsed),
        'settlement_keys_frequency': dict(keys_seen),
        'numeric_as_string_rate': mean(typemis),
        'strict_pass_rate': strict,
        'normalised_pass_rate': norm,
        'normalisation_sensitivity': safe(lambda: norm - strict),
    }

    # ================================================== G. POLICY DIAGNOSTICS
    acts = collections.Counter(a for r in R for a in (r.get('act_history') or []))
    bigram = collections.Counter()
    for r in R:
        h = r.get('act_history') or []
        for a, b in zip(h, h[1:]):
            bigram['%s->%s' % (a, b)] += 1
    rewards = [r.get('reward') for r in R if isinstance(r.get('reward'), (int, float))]
    res['G_policy'] = {
        'act_distribution': dict(acts),
        'act_entropy_bits': safe(lambda: entropy(acts)),
        'act_bigrams_top': dict(bigram.most_common(10)),
        'return_mean': mean(rewards),
        'return_variance': safe(lambda: (sum((x - mean(rewards)) ** 2 for x in rewards)
                                         / max(1, len(rewards) - 1))),
    }

    # ================================================== H. BREAKDOWNS
    def by(keyfn, name):
        g = collections.defaultdict(list)
        for r, j in zip(R, joint):
            g[keyfn(r)].append(j)
        return {str(k): {'n': len(v), 'joint_success': mean(v)}
                for k, v in sorted(g.items(), key=lambda x: str(x[0]))}
    res['H_breakdown'] = {
        'by_domain': by(lambda r: r.get('domain'), 'domain'),
        'by_num_agents': by(lambda r: r.get('num_agents'), 'agents'),
        'by_scenario_type': by(lambda r: r.get('scenario_type'), 'type'),
        'by_decisive_fact_count': by(
            lambda r: len(decisive_of(r)), 'nfacts'),
        'by_turn_cap_bucket': by(
            lambda r: 'short' if (r.get('max_turn') or 0) <= 12 else
                      ('medium' if (r.get('max_turn') or 0) <= 24 else 'long'), 'cap'),
    }

    # ================================================== I. BASELINES / SIGNIFICANCE
    # paired against the per-scenario shared-context floor recorded with each episode
    fj = [1.0 if floor_val(r, 'joint') else 0.0 for r in R]
    wins = sum(1 for j, f in zip(joint, fj) if j > f)
    losses = sum(1 for j, f in zip(joint, fj) if j < f)
    ties = len(R) - wins - losses
    chi2 = ((abs(wins - losses) - 1) ** 2 / (wins + losses)) if (wins + losses) else 0.0
    res['I_baselines'] = {
        'floor_joint': floor_joint, 'floor_dca': floor_dca,
        'paired_vs_floor': {'win': wins, 'loss': losses, 'tie': ties,
                            'mcnemar_chi2': chi2,
                            'p': math.erfc(math.sqrt(chi2 / 2)) if chi2 else 1.0},
        'absolute_gain_vs_floor': safe(lambda: mean(joint) - floor_joint),
        'relative_gain_vs_floor': safe(
            lambda: (mean(joint) - floor_joint) / floor_joint if floor_joint
            else float('nan')),
    }
    res['omitted'] = {
        'F_language_quality': 'needs reference justifications and/or a judge model',
        'J_distributional': 'needs a second distribution (other roles/models) to compare',
        'E_numeric': 'needs gold numeric settlements (Phase 1 validation not yet done)',
    }

    # ------------------------------------------------------------- printing
    A, B, C = res['A_task_outcome'], res['B_pooling'], res['C_efficiency']
    print('=' * 74)
    print('HEADLINE (Table 2)')
    print('=' * 74)
    for k, v in (('Joint Success Rate', A['joint_success_rate']),
                 ('DCA', A['DCA']),
                 ('NDG', A['NDG']),
                 ('Decisive Credit Rate', B['decisive_credit_rate']),
                 ('Avg T / turn cap', C['avg_T_norm_by_cap']),
                 ('Calls per episode', C['calls_per_episode'])):
        print('  %-26s %s' % (k, pct(v)))

    print('\n' + '=' * 74)
    print('MAIN TABLE')
    print('=' * 74)
    for k, v in (('Settlement Success Rate', A['settlement_success_rate']),
                 ('Provenance Success Rate', A['provenance_success_rate']),
                 ('C_micro', A['C_micro']), ('C_macro', A['C_macro']),
                 ('P_micro', A['P_micro']), ('P_macro', A['P_macro']),
                 ('Non-decisive accuracy', A['non_decisive_accuracy']),
                 ('Shared-context floor (DCA)', A['floor_dca']),
                 ('Shared-context floor (joint)', A['floor_joint']),
                 ('Decisive Disclosure Rate', B['decisive_disclosure_rate']),
                 ('Pooling Completeness', B['pooling_completeness']),
                 ('Disclosure F1', B['disclosure_f1']),
                 ('Elicited fraction', B['elicited_fraction']),
                 ('Premature closure rate', B['premature_closure_rate']),
                 ('Hallucinated credit rate', B['hallucinated_credit_rate']),
                 ('JSON parse rate', res['D_format']['json_parse_rate']),
                 ('Avg T', C['avg_T']), ('Timeout rate', C['timeout_rate'])):
        print('  %-26s %s' % (k, pct(v)))

    print('\n' + '=' * 74)
    print('APPENDIX / DIAGNOSTIC')
    print('=' * 74)
    print('  act distribution      %s' % res['G_policy']['act_distribution'])
    print('  act entropy (bits)    %s' % pct(res['G_policy']['act_entropy_bits']))
    print('  act bigrams (top)     %s' % res['G_policy']['act_bigrams_top'])
    print('  act-conditional yield %s' % {k: round(v, 3)
                                          for k, v in B['act_conditional_yield'].items()})
    print('  return mean/var       %s / %s' % (pct(res['G_policy']['return_mean']),
                                               pct(res['G_policy']['return_variance'])))
    print('  normalisation sens.   %s' % pct(res['D_format']['normalisation_sensitivity']))
    print('  unreachable checks    %s' % (unreachable[:12] or 'none'))
    print('  vs floor  win/loss/tie %s  p=%.4g'
          % ((res['I_baselines']['paired_vs_floor']['win'],
              res['I_baselines']['paired_vs_floor']['loss'],
              res['I_baselines']['paired_vs_floor']['tie']),
             res['I_baselines']['paired_vs_floor']['p']))
    for name, d in res['H_breakdown'].items():
        print('  %-22s %s' % (name, {k: '%.3f (n=%d)' % (v['joint_success'], v['n'])
                                     for k, v in d.items()}))
    print('\n  OMITTED: %s' % ', '.join(res['omitted']))

    # learning curve across epochs, if more than one
    if len(epochs) > 1:
        print('\n  learning curve (Joint Success by epoch)')
        for e in epochs:
            rr = by_epoch[e]
            if rr:
                js = mean([1.0 if (allpass((sc(r).get('content') or {}))
                                   and allpass((sc(r).get('provenance') or {}))) else 0.0
                           for r in rr])
                print('    epoch %-3d n=%-4d joint success %s' % (e, len(rr), pct(js)))
                res.setdefault('learning_curve', {})[str(e)] = js

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    json.dump(res, open(cli.out, 'w', encoding='utf-8'), indent=1, default=str)
    print('\nwrote %s' % cli.out)


if __name__ == '__main__':
    main()
