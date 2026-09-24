"""Checks that need no GPU. Run before spending time on the box.

    python selftest.py

The two that matter most:
  * the from-scratch split reproduces the published 99/9/42 exactly, so this arm is
    comparable to the other baselines;
  * the from-scratch verifier reproduces published cbar/pbar/disclosure on 297 real
    episodes, so "we reimplemented it" is a claim with evidence behind it.

Both compare against files from another baseline as DATA. Nothing in the training path
reads them, and both checks skip cleanly if those files are absent -- or, for the split,
if the only split files found were exported under a different configuration.
"""
import json
import sys
import traceback

import attribution
import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import compat as compat
from csa_core import data_csa as data_csa
from csa_core import detectors as D
import prompts_sr as P
from csa_core import verifier as V

FAILS = []


def check(name, fn):
    try:
        fn()
        print('  ok    %s' % name)
    except Exception as e:                           # noqa: BLE001
        FAILS.append((name, e))
        print('  FAIL  %s -- %s' % (name, e))
        traceback.print_exc(limit=2)


def t_split():
    d = data_csa.load()
    got = {k: len(v) for k, v in d.items()}
    if paths.is_published_config():
        assert got == {'train': 99, 'valid': 9, 'test': 42}, got
    else:
        # A reconfigured benchmark has no published shape to match, and the explicit
        # split files have their own (documented) shape, so delegate to the loader's
        # own audit: it checks whatever must hold for the configuration in play.
        print('        ' + data_csa.audit_splits())
    bad = data_csa.check_invariants(data_csa.load_raw())
    assert not bad, bad[:3]


def t_split_matches_published():
    if not paths.is_published_config():
        print('        (skipped: benchmark reconfigured to %d domains x %d; '
              'the published split describes a different dataset)'
              % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN))
        return
    refs = {}
    for name, want in zip(('train', 'valid', 'test'), paths.PUBLISHED_SPLIT):
        # An archived copy under its own name wins. ppdpp/data/csa-<split>.txt is whatever
        # export_csa.py last wrote, so it is only the published split if the size agrees.
        ref = (paths.find_reference('data/csa-published-%s.txt' % name)
               or paths.find_reference('data/csa-%s.txt' % name))
        if not ref:
            print('        (skipped: no published split to compare)')
            return
        n = sum(1 for l in open(ref, encoding='utf-8') if l.strip())
        if n != want:
            print('        (skipped: %s holds %d scenarios, not the published %d, so it was\n'
                  '         exported under another configuration. Archive the 99/9/42 split as\n'
                  '         ppdpp/data/csa-published-<split>.txt under CSA_ARTIFACTS_DIR to '
                  're-enable this check)' % (ref, n, want))
            return
        refs[name] = ref
    for name, ref in refs.items():
        theirs = [eval(l) for l in open(ref, encoding='utf-8') if l.strip()]
        mine = [r['uid'] for r in data_csa.load(name)]
        assert mine == [r['uid'] for r in theirs], \
            '%s split differs from the published one' % name


def t_verifier_matches_published():
    ref = paths.find_reference('conversations/conversations-train-labelled.jsonl')
    if not ref:
        print('        (skipped: no published episodes to compare)')
        return
    rows = [json.loads(l) for l in open(ref, encoding='utf-8')]
    cases = data_csa.case_index()
    # The archived episodes were produced under whatever dataset was configured then.
    # When that is not the dataset in play now, nearly every uid misses and the check
    # would fail for the wrong reason, so recognise it and skip instead.
    matched = sum(1 for r in rows if r['uid'] in cases)
    if matched <= 200:
        print('        (skipped: only %d of %d archived episodes name a scenario in this '
              'configuration;\n         those records describe a different dataset)'
              % (matched, len(rows)))
        return
    n = 0
    for r in rows:
        case = cases.get(r['uid'])
        if not case:
            continue
        # published values were computed without provenance resolution
        s = V.score(case, r.get('settlement') or {}, set(r.get('revealed') or []),
                    resolve=False)
        for mine, theirs in (('cbar', 'cbar'), ('pbar', 'pbar'),
                             ('disclosure_rate', 'disclosure')):
            assert abs(s[mine] - r[theirs]) < 1e-9, \
                '%s %s: %s vs published %s' % (r['uid'], mine, s[mine], r[theirs])
        n += 1
    assert n > 200, 'only compared %d episodes' % n
    print('        %d episodes agree exactly' % n)


def t_chair_prompt_is_filtered():
    """A chair that can see a private fact would score perfectly while measuring nothing."""
    n = 0
    for case in data_csa.load_raw():
        msgs = P.chair_messages(case, [{'role': 'Meeting', 'content': 'x'}])
        blob = ' '.join(m['content'] for m in msgs)
        for fid, fact in case['private_facts'].items():
            assert fact['text'] not in blob, '%s leaked to the chair in %s' % (fid, case['uid'])
        for f in P.ORACLE_ONLY:
            assert f not in blob, 'oracle field %s leaked in %s' % (f, case['uid'])
        n += 1
    assert n == len(data_csa.load_raw()), n


def t_advisor_sees_only_own_facts():
    for case in data_csa.load_raw()[:40]:
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            blob = ' '.join(m['content'] for m in
                            P.advisor_messages(case, [], a['agent_id']))
            for fid, fact in case['private_facts'].items():
                if fact['owner'] == a['agent_id']:
                    continue
                assert fact['text'] not in blob, \
                    '%s saw %s belonging to %s' % (a['agent_id'], fid, fact['owner'])


def t_verifier_semantics():
    case = data_csa.load('train')[0]
    empty = V.floor_score(case)
    assert empty['dca'] == 0.0 and empty['cbar'] == 0.0
    assert empty['schema_valid'] is False
    # a missing field must FAIL its check, not raise
    ctx = V.build_context({'decisions': {}}, set(), norm=True)
    assert V.eval_checks({'X': "decisions['nope'] == 'a'"}, ctx) == {'X': False}
    assert V.eval_checks({'X': "'PF1' in credited_facts"}, ctx) == {'X': False}
    # canonicalisation folds formatting, not content
    assert V.canonical('3:00 PM') == V.canonical('3 pm')
    assert V.canonical('1,800 kilograms') == V.canonical('1800 kilograms')
    assert V.canonical('Lane 2') != V.canonical('Lane 3')


def t_eliciting_detector():
    """No planner here, so eliciting-ness is read off the text. Report agreement with the
    published act annotations rather than assuming it."""
    ref = paths.find_reference('data_sft/csa-train.txt')
    if not ref:
        print('        (skipped: no annotated turns to compare)')
        return
    cases = data_csa.case_index()
    tp = fp = tn = fn = 0
    for line in open(ref, encoding='utf-8'):
        conv = json.loads(line)
        case = cases.get(conv['uid'])
        if not case:
            continue
        for t in conv['dialog']:
            if t.get('speaker') != 'sys' or not t.get('strategy'):
                continue
            gold = t['strategy'] in ('ask', 'followup')
            pred = D.is_eliciting(t['content'], case['agents'], case['decision_maker'])
            tp += gold and pred; fp += (not gold) and pred
            tn += (not gold) and (not pred); fn += gold and (not pred)
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print('        vs %d annotated turns: acc %.3f  precision %.3f  recall %.3f'
          % (n, acc, prec, rec))
    assert acc > 0.6, 'is_eliciting agrees with the annotations only %.2f of the time' % acc


def t_attribution():
    ref = paths.find_reference('conversations/conversations-train-labelled.jsonl')
    if not ref:
        print('        (skipped)')
        return
    rows = [json.loads(l) for l in open(ref, encoding='utf-8')]
    cases = data_csa.case_index()
    raws = []
    for r in rows:
        case = cases.get(r['uid'])
        if case:
            raw, _d = attribution.attribute(case, r['dialog'], r.get('settlement'))
            raws.append(raw)
    assert raws
    for raw in raws:
        for d in attribution.DIMS:
            assert all(0.0 <= x <= 1.0 + 1e-9 for x in raw[d]), \
                '%s attribution outside [0,1] -- flips summed instead of unioned?' % d
    norm = attribution.Normaliser().fit(raws)
    vals = [x for raw in raws for x in norm.apply(raw)]
    desc = attribution.describe(vals)
    print('        labels over %d turns: %s' % (desc['n'], desc))
    assert desc['frac_non_zero'] > 0.15, 'labels are too sparse to regress on'
    assert desc['max'] <= 1.0 + 1e-9


def t_normaliser_roundtrip():
    raws = [{d: [0.0, 0.5, 1.0] for d in attribution.DIMS} for _ in range(3)]
    n = attribution.Normaliser().fit(raws)
    st = json.loads(json.dumps(n.state()))
    n2 = attribution.Normaliser.from_state(st)
    assert n.apply(raws[0]) == n2.apply(raws[0])
    # within-episode normalisation cancels a constant factor; that is the point of the
    # docstring warning, and it should stay reproducible
    a = attribution.Normaliser(within_episode=True)
    x = {d: [0.0, 0.25, 0.5] for d in attribution.DIMS}
    y = {d: [0.0, 0.5, 1.0] for d in attribution.DIMS}
    assert a.apply(x) == a.apply(y), 'within-episode norm should be scale-invariant'


def t_headline_metrics():
    """eval.py's headline metrics must agree with the verifier they are read from, and the
    paired bootstrap must behave at its two edges."""
    from csa_core import headline as H
    case = next(c for c in data_csa.load('test') if c.get('provenance_checks'))
    decisive = [d['fact_id'] for d in case['decisive_facts']]
    holder = case['private_facts'][decisive[0]]['owner']
    rec = {'uid': case['uid'], 'settlement': {'decisions': {}}, 'revealed': decisive,
           'addressed': [holder], 'settle_turn': 2, 'settled_by': 'chair',
           'leaks': [{'fact': decisive[0], 'by': case['decision_maker'], 'turn': 0}],
           'dialog': [{'role': 'Meeting', 'content': 'x'},
                      {'role': H.chair_name(case), 'content': 'hello'}],
           'score': V.score(case, {'decisions': {}}, set(decisive))}
    m = H.episode_metrics(case, rec)
    s = rec['score']
    assert m['content_total'] == len(case['content_checks'])
    assert m['prov_total'] == len(case['provenance_checks'])
    assert m['content_passed'] == sum(bool(v) for v in s['content'].values())
    assert m['checks_passed'] == m['content_passed'] + m['prov_passed']
    assert m['success'] == int(m['checks_passed'] == m['checks_total'])
    assert m['decisive_revealed'] == m['decisive_total'] == len(set(decisive))
    assert m['settled'] == 1 and m['turns_to_settle'] == 3 and m['turns_used'] == 1, m
    assert m['address_precision'] == 1.0 and m['chair_leaks'] == 1, m
    m0 = H.episode_metrics(case, dict(rec, settlement={}, settle_turn=None))
    assert m0['settled'] == 0 and m0['turns_to_settle'] is None and m0['settled_by'] == ''
    b = H.summary([m, m0])['bottleneck']
    assert sum(b[k]['n'] for k in b) == 2, 'the bottleneck must partition every episode'
    obs, lo, hi, p = H.paired_bootstrap([0.2, 0.4, 0.6], [0.2, 0.4, 0.6], 200)
    assert obs == lo == hi == 0.0 and p == 1.0
    obs, lo, hi, p = H.paired_bootstrap([0.1] * 30, [0.3] * 30, 200)
    assert abs(obs - 0.2) < 1e-9 and lo > 0 and p < 0.05


if __name__ == '__main__':
    print('sotopia_rl selftest\n')
    blocking = compat.report()
    print('\nscenarios: %s\n' % data_csa.source())

    for name, fn in [
        ('split is complete and invariants hold', t_split),
        ('split matches the published one exactly', t_split_matches_published),
        ('verifier matches published scores', t_verifier_matches_published),
        ('chair prompt is view-filtered (every case)', t_chair_prompt_is_filtered),
        ('advisors see only their own facts', t_advisor_sees_only_own_facts),
        ('verifier semantics', t_verifier_semantics),
        ('is_eliciting vs annotated acts', t_eliciting_detector),
        ('attribution stays in [0,1] and is dense enough', t_attribution),
        ('normaliser round-trips', t_normaliser_roundtrip),
        ("headline metrics agree with the verifier", t_headline_metrics),
    ]:
        check(name, fn)

    print()
    if FAILS:
        print('%d FAILED' % len(FAILS))
        sys.exit(1)
    if blocking:
        print('logic checks passed, but the environment BLOCKS training (see above).')
        sys.exit(2)
    print('all passed')
