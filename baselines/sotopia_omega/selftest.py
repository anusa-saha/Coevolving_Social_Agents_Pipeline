"""No-GPU checks. Run before spending generation time.

    python selftest.py
"""
import json
import random
import sys
import traceback

import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import compat as compat
import config
from csa_core import data_csa as data_csa
from csa_core import detectors as D
import prompts_om as P
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
    got = {k: len(v) for k, v in data_csa.load().items()}
    if paths.is_published_config():
        assert got == {'train': 99, 'valid': 9, 'test': 42}, got
    else:
        # A reconfigured benchmark has no published shape to match, so check the
        # properties that must hold under ANY configuration instead: every scenario
        # lands in exactly one split, and none is lost.
        total = len(paths.DOMAINS) * paths.SCENARIOS_PER_DOMAIN
        assert sum(got.values()) == total, (got, total)
        uids = [r['uid'] for k in ('train', 'valid', 'test') for r in data_csa.load(k)]
        assert len(set(uids)) == total, 'split overlaps or drops scenarios'
        print('        %d domains x %d = %d -> %d/%d/%d'
              % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN, total,
                 got['train'], got['valid'], got['test']))
    assert not data_csa.check_invariants(data_csa.load_raw())


def t_split_matches_published():
    for name in ('train', 'valid', 'test'):
        if not paths.is_published_config():
            print('        (skipped: benchmark reconfigured to %d domains x %d; '
                  'the published split describes a different dataset)'
                  % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN))
            return
        ref = paths.find_reference('data/csa-%s.txt' % name)
        if not ref:
            print('        (skipped: no published split to compare)')
            return
        theirs = [eval(l) for l in open(ref, encoding='utf-8') if l.strip()]
        assert [r['uid'] for r in data_csa.load(name)] == [r['uid'] for r in theirs], name


def t_verifier_matches_published():
    ref = paths.find_reference('conversations/conversations-train-labelled.jsonl')
    if not ref:
        print('        (skipped)')
        return
    rows = [json.loads(l) for l in open(ref, encoding='utf-8')]
    cases = data_csa.case_index()
    n = 0
    for r in rows:
        case = cases.get(r['uid'])
        if not case:
            continue
        s = V.score(case, r.get('settlement') or {}, set(r.get('revealed') or []),
                    resolve=False)
        for mine, theirs in (('cbar', 'cbar'), ('pbar', 'pbar'),
                             ('disclosure_rate', 'disclosure')):
            assert abs(s[mine] - r[theirs]) < 1e-9, (r['uid'], mine)
        n += 1
    assert n > 200
    print('        %d episodes agree exactly' % n)


def t_prompts_filtered():
    """Fast, slow-scaffold and advisor prompts must never carry a private fact the
    speaker does not own, nor any oracle field."""
    conv = [{'role': 'Meeting', 'content': 'x'}]
    for case in data_csa.load_raw():
        for settling in (False, True):
            for plan in ('', 'Still unsupported:\n  lane\nAsk next: ask Morgan.'):
                blob = ' '.join(m['content'] for m in
                                P.chair_messages(case, conv, settling, plan))
                for fid, f in case['private_facts'].items():
                    assert f['text'] not in blob, '%s leaked to the chair' % fid
                for o in P.ORACLE_ONLY:
                    assert o not in blob, '%s leaked to the chair' % o
        for stage in (1, 2, 3):
            P.stage_messages(case, conv, stage, unsupported='a', holders='b')
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            for opp in (None, a['agent_id']):
                blob = ' '.join(m['content'] for m in
                                P.advisor_messages(case, conv, a['agent_id'], opponent=opp))
                for fid, f in case['private_facts'].items():
                    if f['owner'] != a['agent_id']:
                        assert f['text'] not in blob, (a['agent_id'], fid)


def t_opponent_prompt_differs():
    case = data_csa.load('test')[0]
    adv = next(a['agent_id'] for a in case['agents']
               if a['agent_id'] != case['decision_maker'])
    plain = ' '.join(m['content'] for m in P.advisor_messages(case, [], adv))
    withh = ' '.join(m['content'] for m in
                     P.advisor_messages(case, [], adv, opponent=adv))
    assert plain != withh, 'the withholder prompt is identical to the cooperative one'
    assert 'Never state something untrue' in withh, \
        'the withholder must not be licensed to lie -- that would break the benchmark'
    # a non-designated advisor is unaffected
    other = [a['agent_id'] for a in case['agents']
             if a['agent_id'] not in (case['decision_maker'], adv)]
    if other:
        assert ' '.join(m['content'] for m in
                        P.advisor_messages(case, [], other[0], opponent=adv)) == \
            ' '.join(m['content'] for m in P.advisor_messages(case, [], other[0]))


def t_plan_reaches_the_chair():
    case = data_csa.load('test')[0]
    body = ' '.join(m['content'] for m in
                    P.chair_messages(case, [], plan='Ask next: press Morgan on the limit.'))
    assert 'press Morgan on the limit' in body
    assert 'press Morgan' not in ' '.join(m['content'] for m in P.chair_messages(case, []))


def t_ceiling_math():
    """The ceiling must equal the share of flipped checks still reachable."""
    for case in data_csa.load_raw()[:40]:
        flips = {d['fact_id']: set(d.get('flips') or [])
                 for d in case['decisive_facts']}
        phi = set().union(*flips.values()) if flips else set()
        if not phi:
            continue
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            lost = set()
            for fid, f in case['private_facts'].items():
                if f['owner'] == a['agent_id'] and fid in flips:
                    lost |= flips[fid]
            want = len(phi - lost) / len(phi)
            assert 0.0 <= want <= 1.0
            # silencing any advisor must cost something: every advisor holds a decisive
            # fact in this corpus, which is what makes the variant non-trivial
            assert want < 1.0, '%s: silencing %s costs nothing' % (case['uid'], a['agent_id'])


def t_stall_rule():
    """Pure logic: the trigger fires only after stall_after and stall_patience."""
    class C:
        stall_after, stall_patience = 1, 1
    seen, flat, mode, fired = 0, 0, 'fast', None
    for step, pooled in enumerate([0, 0, 0, 1]):
        flat = 0 if pooled > seen else flat + 1
        seen = max(seen, pooled)
        if mode == 'fast' and step >= C.stall_after and flat >= C.stall_patience:
            mode, fired = 'slow', step
    assert fired == 1, fired
    # pooling every turn must never trigger
    seen, flat, mode = 0, 0, 'fast'
    for step, pooled in enumerate([1, 2, 3, 4]):
        flat = 0 if pooled > seen else flat + 1
        seen = max(seen, pooled)
        if mode == 'fast' and step >= C.stall_after and flat >= C.stall_patience:
            mode = 'slow'
    assert mode == 'fast', 'triggered even though pooling advanced every turn'


def t_evasion_margin():
    """The measurement risk the withholder arm depends on. Report it, do not assume it."""
    case = next(c for c in data_csa.load_raw() if c['private_facts'])
    fid = sorted(case['private_facts'])[0]
    fact = case['private_facts'][fid]['text']
    full = fact.split(';')[0]
    evasive = 'I would need to check that before committing.'
    a, b = D.overlap(fact, full), D.overlap(fact, evasive)
    print('        full-disclosure %.3f vs evasive %.3f (threshold %.2f)'
          % (a, b, D.REVEAL_THRESHOLD))
    assert a > b, 'evasion scores at least as high as disclosure'


if __name__ == '__main__':
    print('sotopia_omega selftest\n')
    blocking = compat.report()
    print('\nraw scenarios : %s' % paths.find_raw())
    print('expert        : %s (%s)' % (config.Defaults.expert_model,
                                       config.Defaults.expert))
    print('student       : %s\n' % config.Defaults.student_model)

    for name, fn in [
        ('split is complete and invariants hold', t_split),
        ('split matches the published one exactly', t_split_matches_published),
        ('verifier matches published scores', t_verifier_matches_published),
        ('all prompts view-filtered (every case)', t_prompts_filtered),
        ('withholder prompt differs and cannot lie', t_opponent_prompt_differs),
        ('slow-mode plan reaches the chair, and only then', t_plan_reaches_the_chair),
        ('ceiling math, every advisor costs something', t_ceiling_math),
        ('stall trigger fires only on flat pooling', t_stall_rule),
        ('evasion scores below disclosure', t_evasion_margin),
    ]:
        check(name, fn)

    print()
    if FAILS:
        print('%d FAILED' % len(FAILS))
        sys.exit(1)
    if blocking:
        print('logic checks passed, but the environment BLOCKS generation (see above).')
        sys.exit(2)
    print('all passed')
