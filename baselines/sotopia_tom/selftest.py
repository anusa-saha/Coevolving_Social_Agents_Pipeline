"""No-GPU checks. Run before spending inference time.

    python selftest.py
"""
import json
import sys
import traceback

import compat
import config
import data_csa
import detectors_tom as D
import metrics_tom as M
import paths
import prompts_tom as P
import verifier_tom as V

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
    assert got == {'train': 99, 'valid': 9, 'test': 42}, got
    assert not data_csa.check_invariants(data_csa.load_raw())


def t_split_matches_published():
    for name in ('train', 'valid', 'test'):
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


def t_every_strategy_builds_and_is_filtered():
    """Every arm, every scenario: no private fact and no oracle field may reach the chair."""
    conv = [{'role': 'Meeting', 'content': 'x'}]
    n = 0
    for case in data_csa.load_raw():
        for arm in P.STRATEGIES:
            for settling in (False, True):
                msgs = P.chair_messages(case, conv, strategy=arm, settling=settling,
                                        tom_note='some analysis')
                blob = ' '.join(m['content'] for m in msgs)
                for fid, fact in case['private_facts'].items():
                    assert fact['text'] not in blob, '%s leaked (%s)' % (fid, arm)
                for f in P.ORACLE_ONLY:
                    assert f not in blob, '%s leaked (%s)' % (f, arm)
        n += 1
    assert n == 150, n


def t_strategies_actually_differ():
    """A prompting study is worthless if two arms send the same prompt."""
    case = data_csa.load('test')[0]
    conv = [{'role': 'Meeting', 'content': 'x'}]
    seen = {}
    for arm in P.STRATEGIES:
        blob = ' '.join(m['content'] for m in
                        P.chair_messages(case, conv, strategy=arm, tom_note='NOTE-XYZ'))
        for other, prev in seen.items():
            assert blob != prev, '%s and %s produce identical prompts' % (arm, other)
        seen[arm] = blob
    # the control must not mention drawing information out
    stripped = seen['stripped'].lower()
    for phrase in ('draw it out', 'draw them out', 'they hold information'):
        assert phrase not in stripped, 'the control still contains an elicitation cue'
    # the ToM arms must actually carry the injected note
    assert 'NOTE-XYZ' in seen['tom_coach'] and 'NOTE-XYZ' in seen['tom_belief']
    assert 'NOTE-XYZ' not in seen['basic']


def t_settling_turn_keeps_schema():
    case = data_csa.load('test')[0]
    for arm in P.STRATEGIES:
        body = P.chair_messages(case, [], strategy=arm, settling=True)[1]['content']
        assert 'settlement schema is' in body, arm
        assert 'JSON object only' in body, arm
    # the CoT arm must be told to drop THINKING, or the JSON is unparseable
    cot = P.chair_messages(case, [], strategy='cot', settling=True)[1]['content']
    assert 'Do not include a THINKING line' in cot


def t_split_thinking():
    assert P.split_thinking('THINKING: a\nb\nTURN: Ask Chen about the limit.') \
        == 'Ask Chen about the limit.'
    assert P.split_thinking('Ask Chen about the limit.') == 'Ask Chen about the limit.'
    assert P.split_thinking('THINKING: only reasoning\nnothing else') != ''


def t_advisors_see_only_own_facts():
    for case in data_csa.load_raw()[:40]:
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            blob = ' '.join(m['content'] for m in
                            P.advisor_messages(case, [], a['agent_id']))
            for fid, fact in case['private_facts'].items():
                if fact['owner'] != a['agent_id']:
                    assert fact['text'] not in blob, (a['agent_id'], fid)


def t_infomgmt():
    """The geometric mean must zero out on any failed dimension -- that is its purpose."""
    rec = {'score': {'disclosure_rate': 0.5}, 'reveal_elicited': {'PF1': True},
           'reveal_turn': {'PF1': 0}, 'max_turn': 4}
    m = M.infomgmt(rec)
    assert abs(m['DA'] - 0.5) < 1e-9 and abs(m['IA'] - 1.0) < 1e-9
    assert m['EFF'] == 1.0 and m['InfoMgmt3'] > 0
    # nothing elicited -> IA 0 -> composite 0
    rec2 = dict(rec, reveal_elicited={'PF1': False})
    assert M.infomgmt(rec2)['IA'] == 0.0
    assert M.infomgmt(rec2)['InfoMgmt3'] == 0.0
    # nothing disclosed at all -> everything 0
    rec3 = {'score': {'disclosure_rate': 0.0}, 'reveal_elicited': {}, 'reveal_turn': {},
            'max_turn': 4}
    assert M.infomgmt(rec3)['InfoMgmt3'] == 0.0
    # late disclosure scores lower than early
    late = dict(rec, reveal_turn={'PF1': 3})
    assert M.efficiency(late) < M.efficiency(rec)


def t_summarise_averages_episodes_not_dimensions():
    """Composite must be averaged over episodes; averaging the dimensions first would let
    a zero episode be rescued by the others."""
    good = {'score': {'disclosure_rate': 1.0}, 'reveal_elicited': {'PF1': True},
            'reveal_turn': {'PF1': 0}, 'max_turn': 4, 'done': 1}
    dead = {'score': {'disclosure_rate': 0.0}, 'reveal_elicited': {}, 'reveal_turn': {},
            'max_turn': 4, 'done': -1}
    s = M.summarise([good, dead])
    assert abs(s['InfoMgmt3'] - 0.5) < 1e-9, s['InfoMgmt3']
    assert s['CPV'] is None


def t_paired():
    a = [{'uid': 'x', 'score': {'disclosure_rate': 1.0}, 'reveal_elicited': {'p': True},
          'reveal_turn': {'p': 0}, 'max_turn': 4}]
    b = [{'uid': 'x', 'score': {'disclosure_rate': 0.0}, 'reveal_elicited': {},
          'reveal_turn': {}, 'max_turn': 4}]
    assert M.paired(a, b, 'DA')['win'] == 1
    assert M.paired(b, a, 'DA')['loss'] == 1


if __name__ == '__main__':
    print('sotopia_tom selftest\n')
    blocking = compat.report()
    print('\nraw scenarios: %s' % paths.find_raw())
    print('model        : %s\n' % config.Defaults.model)

    for name, fn in [
        ('split is 99/9/42 and invariants hold', t_split),
        ('split matches the published one exactly', t_split_matches_published),
        ('verifier matches published scores', t_verifier_matches_published),
        ('all 5 arms build, chair prompt filtered (150 cases)',
         t_every_strategy_builds_and_is_filtered),
        ('the 5 arms send genuinely different prompts', t_strategies_actually_differ),
        ('settling turn keeps the schema in every arm', t_settling_turn_keeps_schema),
        ('CoT THINKING/TURN split', t_split_thinking),
        ('advisors see only their own facts', t_advisors_see_only_own_facts),
        ('InfoMgmt3 geometric mean', t_infomgmt),
        ('composite averages episodes, not dimensions',
         t_summarise_averages_episodes_not_dimensions),
        ('paired sign test', t_paired),
    ]:
        check(name, fn)

    print()
    if FAILS:
        print('%d FAILED' % len(FAILS))
        sys.exit(1)
    if blocking:
        print('logic checks passed, but the environment BLOCKS inference (see above).')
        sys.exit(2)
    print('all passed')
