"""Fast checks that need no GPU. Run before burning time on the box.

    python selftest.py

Covers the parts that fail silently: split integrity, view filtering of the strategist
prompt, strategy parsing, the process reward, and (where torch is present) that the
copied disclosure detector still matches ppdpp_csa.
"""
import sys
import traceback

import compat
import config
import prm
import prompt_epo as pe

FAILS = []


def check(name, fn):
    try:
        fn()
        print('  ok    %s' % name)
    except Exception as e:                           # noqa: BLE001
        FAILS.append((name, e))
        print('  FAIL  %s -- %s' % (name, e))
        traceback.print_exc(limit=2)


# ------------------------------------------------------------------ splits
def t_splits():
    d = config.load_csa()
    got = {k: len(v) for k, v in d.items()}
    assert got == {'train': 99, 'valid': 9, 'test': 42}, got
    uids = [r['uid'] for rows in d.values() for r in rows]
    assert len(uids) == len(set(uids)), 'uid collision across splits'
    for name, rows in d.items():
        for r in rows:
            dm = r['decision_maker']
            for fid, fact in r['private_facts'].items():
                assert fact['owner'] != dm, '%s: %s owned by the chair' % (r['uid'], fid)
                assert fid not in r['views'][dm], '%s: %s visible to chair' % (r['uid'], fid)


# ------------------------------------------------------------------ prompts
def t_strategist_prompt_is_filtered():
    """The assertion that matters most: a strategist that can see a private fact would
    score perfectly while measuring nothing."""
    n = 0
    for rows in config.load_csa().values():
        for case in rows:
            conv = [{'role': 'Meeting', 'content': 'x'}]
            msgs = pe.strategist_messages(case, conv)      # asserts internally
            blob = ' '.join(m['content'] for m in msgs)
            for fid, fact in case['private_facts'].items():
                assert fact['text'] not in blob, '%s leaked %s' % (case['uid'], fid)
            for f in pe.CSA_ORACLE_ONLY:
                assert f not in blob, '%s leaked oracle field %s' % (case['uid'], f)
            n += 1
    assert n == 150, n


def t_chair_injection():
    case = config.load_csa('train')[0]
    conv = [{'role': 'Meeting', 'content': 'x'}]
    m = pe.chair_messages(case, conv, 'ask', 'get Morgan to state the limit')
    assert 'get Morgan to state the limit' in m[1]['content']
    assert pe.CSAAct['ask'] in m[1]['content']
    # on a settling turn the schema must stay last, or the JSON instruction is buried
    m = pe.chair_messages(case, conv, 'decide', 'record it now')
    body = m[1]['content']
    assert 'record it now' in body
    assert body.index('record it now') < body.index('The settlement schema is:')


def t_parse():
    cases = [
        ('followup: press Patel for the exact clearance', 'followup', True),
        ('**share**: relay what Morgan said', 'share', True),
        ('DECIDE: Record it.', 'decide', True),
        ('  ask - SFC Morgan, limits?', 'ask', True),
        ('I think we should ask Morgan', 'ask', False),
        ('', 'ask', False),
    ]
    for text, want_tau, want_ok in cases:
        tau, sigma, ok = pe.parse_strategy(text)
        assert tau == want_tau, (text, tau)
        assert ok == want_ok, (text, ok)
        assert not sigma.startswith('*'), (text, sigma)
    # a sigma that legitimately begins with an act word keeps it
    assert pe.clean_sigma('ask Patel whether the tilt-bed fits').startswith('ask Patel')
    # a duplicated tag with its separator is removed
    assert pe.clean_sigma('followup: press for the number') == 'press for the number'


# ------------------------------------------------------------------ reward
def _trace(case, elicited, dca, leaks=(), settle=3, n=4):
    dec = case['decisive_facts']
    return prm.Trace(n_turns=n,
                     reveal_turn={d['fact_id']: i + 1 for i, d in enumerate(dec)},
                     reveal_elicited={d['fact_id']: elicited for d in dec},
                     decisive=dec, settle_turn=settle, dca=dca, schema_valid=True,
                     leaks=list(leaks), acts=['ask'] * n, conversation=[], case=case)


def t_prm():
    case = next(c for c in config.load_csa('train') if len(c['decisive_facts']) >= 2)
    binary = prm.VerifierPRM(mode='binary', done_tau=0.6)
    graded = prm.VerifierPRM(mode='graded', done_tau=0.6)

    r = binary(_trace(case, True, 0.75))
    assert sum(r) > 0 and len(r) == 4
    # volunteered, not elicited -> the chair earned nothing for the disclosure
    r_vol = binary(_trace(case, False, 0.75))
    assert sum(r_vol) < sum(r), (r, r_vol)
    # a leak suppresses the settle term entirely
    assert binary(_trace(case, True, 0.9, leaks=[{'fact': 'PF1'}]))[3] == 0.0
    # sub-threshold dca suppresses it too, in binary mode
    assert binary(_trace(case, True, 0.2))[3] == 0.0
    # graded is continuous in dca rather than thresholded
    assert graded(_trace(case, True, 0.2))[3] > 0.0
    # rewards land on the recorded turn, never out of range
    for mode in (binary, graded):
        assert all(0.0 <= x for x in mode(_trace(case, True, 0.5)))


def t_kappa():
    assert abs(prm.agreement([[1, 0, 1]], [[1, 0, 1]]) - 1.0) < 1e-9
    assert prm.agreement([[1, 0, 1, 0]], [[0, 1, 0, 1]]) < 0
    assert prm.agreement([], []) != prm.agreement([], [])   # nan


# ------------------------------------------------------------------ sft data
def t_sft_targets_if_present():
    import os
    path = os.path.join(config.DATA, 'strategies-train.jsonl')
    if not os.path.exists(path):
        print('        (skipped: run make_strategies.py first)')
        return
    import json
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    assert rows, 'empty'
    cases = config.case_index()
    for r in rows:
        assert r['tau'] in pe.ACTS, r['tau']
        assert r['uid'] in cases, r['uid']
        tau, sigma, ok = pe.parse_strategy(r['target'])
        assert ok and tau == r['tau'], (r['target'], tau, r['tau'])
        assert sigma, 'empty sigma in %s' % r['uid']
    print('        %d targets, all parseable' % len(rows))


def t_detector_drift():
    from detectors import assert_identical_to_ppdpp
    assert_identical_to_ppdpp()          # returns None if torch is unavailable


if __name__ == '__main__':
    print('epo selftest\n')
    blocking = compat.report()
    print('\nppdpp_csa: %s\n' % config.PPDPP)

    for name, fn in [
        ('splits 99/9/42, chair holds no private fact', t_splits),
        ('strategist prompt is view-filtered (all 150 cases)', t_strategist_prompt_is_filtered),
        ('chair injection keeps schema last', t_chair_injection),
        ('strategy parsing', t_parse),
        ('verifier process reward', t_prm),
        ("Cohen's kappa", t_kappa),
        ('manufactured SFT targets', t_sft_targets_if_present),
        ('disclosure detector matches ppdpp_csa', t_detector_drift),
    ]:
        check(name, fn)

    print()
    if FAILS:
        print('%d FAILED' % len(FAILS))
        sys.exit(1)
    if blocking:
        print('logic checks passed, but the environment BLOCKS training (see above). '
              'Stage 1 can still run with --fallback_only.')
        sys.exit(2)
    print('all passed')
