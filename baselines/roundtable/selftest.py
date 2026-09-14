"""CPU checks for the round-table baseline. No model, no GPU, about a minute.

The arm has almost no machinery, so there is little to get wrong in the method -- but the
two things it DOES do are the two that fail silently: view filtering, and turning N
proposals into one settlement. Both are checked here against every scenario in the
configured split, with a scripted backend standing in for the model.

    python selftest.py
"""
import json
import os
import sys
import traceback

import _compat as compat
import _data_csa as data_csa
import _paths as core_paths
import paths  # noqa: F401
from _verifier import score

import config
import prompts_rt as P
from env_rt import RoundTableEnv


HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, fn):
    try:
        fn()
        print('  ok    %s' % name)
        return 0
    except AssertionError as e:
        print('  FAIL  %s -- %s' % (name, e))
        return 1
    except Exception as e:                           # noqa: BLE001
        print('  FAIL  %s -- %s' % (name, e))
        traceback.print_exc(limit=2)
        return 1


# ------------------------------------------------------------------ data
def t_split():
    d = {k: len(data_csa.load(k)) for k in ('train', 'valid', 'test')}
    total = len(core_paths.DOMAINS) * core_paths.SCENARIOS_PER_DOMAIN
    assert sum(d.values()) == total, (d, total)
    uids = [r['uid'] for k in ('train', 'valid', 'test') for r in data_csa.load(k)]
    assert len(set(uids)) == total, 'split overlaps or drops scenarios'
    print('        %d domains x %d = %d -> %d/%d/%d'
          % (len(core_paths.DOMAINS), core_paths.SCENARIOS_PER_DOMAIN, total,
             d['train'], d['valid'], d['test']))


# ------------------------------------------------------------------ prompts
def t_every_agent_sees_only_its_own():
    """The one thing this arm must get right. Every agent, every scenario."""
    cases = data_csa.load_raw()
    conv = [{'role': 'Meeting', 'content': 'x'}]
    n = 0
    for case in cases:
        for a in case['agents']:
            aid = a['agent_id']
            mine = set(case['views'][aid])
            msgs = P.agent_messages(case, conv, aid)        # asserts internally too
            blob = ' '.join(m['content'] for m in msgs)
            for fid, fact in case['private_facts'].items():
                if fid in mine:
                    continue
                assert fact['text'] not in blob, \
                    '%s leaked to %s in %s' % (fid, aid, case['uid'])
            for f in P.ORACLE_ONLY:
                assert f not in blob, 'oracle field %s leaked in %s' % (f, case['uid'])
            n += 1
    assert n == sum(len(c['agents']) for c in cases), n


def t_chair_holds_no_private_fact():
    """CSA's premise: if the chair could see a private fact it would score without ever
    having to elicit anything, and the benchmark would measure nothing."""
    for case in data_csa.load_raw():
        mine = set(case['views'][case['decision_maker']])
        held = [f for f in case['private_facts'] if f in mine]
        assert not held, '%s: chair holds %s' % (case['uid'], held)


def t_settling_turn_carries_the_schema():
    case = data_csa.load('test')[0]
    conv = [{'role': 'Meeting', 'content': 'x'}]
    plain = ' '.join(m['content'] for m in P.agent_messages(case, conv,
                                                            case['decision_maker']))
    settling = ' '.join(m['content'] for m in P.agent_messages(
        case, conv, case['decision_maker'], settling=True))
    key = list((case['settlement_schema'].get('decisions') or {}))[0]
    assert key not in plain, 'the schema leaked into an ordinary turn'
    assert key in settling, 'the settling turn lost the schema'


def t_converge_prompt_offers_both_replies():
    case = data_csa.load('test')[0]
    aid = sorted(set(case['views']) - {case['decision_maker']})[0]
    msgs = P.agent_messages(case, [], aid, proposal={'decisions': {'x': 'y'}})
    blob = ' '.join(m['content'] for m in msgs)
    assert 'AGREE' in blob, 'the converge prompt never offers agreement'
    assert 'corrected JSON' in blob, 'the converge prompt never offers a correction'


# ------------------------------------------------------------------ consensus
class _Scripted(object):
    """A backend that replays canned turns, so the loop can be exercised with no model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.n_calls = 0
        self.n_failed = 0

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        self.n_calls += 1
        return self.replies.pop(0) if self.replies else ''


def _cfg(**kw):
    class C(object):
        pass
    for k, v in vars(config.Defaults).items():
        if not k.startswith('__'):
            setattr(C, k, v)
    for k, v in kw.items():
        setattr(C, k, v)
    return C


def t_vote_takes_the_majority():
    """Two agents say 'a', one says 'b' -> 'a' wins, regardless of who the chair is."""
    case = data_csa.load('test')[0]
    field = list((case['settlement_schema'].get('decisions') or {}))[0]
    agents = list(case['views'])
    prop = lambda v: json.dumps({'decisions': {field: v},     # noqa: E731
                                 'credited_facts': ['PF1']})

    cfg = _cfg(decide='vote', rounds=0, backend='local')
    env = RoundTableEnv(cfg, _Scripted([]))
    env.reset(case)
    order = env.order[:len(env.names)]
    env.backend = _Scripted([prop('a' if i < 2 else 'b') for i in range(len(order))])
    got = env._settle_vote()
    assert (got.get('decisions') or {}).get(field) == 'a', got
    assert 'PF1' in (got.get('credited_facts') or []), 'fact lists must union, not vote'
    assert len(agents) >= 3


def t_vote_ties_break_to_the_chair():
    case = data_csa.load('test')[0]
    field = list((case['settlement_schema'].get('decisions') or {}))[0]
    cfg = _cfg(decide='vote', rounds=0, backend='local')
    env = RoundTableEnv(cfg, _Scripted([]))
    env.reset(case)
    order = env.order[:len(env.names)]
    # one vote each, so every option ties; the chair's own pick must win
    replies = []
    for aid in order:
        v = 'chair-pick' if aid == env.dm else 'other-%s' % aid
        replies.append(json.dumps({'decisions': {field: v}}))
    env.backend = _Scripted(replies)
    got = env._settle_vote()
    assert (got.get('decisions') or {}).get(field) == 'chair-pick', got


def t_converge_stops_when_everyone_agrees():
    case = data_csa.load('test')[0]
    field = list((case['settlement_schema'].get('decisions') or {}))[0]
    cfg = _cfg(decide='converge', rounds=0, converge_rounds=3, backend='local')
    env = RoundTableEnv(cfg, _Scripted([]))
    env.reset(case)
    n_adv = len(env.order[:len(env.names)]) - 1
    # chair proposes, then every advisor agrees -> exactly one pass, no more
    env.backend = _Scripted([json.dumps({'decisions': {field: 'v'}})]
                            + ['AGREE'] * n_adv + ['SHOULD NOT BE REACHED'] * 5)
    got = env._settle_converge()
    assert (got.get('decisions') or {}).get(field) == 'v', got
    assert env.backend.n_calls == 1 + n_adv, \
        'converge kept going after unanimous agreement (%d calls)' % env.backend.n_calls
    assert all(env.agreed.values()), env.agreed


def t_converge_last_correction_wins():
    case = data_csa.load('test')[0]
    field = list((case['settlement_schema'].get('decisions') or {}))[0]
    cfg = _cfg(decide='converge', rounds=0, converge_rounds=1, backend='local')
    env = RoundTableEnv(cfg, _Scripted([]))
    env.reset(case)
    adv = [a for a in env.order[:len(env.names)] if a != env.dm]
    replies = [json.dumps({'decisions': {field: 'chair'}})]
    for i, _ in enumerate(adv):
        replies.append(json.dumps({'decisions': {field: 'fix-%d' % i}}))
    env.backend = _Scripted(replies)
    got = env._settle_converge()
    assert (got.get('decisions') or {}).get(field) == 'fix-%d' % (len(adv) - 1), got


def t_record_schema_matches_the_other_arms():
    """compute_extended_metrics reads these keys off every arm; a missing one silently
    drops this arm out of a whole table."""
    case = data_csa.load('test')[0]
    cfg = _cfg(decide='chair', rounds=1, backend='local')
    env = RoundTableEnv(cfg, _Scripted(['hello'] * 40))
    rec = env.run(case)
    required = {'uid', 'domain', 'num_agents', 'scenario_type', 'settlement', 'score',
                'floor', 'revealed', 'reveal_elicited', 'reveal_turn', 'addressed',
                'leaks', 'done', 'turns', 'max_turn', 'n_calls', 'calls_by_role',
                'prompt_chars', 'dialog', 'reward'}
    missing = required - set(rec)
    assert not missing, 'record is missing %s' % sorted(missing)
    for k in ('dca', 'cbar', 'pbar', 'disclosure_rate', 'schema_valid'):
        assert k in rec['score'], 'score has no %r' % k
    assert all('speaker' in t for t in rec['dialog']), \
        'dialog turns need a speaker tag or section F mis-attributes chair turns'


def t_vendored_copies_match_core():
    """The vendored copies must still render identically from csa_core.

    This folder is standalone on purpose, which means its scoring rule CAN drift from the
    one the other arms use -- and drift would not break anything, it would quietly make
    the numbers incomparable. vendor.py renders what each copy SHOULD be, so it is the
    one definition of "unchanged"; re-implementing the comparison here would just be a
    second thing to keep in sync.

    Skips, loudly, when csa_core is not reachable -- which is the normal state once this
    folder has been lifted out on its own.
    """
    import vendor
    if not os.path.isdir(vendor.CORE):
        print('        (csa_core not reachable; running standalone, drift unchecked)')
        return
    drift = []
    for src_name, out_name, rewrites in vendor.PLAN:
        want = vendor.render(src_name, rewrites)
        path = os.path.join(HERE, out_name)
        have = open(path, encoding='utf-8').read() if os.path.isfile(path) else None
        if have != want:
            drift.append(out_name)
    assert not drift, ('vendored copies have drifted from csa_core: %s. '
                       'Run: python vendor.py' % ', '.join(drift))


def main():
    print()
    print('raw scenarios : %s' % core_paths.find_raw())
    print('backend       : %s / %s' % (config.Defaults.model, config.Defaults.api_model))
    print()
    bad = 0
    for name, fn in (
            ('split is complete and invariants hold', t_split),
            ('every agent sees only its own facts', t_every_agent_sees_only_its_own),
            ('chair holds no private fact', t_chair_holds_no_private_fact),
            ('settling turn carries the schema', t_settling_turn_carries_the_schema),
            ('converge prompt offers agree and correct',
             t_converge_prompt_offers_both_replies),
            ('vote takes the majority, facts union', t_vote_takes_the_majority),
            ('vote ties break to the chair', t_vote_ties_break_to_the_chair),
            ('converge stops on unanimous agreement',
             t_converge_stops_when_everyone_agrees),
            ('converge keeps the last correction', t_converge_last_correction_wins),
            ('record schema matches the other arms',
             t_record_schema_matches_the_other_arms),
            ('vendored copies match csa_core', t_vendored_copies_match_core)):
        bad += run(name, fn)

    print()
    problems = compat.problems()
    if problems:
        print('logic checks passed, but the environment BLOCKS inference:')
        for p in problems:
            print('  - %s' % p)
    else:
        print('all checks passed')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
