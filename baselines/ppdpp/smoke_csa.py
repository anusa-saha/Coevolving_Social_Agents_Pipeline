"""End-to-end verification of the csa adapter, with a scripted LLM in place of a real one.

Stubs the serving stack so env.py imports without it, then drives the real Env.step and
compute_reward paths. Runs in seconds with no GPU. Its job is to catch the failures that
would otherwise be silent: a private fact reaching the wrong prompt, a reward wired to
the wrong signal, shaping that is not potential-based, sampling settings drifting from
upstream.

    python smoke_csa.py
"""
import os
import sys
import types
import re as _re

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def _stubs():
    fc = types.ModuleType('fastchat'); fcm = types.ModuleType('fastchat.model')
    fcm.load_model = lambda *a, **k: (None, None)
    fcm.get_conversation_template = lambda *a, **k: None
    fcm.add_model_args = lambda p: None
    fc.model = fcm
    sys.modules.setdefault('fastchat', fc); sys.modules.setdefault('fastchat.model', fcm)
    op = types.ModuleType('openai'); op.api_key = None
    op.ChatCompletion = types.SimpleNamespace(create=lambda **k: None)
    sys.modules.setdefault('openai', op)
    nl = types.ModuleType('nltk')
    nl.sent_tokenize = lambda t: [s for s in _re.split(r'(?<=[.!?])\s+', t) if s]
    sys.modules.setdefault('nltk', nl)


_stubs()

import env as envmod                                            # noqa: E402
from env import Env                                             # noqa: E402
from prompt import CSAMessages, CSAAct, CSA_ELICITING_ACTS      # noqa: E402
from utils import load_dataset                                  # noqa: E402
_CSA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CSA_ROOT not in sys.path:
    sys.path.insert(0, _CSA_ROOT)
from csa_core.verifier import score, floor_score, canonical    # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('%-60s %s' % (name, 'ok' if cond else 'FAIL ' + str(detail)))
    if not cond:
        FAILS.append(name)


class Args:
    data_name = 'csa'
    system = user = critic = 'chatgpt'
    max_turn = 36
    max_new_tokens = 96
    max_seq_length = 512
    seed = 1
    gamma = 0.999
    csa_reward = 'critic'
    critic_every = 1
    critic_batch = 10
    csa_reveal_threshold = 0.35
    csa_settlement_max_tokens = 512
    csa_w_content = 0.5
    csa_w_prov = 0.5
    csa_w_halluc = 0.0
    csa_w_schema = 0.0
    csa_w_accept = 0.0
    csa_w_shape_disc = 1.0
    csa_w_shape_elic = 0.0
    csa_w_shape_cover = 0.0
    openai_model = 'gpt-3.5-turbo-0613'
    device = 'cpu'


# ------------------------------------------------------------------ 1. hidden profiles
def test_views(case):
    dm = case['decision_maker']
    chair = ' '.join(m['content'] for m in CSAMessages(case, 'system', [], action='ask'))
    leaked = [f for f, v in case['private_facts'].items() if v['text'][:60] in chair]
    check('chair prompt leaks no private fact', not leaked, leaked)

    for fid, fact in case['private_facts'].items():
        own = ' '.join(m['content'] for m in
                       CSAMessages(case, 'user', [], agent_id=fact['owner']))
        check('  %s reaches its owner %s' % (fid, fact['owner']), fact['text'][:60] in own)
        for other in (a['agent_id'] for a in case['agents']):
            if other in (fact['owner'], dm):
                continue
            oth = ' '.join(m['content'] for m in CSAMessages(case, 'user', [], agent_id=other))
            check('  %s hidden from %s' % (fid, other), fact['text'][:60] not in oth)

    allp = chair + ' '.join(' '.join(m['content'] for m in
                                     CSAMessages(case, 'user', [], agent_id=a['agent_id']))
                            for a in case['agents'] if a['agent_id'] != dm)
    check('no content check expression in any participant prompt',
          not any(e in allp for e in case['content_checks'].values()))
    check('no acceptance condition in any participant prompt',
          not any(a[:40] in allp for a in case['acceptance_conditions']))

    none = ' '.join(m['content'] for m in CSAMessages(case, 'system', [], action=None))
    check('no-planner prompt carries no act instruction',
          not any(v in none for v in CSAAct.values()))
    check('decide prompt carries the settlement schema',
          'settlement schema is' in ' '.join(
              m['content'] for m in CSAMessages(case, 'system', [], action='decide')))

    crit = ' '.join(m['content'] for m in CSAMessages(case, 'critic', []))
    check('critic sees the private facts (oracle)',
          all(f['text'][:40] in crit for f in case['private_facts'].values()))
    check('critic does not see the checks',
          not any(e in crit for e in case['content_checks'].values()))


# ------------------------------------------------------------------ 2. verifier
def test_verifier(case):
    good = {
        'decisions': {'medication_approved': 'dalbavancin',
                      'dose_schedule': '1500 mg IV today, second dose 1500 mg on day 8',
                      'dispensing_pharmacy': 'NorthStar Specialty Pharmacy',
                      'prior_authorization_submitted': True,
                      'prior_authorization_deadline': '3:00 PM today'},
        'credited_facts': ['PF1', 'PF2'], 'justification_fact_ids': ['PF1', 'PF2'],
        'commitments': [{'type': 'dispense', 'target': 'A1',
                         'detail': 'Send the order to NorthStar Specialty Pharmacy'}]}
    s = score(case, good, {'PF1', 'PF2'})
    check('gold settlement passes every content check', s['all_content'], s['content'])
    check('gold settlement passes provenance', s['all_prov'], s['provenance'])
    check('gold settlement is a joint success', s['joint'])
    check('gold has no hallucinated credit', s['hallucinated_credit'] == 0.0)

    f = floor_score(case)
    check('floor is not a joint success', not f['joint'])
    check('floor scores 0 on decisive checks', f['dca'] == 0.0, f['dca'])

    s2 = score(case, good, set())
    check('credit without disclosure fails provenance', not s2['all_prov'])
    check('credit without disclosure is flagged hallucinated', s2['hallucinated_credit'] > 0)

    loose = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in good.items()}
    loose['decisions']['prior_authorization_deadline'] = '3 p.m. today'
    check('reformatted deadline fails strict',
          not score(case, loose, {'PF1', 'PF2'})['all_content'])
    check('reformatted deadline passes normalised',
          score(case, loose, {'PF1', 'PF2'}, norm=True)['all_content'])
    check('canonical folds 3:00 PM and 3 p.m.',
          canonical('3:00 PM today') == canonical('3 p.m. today'))
    check('missing fields fail rather than raise',
          not score(case, {'decisions': {}}, set())['all_content'])


# ------------------------------------------------------------------ 3. episode
def scripted(case, disclose=True, volunteer=False):
    """Advisors recite their fact; the chair speaks generically unless finalising."""
    def f(api_key=None, messages=None, model=None, max_tokens=None, temperature=0, n=1):
        blob = ' '.join(m['content'] for m in messages)
        if n > 1:                                    # critic
            transcript = blob.split('The following is the meeting:')[-1]
            ok = all(v['text'][:50] in transcript for v in case['private_facts'].values())
            return [('Yes, the settlement reflects all of the decisive information the '
                     'advisors held.') if ok else
                    'No, the settlement ignores the information the advisors held.'] * n
        if 'Extract the final settlement' in blob:
            return '{"decisions": {}, "credited_facts": [], "commitments": []}'
        for fid, fact in case['private_facts'].items():
            if fact['text'][:60] in blob:            # this advisor's own prompt
                return fact['text'] if disclose else 'Nothing to add.'
        if 'Reply with the JSON object only' in blob:
            return '{"decisions": {"x": "y"}, "credited_facts": ["PF1"]}'
        return 'Let us settle this. What do you each know?'
    return f


def test_episode(rows):
    case = rows[0]
    args = Args()
    env = Env(args, {'train': rows, 'test': rows}, mode='test')
    envmod.query_openai_model = scripted(case)
    env.reset()

    order = case['interaction_config']['turn_order']
    dm_name = env.names[case['decision_maker']]
    check('reset uses the per-scenario cap, not the global one',
          env.utterance_cap == case['interaction_config']['turn_cap'])
    check('max_turn counts chair slots inside the cap',
          0 < env.max_turn <= env.utterance_cap)
    check('conversation is seeded with the shared framing',
          env.conversation[0]['role'] == 'Meeting')

    conv, reward, done = env.step('ask')
    check('one step yields exactly one chair utterance',
          sum(1 for t in conv if t['role'] == dm_name) == 1,
          [t['role'] for t in conv])
    check('advisors before the chair spoke first',
          len(conv) == 1 + order.index(case['decision_maker']) + 1, len(conv))
    check('critic reward is in range', -1.0 <= reward <= 1.0, reward)

    steps, guard = 1, 0
    while not done and guard < 200:
        conv, reward, done = env.step('followup'); steps += 1; guard += 1
    check('episode terminates', bool(done), done)
    check('chair spoke once per step',
          sum(1 for t in conv if t['role'] == dm_name) == steps)
    check('never exceeds the chair-turn budget', steps <= env.max_turn)
    check('disclosures recorded', len(env.revealed) > 0, env.revealed)
    check('every disclosure carries an elicitation flag',
          set(env.reveal_elicited) == set(env.revealed))
    check('followup counts as an eliciting act',
          all(env.reveal_elicited.values()), env.reveal_elicited)

    # Silent advisors must not succeed.
    envmod.query_openai_model = scripted(case, disclose=False)
    env.test_num = 0; env.reset()
    d, guard = 0, 0
    while not d and guard < 200:
        _, _, d = env.step('followup'); guard += 1
    check('silent advisors do not produce a success', d == -1, d)
    check('silent run discloses nothing', not env.revealed, env.revealed)

    # Volunteered vs elicited: a non-eliciting act must not be credited with elicitation.
    envmod.query_openai_model = scripted(case)
    env.test_num = 0; env.reset()
    d, guard = 0, 0
    while not d and guard < 30:
        _, _, d = env.step('share'); guard += 1
    if env.reveal_elicited:
        check('share is not credited as elicitation',
              not any(env.reveal_elicited.values()), env.reveal_elicited)


# ------------------------------------------------------------------ 4. reward wiring
def test_reward(rows):
    case = rows[0]
    args = Args(); args.csa_reward = 'verifier'
    env = Env(args, {'train': rows, 'test': rows}, mode='test')
    calls = {'n': 0}
    base = scripted(case)

    def counting(**kw):
        if kw.get('n', 1) > 1:
            calls['n'] += 1
        return base(**kw)

    envmod.query_openai_model = counting
    env.reset()
    d, guard = 0, 0
    while not d and guard < 200:
        _, reward, d = env.step('followup'); guard += 1
    check('verifier arm makes zero critic calls', calls['n'] == 0, calls['n'])
    check('verifier terminal reward in range', -1.0 <= reward <= 1.0, reward)
    check('terminal score recorded', env.last_score is not None)

    # Shaping must be potential-based: zero when the potential does not move.
    env.test_num = 0; env.reset()
    env.prev_phi = env._csa_potentials()
    check('shaping is zero when no potential changes',
          abs(env._csa_shaping()) < 1e-9, env._csa_shaping())
    env.revealed = {d_['fact_id'] for d_ in case['decisive_facts']}
    env.prev_phi = {'disclosure': 0.0, 'elicitation': 0.0, 'coverage': 0.0}
    check('shaping is positive when disclosure advances', env._csa_shaping() > 0)

    # Weights must actually gate their terms.
    args.csa_w_halluc = 1.0
    s = score(case, {'credited_facts': ['PF1'], 'justification_fact_ids': ['PF1']}, set())
    r_pen = env._csa_terminal_reward(s)
    args.csa_w_halluc = 0.0
    check('hallucination penalty lowers the terminal reward',
          r_pen < env._csa_terminal_reward(s), (r_pen,))
    check('acceptance judge is off by default', Args.csa_w_accept == 0.0)


# ------------------------------------------------------------------ 5. leakage
def test_leakage(rows):
    case = rows[0]
    args = Args()
    env = Env(args, {'train': rows, 'test': rows}, mode='test')
    envmod.query_openai_model = scripted(case)
    env.reset()
    fid, fact = sorted(case['private_facts'].items())[0]
    other = next(a['agent_id'] for a in case['agents']
                 if a['agent_id'] not in (fact['owner'], case['decision_maker']))
    env._csa_note_leaks(other, fact['text'])
    check('an agent stating a fact it never saw is flagged as a leak',
          any(l['fact'] == fid for l in env.leaks), env.leaks)
    env.leaks = []
    env._csa_note_leaks(fact['owner'], fact['text'])
    check('the owner stating its own fact is not a leak', not env.leaks, env.leaks)


# ------------------------------------------------------------------ 6. json
def test_json():
    p = Env._csa_parse_json
    check('parses fenced json', p('```json\n{"a": 1}\n```') == {'a': 1})
    check('parses json with prose around it', p('Sure! {"a": {"b": 2}} done') == {'a': {'b': 2}})
    check('returns {} on garbage', p('no json here') == {})
    check('returns {} on truncated json', p('{"a": ') == {})


def main():
    data = load_dataset('csa')
    allc = data['train'] + data['valid'] + data['test']
    target = next(c for c in allc if c['uid'] == 'healthcare::scenario_1')

    print('\n--- hidden profiles ---'); test_views(target)
    print('\n--- verifier ---'); test_verifier(target)
    print('\n--- json extraction ---'); test_json()
    print('\n--- episode loop ---'); test_episode([target])
    print('\n--- reward wiring ---'); test_reward([target])
    print('\n--- leakage detection ---'); test_leakage([target])

    print('\n--- corpus sweep (whole corpus) ---')
    args = Args()
    env = Env(args, {'train': allc, 'test': allc}, mode='test')
    caps, budgets = [], []
    for i in range(len(allc)):
        env.test_num = i; env.reset()
        caps.append(env.max_turn); budgets.append(env.utterance_cap)
    check('every scenario yields at least one chair turn', min(caps) >= 1)
    check('chair turns never exceed the utterance budget',
          all(c <= b for c, b in zip(caps, budgets)))
    print('    chair turns: min %d max %d | utterance cap: min %d max %d'
          % (min(caps), max(caps), min(budgets), max(budgets)))

    print('\n%s' % ('ALL CHECKS PASSED' if not FAILS else 'FAILURES: %s' % FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
