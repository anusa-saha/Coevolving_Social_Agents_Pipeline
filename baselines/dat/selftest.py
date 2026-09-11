"""No-GPU checks. Run before spending inference time.

    python selftest.py

Everything here runs on CPU in about a minute and loads no model. The checks are the ones
that would otherwise fail silently and waste a run: a chair prompt that is not the control
after all, a planner whose prefix has the wrong shape, a reward that has drifted from
EPO's, a record missing a field the metrics scripts read.
"""
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch

import paths                                         # noqa: E402
from csa_core import data_csa                        # noqa: E402
from csa_core import verifier as V                   # noqa: E402
import config                                        # noqa: E402
import metrics_dat as M                              # noqa: E402
import planner as PL                                 # noqa: E402
import prompts_dat as P                              # noqa: E402
import reward_dat as R                               # noqa: E402
from env_dat import DATEnv                           # noqa: E402
from prompt import CSAMessages                       # noqa: E402
from td3bc import ReplayBuffer, TD3BC                # noqa: E402

FAILS = []
CONV = [{'role': 'Meeting', 'content': 'The group convenes to decide: x'}]


def check(name, fn):
    try:
        fn()
        print('  ok    %s' % name)
    except Exception as e:                           # noqa: BLE001
        FAILS.append((name, e))
        print('  FAIL  %s -- %s' % (name, e))
        traceback.print_exc(limit=2)


# ------------------------------------------------------------------ the instrument
def t_split():
    got = {k: len(v) for k, v in data_csa.load().items()}
    if paths.is_published_config():
        assert got == {'train': 99, 'valid': 9, 'test': 42}, got
    else:
        total = len(paths.DOMAINS) * paths.SCENARIOS_PER_DOMAIN
        assert sum(got.values()) == total, (got, total)
        uids = [r['uid'] for k in ('train', 'valid', 'test') for r in data_csa.load(k)]
        assert len(set(uids)) == total, 'split overlaps or drops scenarios'
        print('        %d domains x %d = %d -> %d/%d/%d'
              % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN, total,
                 got['train'], got['valid'], got['test']))
    assert not data_csa.check_invariants(data_csa.load_raw())


def t_split_matches_published():
    if not paths.is_published_config():
        print('        (skipped: benchmark reconfigured to %d domains x %d)'
              % (len(paths.DOMAINS), paths.SCENARIOS_PER_DOMAIN))
        return
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


# ------------------------------------------------------------------ the prompt
def t_chair_prompt_is_the_control():
    """DAT's claim is that only the prefix differs. If the chair prompt is not the
    no-planner control byte-for-byte, a gain could be the wording instead."""
    n = 0
    for case in data_csa.load_raw():
        mine = P.chair_messages(case, CONV)
        theirs = CSAMessages(case, 'system', CONV, action=None)
        assert mine == theirs, case['uid']
        settling = P.chair_messages(case, CONV, settling=True)
        assert settling == CSAMessages(case, 'system', CONV, action='decide'), case['uid']
        n += 1
    print('        %d scenarios, non-settling and settling both identical' % n)


def t_prompts_are_filtered():
    for case in data_csa.load_raw():
        for settling in (False, True):
            P.assert_filtered(case, P.chair_messages(case, CONV, settling=settling))


def t_advisors_see_only_own_facts():
    for case in data_csa.load_raw()[:40]:
        for a in case['agents']:
            if a['agent_id'] == case['decision_maker']:
                continue
            blob = ' '.join(m['content'] for m in
                            P.advisor_messages(case, CONV, a['agent_id']))
            for fid, fact in case['private_facts'].items():
                if fact['owner'] != a['agent_id']:
                    assert fact['text'] not in blob, (a['agent_id'], fid)


def t_settling_turn_carries_the_schema():
    case = data_csa.load('test')[0]
    body = P.chair_messages(case, CONV, settling=True)[1]['content']
    assert 'settlement schema is' in body
    assert 'JSON object only' in body
    plain = P.chair_messages(case, CONV)[1]['content']
    assert 'settlement schema is' not in plain


# ------------------------------------------------------------------ the planner
def t_planner_shapes():
    p = PL.DATPlanner(d_model=32, action_dim=8, n_prefix=2, hidden=16, layers=2)
    s = torch.randn(32)
    assert p.action(s).shape == (8,)
    assert p.prefix_from_state(s).shape == (2, 32)
    b = torch.randn(5, 32)
    assert p.action(b).shape == (5, 8)
    assert p.prefix_from_state(b).shape == (5, 2, 32)
    # the RL head is zero at init, so `dat` starts exactly at `selfclone`
    assert float(p.rl_action(s).detach().abs().max()) == 0.0
    assert torch.allclose(p.action(s), p.base_action(s))


def t_planner_roundtrip():
    p = PL.DATPlanner(d_model=16, action_dim=4, n_prefix=3, hidden=8, layers=1)
    p.set_state_stats(np.arange(16), np.ones(16) * 2)
    p.set_max_action(0.7)
    s = torch.randn(16)
    a = p.action(s)
    path = os.path.join(paths.CKPT, '_selftest.pt')
    p.save(path)
    q = PL.DATPlanner.load(path)
    assert torch.allclose(q.action(s), a, atol=1e-6)
    assert abs(float(q.max_action) - 0.7) < 1e-6      # stored as float32
    assert torch.allclose(q.state_mu, p.state_mu)
    os.remove(path)
    os.remove(os.path.splitext(path)[0] + '.json')


def t_state_normalisation():
    p = PL.DATPlanner(d_model=8, action_dim=2, hidden=4, layers=1)
    S = torch.randn(200, 8) * 3 + 5
    p.set_state_stats(S.mean(0), S.std(0))
    z = p.normalise(S)
    assert float(z.mean().abs()) < 0.05, float(z.mean())
    assert abs(float(z.std()) - 1.0) < 0.1, float(z.std())


def t_pca_up_map():
    p = PL.DATPlanner(d_model=12, action_dim=4, n_prefix=2, hidden=8, layers=1)
    PL.init_up_from_embeddings(p, torch.randn(300, 12))
    assert p.up.weight.shape == (2 * 12, 4)
    assert p.prefix(torch.randn(4)).shape == (2, 12)


def t_rl_head_is_bounded():
    p = PL.DATPlanner(d_model=8, action_dim=4, hidden=8, layers=1)
    p.set_max_action(0.5)
    for m in p.rl.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=5.0)
            torch.nn.init.normal_(m.bias, std=5.0)
    u = p.rl_action(torch.randn(64, 8))
    assert float(u.abs().max()) <= 0.5 + 1e-6, float(u.abs().max())


# ------------------------------------------------------------------ the steering
class _FakeTok(object):
    """Just enough tokenizer to drive SteeredLM. No download, no vocabulary."""

    pad_token_id = 0
    eos_token = '</s>'
    chat_template = 'present'

    def __call__(self, texts, return_tensors=None, add_special_tokens=True):
        import types
        if isinstance(texts, str):
            texts = [texts]
        n = max(4, min(24, len(texts[0]) // 5))
        return types.SimpleNamespace(input_ids=torch.arange(1, n + 1).unsqueeze(0))

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            **kw):
        if kw:                                       # mimic a tokenizer without
            raise TypeError('enable_thinking')       # enable_thinking, as Qwen2.5 is
        return ' '.join(m['content'] for m in messages)


def t_steering_mechanics():
    """The prefix path, on a two-layer Qwen2 built from config -- nothing is downloaded.

    This is where the subtle bugs live: an off-by-one in the clone loss trains on the
    wrong tokens, a generate() that echoes its prompt returns the prompt as an utterance,
    and a backward that reaches the LM's parameters quietly turns DAT into fine-tuning.
    All three are checked here rather than discovered after a GPU run.
    """
    try:
        from transformers import AutoModelForCausalLM, Qwen2Config
    except Exception as e:                           # noqa: BLE001
        print('        (skipped: %s)' % e)
        return
    from steering import SteeredLM

    cfg = config.Defaults
    dtype, device = cfg.dtype, cfg.device
    cfg.dtype, cfg.device = 'float32', 'cpu'
    try:
        mcfg = Qwen2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2,
                           intermediate_size=64, max_position_embeddings=128)
        lm = SteeredLM(cfg, model=AutoModelForCausalLM.from_config(mcfg),
                       tokenizer=_FakeTok())
        msgs = [{'role': 'system', 'content': 'a chair prompt of some believable length'},
                {'role': 'USER', 'content': 'please take your turn now'}]
        ids = lm.encode(msgs, 'Chair')
        s = lm.state(ids)
        assert s.shape == (lm.d_model,) and s.dtype == torch.float32

        p = PL.DATPlanner(lm.d_model, action_dim=8, n_prefix=2, hidden=16, layers=2)
        prefix = p.prefix_from_state(s).detach()
        assert prefix.shape == (2, lm.d_model)

        # generate() must return the UTTERANCE, not the prompt echoed back
        steered = lm.generate(ids, 5, prefix=prefix)
        assert len(steered.split()) <= 5, steered
        assert len(lm.generate(ids, 5, prefix=None).split()) <= 5

        # the clone gradient reaches the planner and nothing else
        tgt = torch.arange(3, 9).unsqueeze(0)
        loss = lm.clone_loss(ids, tgt, p.prefix_from_state(s))
        loss.backward()
        mass = sum(float(x.grad.abs().sum()) for x in p.clone_parameters()
                   if x.grad is not None)
        assert mass > 0, 'no gradient reached the planner'
        assert not any(x.grad is not None for x in lm.model.parameters()), (
            'a gradient reached the language model -- this is not DAT any more')

        # checkpointing must not change the loss, only where the activations live
        for x in p.clone_parameters():
            x.grad = None
        lm.enable_gradient_checkpointing()
        l2 = lm.clone_loss(ids, tgt, p.prefix_from_state(s))
        l2.backward()
        m2 = sum(float(x.grad.abs().sum()) for x in p.clone_parameters()
                 if x.grad is not None)
        lm.disable_gradient_checkpointing()
        assert abs(float(l2) - float(loss)) < 1e-4, (float(l2), float(loss))
        assert m2 > 0, 'use_reentrant=True would drop the graph here'
        print('        clone NLL %.4f, gradient mass %.4f, identical checkpointed'
              % (float(loss), mass))
    finally:
        cfg.dtype, cfg.device = dtype, device


def t_episode_loop():
    """A whole episode, end to end, on the tiny model: reset -> step -> record.

    The text is meaningless -- a two-layer random transformer is not a chair -- and that
    is fine, because what this checks is the wiring: that the unsteered arm takes no
    planner forwards and produces no transitions, that the steered arms produce one state
    and one action per chair turn, that the clone-pair collector skips the settling turn,
    and that the whole record still survives repr/literal_eval.
    """
    try:
        from transformers import AutoModelForCausalLM, Qwen2Config
    except Exception as e:                           # noqa: BLE001
        print('        (skipped: %s)' % e)
        return
    import ast

    import env_dat
    from steering import SteeredLM

    cfg = config.Defaults
    saved = (cfg.dtype, cfg.device, cfg.max_new_tokens, cfg.settlement_max_tokens)
    cfg.dtype, cfg.device = 'float32', 'cpu'
    cfg.max_new_tokens, cfg.settlement_max_tokens = 6, 8
    # Sentence trimming needs nltk's punkt, which is a download; it is orthogonal to the
    # loop and identical to the other arms', so it is switched off for the check rather
    # than made a prerequisite of running the selftest.
    sent, env_dat._SENT = env_dat._SENT, None
    try:
        mcfg = Qwen2Config(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2,
                           intermediate_size=64, max_position_embeddings=256)
        lm = SteeredLM(cfg, model=AutoModelForCausalLM.from_config(mcfg),
                       tokenizer=_FakeTok())
        p = PL.DATPlanner(lm.d_model, action_dim=8, n_prefix=2, hidden=16, layers=2)
        env = env_dat.DATEnv(cfg, lm=lm, planner=p)
        case = data_csa.load('test')[0]
        gen = torch.Generator().manual_seed(0)

        for arm in env_dat.ARMS:
            env.reset(case, arm=arm)
            done, t = 0, 0
            while not done:
                _c, done = env.step(sigma=(0.25 if arm == 'selfclone' else 0.0),
                                    generator=gen)
                t += 1
            rec, tr = env.record(t), env.transitions()
            assert ast.literal_eval(str(rec))['uid'] == case['uid'], arm
            assert t == env.max_turn, (arm, t, env.max_turn)
            if arm == 'unsteered':
                assert rec['planner_forwards'] == 0 and tr == [] and not env.states
                assert rec['action_cos'] is None
            else:
                assert rec['planner_forwards'] == t
                assert len(env.states) == len(env.actions) == t
                assert len(tr) == t and tr[-1]['done'] == 1.0
        # an untrained RL head and no noise means the same action every turn, which is
        # exactly the collapse `action_cos` exists to expose
        assert abs(rec['action_cos'] - 1.0) < 1e-5, rec['action_cos']

        env.collect_clone_pairs = True
        env.reset(case, arm='unsteered')
        done = 0
        while not done:
            _c, done = env.step()
        assert len(env.clone_pairs) == env.max_turn - 1, 'the settling turn must be skipped'
        assert set(env.clone_pairs[0]) == {'prompt', 'target', 'uid', 'turn'}
        print('        %d chair turns, %d clone pairs, all three arms'
              % (env.max_turn, len(env.clone_pairs)))
    finally:
        env_dat._SENT = sent
        (cfg.dtype, cfg.device, cfg.max_new_tokens, cfg.settlement_max_tokens) = saved


# ------------------------------------------------------------------ the reward
def _epo_prm():
    root = os.path.dirname(paths.HERE)
    path = os.path.join(root, 'epo', 'prm.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('_epo_prm', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_trace(rng, n_turns=3):
    facts = ['PF%d' % i for i in range(1, 4)]
    decisive = [{'fact_id': f, 'flips': ['C%d' % k for k in range(1, rng.integers(2, 4))]}
                for f in facts if rng.random() < 0.8]
    reveal_turn = {f: int(rng.integers(0, n_turns + 1)) for f in facts
                   if rng.random() < 0.7}
    elicited = {f: bool(rng.random() < 0.5) for f in reveal_turn}
    return R.Trace(n_turns=n_turns, reveal_turn=reveal_turn, reveal_elicited=elicited,
                   decisive=decisive,
                   settle_turn=(int(rng.integers(0, n_turns)) if rng.random() < 0.8
                                else None),
                   dca=float(rng.random()), schema_valid=bool(rng.random() < 0.8),
                   leaks=([{'fact': 'PF1'}] if rng.random() < 0.3 else []),
                   acts=['elicit'] * n_turns, conversation=[], case={})


def t_reward_matches_epo():
    """Two RL arms supervised by different rewards are not comparable.

    EPO's VerifierPRM is the reference. DAT's VerifierReward is the same rule with a
    different default mode, and this asserts they agree turn-for-turn.
    """
    epo = _epo_prm()
    if epo is None:
        print('        (skipped: epo/prm.py not present)')
        return
    rng = np.random.default_rng(0)
    for mode in ('binary', 'graded'):
        mine = R.VerifierReward(mode=mode, done_tau=0.6)
        theirs = epo.VerifierPRM(mode=mode, done_tau=0.6)
        for _ in range(200):
            tr = _random_trace(rng)
            a, b = mine(tr), theirs(tr)
            assert a == b, (mode, a, b)
    print('        400 random traces, both modes, identical to epo/prm.py')


def t_terminal_reward_bounds():
    cfg = config.Defaults
    good = {'schema_valid': True, 'disclosure_rate': 1.0, 'dca': 1.0, 'close': 1.0,
            'hallucinated_credit': 0.0}
    assert abs(R.terminal_reward(cfg, good, []) - 1.0) < 1e-9
    assert R.terminal_reward(cfg, dict(good, schema_valid=False), []) == -1.0
    assert R.terminal_reward(cfg, good, [{'fact': 'PF1'}]) == -1.0
    dead = {'schema_valid': True, 'disclosure_rate': 0.0, 'dca': 0.0, 'close': 0.4,
            'hallucinated_credit': 0.0}
    assert R.terminal_reward(cfg, dead, []) == -0.5
    for hc in (0.0, 0.5, 1.0):
        r = R.terminal_reward(cfg, dict(good, hallucinated_credit=hc), [])
        assert -1.0 <= r <= 1.0


# ------------------------------------------------------------------ the record
def _stub_env(case, arm='dat', n_turns=2, d=8, adim=4):
    """A DATEnv with its fields set by hand and no model behind it.

    Built with __new__ so every method is real: this exercises record(), trace(),
    turn_rewards() and transitions() exactly as a run would, without a GPU.
    """
    e = DATEnv.__new__(DATEnv)
    e.cfg = config.Defaults
    e.case, e.arm = case, arm
    e.dm = case['decision_maker']
    e.names = {a['agent_id']: a['name'] for a in case['agents']}
    e.advisors = set(e.names) - {e.dm}
    e.reward_fn = R.build(e.cfg)
    e.step_i = n_turns
    e.revealed = set(list(case['private_facts'])[:1])
    e.reveal_turn = {f: 0 for f in e.revealed}
    e.reveal_elicited = {f: True for f in e.revealed}
    e.addressed = set(list(e.advisors)[:1])
    e.cover_credit = {0: set(list(e.advisors)[:1])}
    e.leaks = []
    e.settlement = {}
    e.settle_turn = n_turns - 1
    e.n_calls, e.calls_by_role, e.prompt_chars = 12, {'chair': 3}, 4321
    e.planner_forwards = n_turns
    e.derived_acts = ['elicit', 'decide'][:n_turns]
    e.conversation = [{'role': 'Meeting', 'content': 'x'},
                      {'role': e.names[e.dm], 'content': 'Could you confirm the limit?'}]
    g = torch.Generator().manual_seed(0)
    e.states = [torch.randn(d, generator=g) for _ in range(n_turns)]
    e.actions = [torch.randn(adim, generator=g) for _ in range(n_turns)]
    e.residuals = [torch.zeros(adim) for _ in range(n_turns)]
    e.last_score = V.score(case, {}, e.revealed)
    e.last_score_norm = V.score(case, {}, e.revealed, norm=True)
    return e


# every key the metric scripts read off a record
NEEDED = ('dialog', 'reward', 'uid', 'domain', 'num_agents', 'scenario_type',
          'settlement', 'score', 'score_norm', 'floor', 'revealed', 'reveal_elicited',
          'reveal_turn', 'addressed', 'leaks', 'done', 'turns', 'max_turn', 'n_calls',
          'calls_by_role', 'prompt_chars')


def t_record_schema():
    case = data_csa.load('test')[0]
    e = _stub_env(case)
    e.max_turn = 2
    rec = DATEnv.record(e, turns=2)
    missing = [k for k in NEEDED if k not in rec]
    assert not missing, missing
    assert rec['arm'] == 'dat'
    assert 'act_history' not in rec, 'DAT has no acts; that field belongs to a planner arm'
    assert rec['derived_acts'] == ['elicit', 'decide']
    # the dialog must carry speaker tags, or section F counts zero chair turns
    assert {t['speaker'] for t in rec['dialog']} <= {'env', 'sys', 'usr'}
    assert any(t['speaker'] == 'sys' for t in rec['dialog'])
    # and it must be repr/literal_eval round-trippable, which is how records are stored.
    # A bare float('nan') anywhere in a record is not a literal: the loader skips the
    # block and the arm vanishes from the metrics tables without an error. The unsteered
    # arm is the case that bites, because it records no actions at all.
    import ast
    assert ast.literal_eval(str(rec))['uid'] == rec['uid']
    for n_actions in (0, 1):
        e2 = _stub_env(case)
        e2.max_turn = 2
        e2.actions = e2.actions[:n_actions]
        e2.residuals = e2.residuals[:n_actions]
        r2 = DATEnv.record(e2, turns=2)
        assert r2['action_cos'] is None, r2['action_cos']
        assert ast.literal_eval(str(r2))['uid'] == r2['uid'], n_actions


def t_transitions():
    case = data_csa.load('test')[0]
    e = _stub_env(case, n_turns=3)
    e.max_turn = 3
    tr = DATEnv.transitions(e)
    assert len(tr) == 3
    assert tr[-1]['done'] == 1.0 and tr[0]['done'] == 0.0
    assert np.allclose(tr[-1]['s2'], 0.0), 'the last successor state must be zeroed'
    assert np.allclose(tr[0]['s2'], e.states[1].numpy())
    assert tr[0]['s'].shape == (8,) and tr[0]['u'].shape == (4,)


def t_buffer_roundtrip():
    rng = np.random.default_rng(0)
    rows = [{'s': rng.normal(size=6), 'u': rng.normal(size=3), 'r': float(rng.random()),
             's2': rng.normal(size=6), 'done': float(i % 3 == 2)} for i in range(30)]
    buf = ReplayBuffer.from_transitions(rows)
    path = os.path.join(paths.DATA, '_selftest.npz')
    buf.save(path)
    back = ReplayBuffer.load(path)
    assert len(back) == 30 and back.s.shape == (30, 6) and back.u.shape == (30, 3)
    assert np.allclose(back.r, buf.r)
    mu, sd = back.state_stats()
    assert mu.shape == (6,) and sd.shape == (6,)
    assert back.reward_summary()['n'] == 30
    os.remove(path)


# ------------------------------------------------------------------ the optimiser
def t_td3bc_moves_toward_reward():
    """A toy one-step MDP: reward is highest at a known action.

    This does not test that DAT works -- it tests that the actor, critic, target networks
    and the BC term are wired together such that the policy moves toward reward at all.
    Getting a sign wrong here would produce a training run that looks healthy and learns
    the opposite of what it should.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d, adim, n = 6, 3, 2000
    p = PL.DATPlanner(d_model=d, action_dim=adim, n_prefix=1, hidden=32, layers=2)
    p.set_max_action(1.0)
    target = np.zeros(adim, dtype=np.float32)
    target[0] = 0.6

    S = rng.normal(size=(n, d)).astype(np.float32)
    U = np.clip(rng.normal(scale=0.5, size=(n, adim)), -1, 1).astype(np.float32)
    Rw = -((U - target) ** 2).sum(1, keepdims=True).astype(np.float32)
    buf = ReplayBuffer(S, U, Rw, np.zeros_like(S), np.ones((n, 1), dtype=np.float32))
    p.set_state_stats(*buf.state_stats())

    cfg = config.Defaults
    old_alpha, old_w = cfg.td3_alpha, cfg.td3_reward_weight
    cfg.td3_alpha, cfg.td3_reward_weight = 20.0, 1.0    # let Q outweigh BC on a toy task
    try:
        agent = TD3BC(p, cfg, device='cpu')
        for _ in range(1500):
            agent.train_step(buf.sample(128, 'cpu', rng))
    finally:
        cfg.td3_alpha, cfg.td3_reward_weight = old_alpha, old_w

    with torch.no_grad():
        pi = p.rl_action(torch.as_tensor(S[:256])).mean(0).numpy()
    moved = np.linalg.norm(pi - target)
    start = np.linalg.norm(target)                     # the zero-initialised policy
    assert moved < start, 'policy did not move toward reward (%.3f vs %.3f)' % (moved,
                                                                                start)
    print('        |pi - a*| %.3f, from %.3f at init' % (moved, start))


# ------------------------------------------------------------------ metrics
def _rec(uid, dca, discl, acts=('elicit', 'decide'), done=1):
    return {'uid': uid, 'arm': 'dat', 'done': done,
            'score': {'dca': dca, 'disclosure_rate': discl, 'cbar': dca, 'pbar': dca,
                      'close': dca, 'joint': False, 'schema_valid': True,
                      'hallucinated_credit': 0.0},
            'derived_acts': list(acts), 'turn_rewards': [0.0, dca],
            'terminal_reward': dca, 'action_norms': [1.0, 1.2],
            'residual_norms': [0.1, 0.1], 'action_cos': 0.5, 'planner_forwards': 2,
            'dialog': [{'role': 'Meeting', 'content': 'x', 'speaker': 'env'},
                       {'role': 'Chair', 'content': 'Could you confirm the exact limit?',
                        'speaker': 'sys'}],
            'revealed': ['PF1'] if discl else [], 'leaks': [], 'cover': 0.5,
            'turns': 2, 'n_calls': 10, 'prompt_chars': 100}


def t_summarise():
    recs = [_rec('a', 1.0, 1.0), _rec('b', 0.0, 0.0, done=-1)]
    s = M.summarise(recs)
    assert s['n'] == 2 and abs(s['dca'] - 0.5) < 1e-9
    assert abs(s['SR'] - 0.5) < 1e-9
    assert abs(s['elicit_rate'] - 0.5) < 1e-9        # 1 of 2 acts per record
    assert s['n_chair_turns'] == 2 and s['distinct_1'] > 0
    assert s['action_norm_mean'] > 0 and s['planner_forwards'] == 2


def t_paired_and_effect():
    a = [_rec('x', 1.0, 1.0), _rec('y', 0.5, 0.5)]
    b = [_rec('x', 0.0, 0.0), _rec('y', 0.5, 0.5)]
    r = M.paired(a, b, 'dca')
    assert r['win'] == 1 and r['tie'] == 1 and r['loss'] == 0
    assert abs(r['mean_delta'] - 0.5) < 1e-9
    assert M.paired(b, a, 'dca')['loss'] == 1
    eff = M.steering_effect(a, b)
    assert set(eff) == {'dca', 'disclosure_rate', 'cbar', 'elicit_rate', 'distinct_2'}


def t_registered_with_the_cross_arm_script():
    """analysis/compute_extended_metrics.py must know where DAT writes its records, or
    `cd analysis && python compute_extended_metrics.py` silently omits the arm."""
    path = os.path.join(os.path.dirname(paths.HERE), 'analysis',
                        'compute_extended_metrics.py')
    if not os.path.exists(path):
        print('        (skipped: analysis/ not present)')
        return
    text = open(path, encoding='utf-8').read()
    assert 'dat/logs/Record-dat-' in text, 'DAT is not registered in discover()'


if __name__ == '__main__':
    print('dat selftest\n')
    blocking = config.preflight()
    print('\nraw scenarios: %s' % paths.find_raw())
    print('model        : %s' % config.Defaults.model)
    print('L = %d prefix tokens, d\' = %d\n'
          % (config.Defaults.n_prefix, config.Defaults.action_dim))

    for name, fn in [
        ('split is complete and invariants hold', t_split),
        ('split matches the published one exactly', t_split_matches_published),
        ('verifier matches published scores', t_verifier_matches_published),
        ('chair prompt IS the no-planner control (every case)',
         t_chair_prompt_is_the_control),
        ('no private fact or oracle field reaches the chair', t_prompts_are_filtered),
        ('advisors see only their own facts', t_advisors_see_only_own_facts),
        ('settling turn carries the schema, other turns do not',
         t_settling_turn_carries_the_schema),
        ('planner shapes, and the RL head starts at zero', t_planner_shapes),
        ('planner save/load keeps stats and max_action', t_planner_roundtrip),
        ('state normalisation whitens the buffer states', t_state_normalisation),
        ('Appendix A PCA up-mapping builds', t_pca_up_map),
        ('RL residual is bounded by max_action', t_rl_head_is_bounded),
        ('prefix steering, state extraction and clone gradient',
         t_steering_mechanics),
        ('a whole episode runs end to end on a tiny model', t_episode_loop),
        ('reward is identical to EPO verifier PRM', t_reward_matches_epo),
        ('terminal reward stays in [-1, 1] and gates on validity',
         t_terminal_reward_bounds),
        ('record carries every field the metric scripts read', t_record_schema),
        ('transitions terminate correctly at the turn cap', t_transitions),
        ('replay buffer round-trips through npz', t_buffer_roundtrip),
        ('TD3+BC moves the policy toward reward on a toy MDP',
         t_td3bc_moves_toward_reward),
        ('summarise averages episodes and reports steering', t_summarise),
        ('paired sign test and steering effect', t_paired_and_effect),
        ('registered in analysis/compute_extended_metrics.py',
         t_registered_with_the_cross_arm_script),
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
