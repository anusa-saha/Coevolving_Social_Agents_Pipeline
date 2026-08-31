"""STAGE 3 -- multi-turn RL for LLM_s, and evaluation.

The loop differs from ppdpp_csa/run.py in one structural way: rewards are not available
during the episode. EPO's credit assignment reads the FINISHED trajectory, so an episode
runs to completion, the PRM labels it, returns are computed, and only then does a
gradient flow. env.step() therefore returns no reward at all.

Advantage is group-relative by default: k rollouts of the SAME scenario, centred on the
group mean. Vanilla EPO uses A_t = R_t / max|R_{1:T}|, which is a scaling and never
subtracts a baseline -- fine at 2050 episodes, but at 700 on a task where the PPDPP
baseline recorded disclosure 0.0 on 91% of steps, most episodes would contribute no
gradient at all. --advantage maxabs restores the paper's rule for the ablation.

    python run_epo.py --episodes 700 --adapter ckpt/sft --eval_every 175
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import compat                                        # noqa: E402
import config                                        # noqa: E402
import prm as prm_mod                                # noqa: E402
import prompt_epo as pe                              # noqa: E402
from detectors import assert_identical_to_ppdpp      # noqa: E402
from env_epo import EPOEnv                           # noqa: E402
from strategist import EPOStrategist                 # noqa: E402


def rollout(env, strat, case, is_test=False):
    """One episode. Returns (samples, trace, terminal, done, turns, record)."""
    env.reset(case)
    samples, prior = [], []
    done, t = 0, 0
    for t in range(env.max_turn):
        text, sample = strat.act(env.case, env.conversation, prior, is_test=is_test)
        tau, _sigma, _ok = pe.parse_strategy(text)
        prior.append(tau)
        samples.append(sample)
        _conv, done = env.step(text)
        if done:
            break
    turns = t + 1
    terminal = env.terminal_reward()
    return samples, env.trace(), terminal, done, turns, env.record(terminal, done, turns)


def returns_of(rewards, gamma):
    R, out = 0.0, []
    for r in rewards[::-1]:
        R = r + gamma * R
        out.insert(0, R)
    return out


def advantages(group_returns, mode):
    """group_returns: list (one per rollout) of per-turn returns."""
    if mode == 'maxabs':                              # vanilla EPO
        out = []
        for rets in group_returns:
            m = max((abs(x) for x in rets), default=0.0) or 1.0
            out.append([x / m for x in rets])
        return out
    flat = [x for rets in group_returns for x in rets]
    if not flat:
        return [[] for _ in group_returns]
    mu = sum(flat) / len(flat)
    sd = statistics.pstdev(flat) if len(flat) > 1 else 0.0
    sd = sd or 1.0
    return [[(x - mu) / sd for x in rets] for rets in group_returns]


def evaluate(env, strat, cases, out_path, tag):
    env.mode = 'test'
    strat.policy.eval()
    recs = []
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            _s, _tr, terminal, done, turns, rec = rollout(env, strat, case, is_test=True)
            recs.append(rec)
            f.write('%s\n\n' % str(rec))
            if (i + 1) % 10 == 0:
                print('  eval %d/%d' % (i + 1, len(cases)), flush=True)
    env.mode = 'train'
    strat.policy.train()

    def avg(fn):
        v = [fn(r) for r in recs if fn(r) is not None]
        return sum(v) / len(v) if v else float('nan')

    sc = lambda r: (r.get('score') or {})            # noqa: E731
    summary = {
        'tag': tag, 'n': len(recs),
        'SR': avg(lambda r: 1.0 if r['done'] == 1 else 0.0),
        'dca': avg(lambda r: sc(r).get('dca')),
        'disclosure_rate': avg(lambda r: sc(r).get('disclosure_rate')),
        'any_reveal': sum(1 for r in recs if r['revealed']),
        'cbar': avg(lambda r: sc(r).get('cbar')),
        'pbar': avg(lambda r: sc(r).get('pbar')),
        'schema_valid': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
        'leaks': sum(1 for r in recs if r['leaks']),
        'reward': avg(lambda r: r['reward']),
        'turns': avg(lambda r: r['turns']),
        'n_calls': avg(lambda r: r['n_calls']),
        'tag_misses': sum(r.get('tag_misses', 0) for r in recs),
        'act_dist': dict(collections.Counter(a for r in recs for a in r['act_history'])),
    }
    print('[eval %s] %s' % (tag, json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in summary.items() if k != 'act_dist'})), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=config.Defaults.total_episodes)
    p.add_argument('--group_k', type=int, default=config.Defaults.group_k)
    p.add_argument('--adapter', default=os.path.join(config.CKPT, 'sft'),
                   help='SFT warm-start adapter; pass "" for the pure-RL ablation')
    p.add_argument('--out', default=os.path.join(config.CKPT, 'rl'))
    p.add_argument('--prm', default='verifier', choices=['verifier', 'judge'])
    p.add_argument('--prm_mode', default='binary', choices=['binary', 'graded'])
    p.add_argument('--advantage', default='group', choices=['group', 'maxabs'])
    p.add_argument('--w_outcome', type=float, default=1.0,
                   help='weight on the terminal outcome added at t=T; 0 = PRM only')
    p.add_argument('--lr', type=float, default=config.Defaults.lr)
    p.add_argument('--kl_beta', type=float, default=config.Defaults.kl_beta)
    p.add_argument('--tag_weight', type=float, default=config.Defaults.tag_weight)
    p.add_argument('--episodes_per_update', type=int,
                   default=config.Defaults.episodes_per_update)
    p.add_argument('--eval_every', type=int, default=175)
    p.add_argument('--eval_split', default='test', choices=['test', 'valid'])
    p.add_argument('--agent_device', default=config.Defaults.agent_device)
    p.add_argument('--strategist_device', default=config.Defaults.strategist_device)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    p.add_argument('--dry_run', type=int, default=0,
                   help='run N episodes with no gradient, to check the plumbing')
    cli = p.parse_args()

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    cfg = config.Defaults
    cfg.lr, cfg.kl_beta, cfg.tag_weight = cli.lr, cli.kl_beta, cli.tag_weight
    cfg.episodes_per_update, cfg.prm_mode = cli.episodes_per_update, cli.prm_mode
    cfg.agent_device, cfg.strategist_device = cli.agent_device, cli.strategist_device

    # Preflight, before anything slow: an unusable environment should fail in seconds,
    # not after a 15GB checkpoint download.
    if compat.report():
        raise SystemExit('\nfix the blocking problems above, then re-run')
    assert_identical_to_ppdpp()                      # disclosure detector drift check

    data = config.load_csa()
    train_cases, eval_cases = data['train'], data[cli.eval_split]
    print('train scenarios %d   eval scenarios %d' % (len(train_cases), len(eval_cases)))

    env = EPOEnv(cfg, train_cases, mode='train')
    strat = EPOStrategist(cfg, adapter_dir=(cli.adapter or None))
    scorer = prm_mod.build(cli.prm, cfg)

    n_groups = max(1, cli.episodes // max(1, cli.group_k))
    strat.set_schedule(max(1, (n_groups * cli.group_k) // cfg.episodes_per_update))

    run_tag = 'epo-%s-%s-%s-seed%d' % (cli.prm, cli.prm_mode, cli.advantage, cli.seed)
    hist_path = os.path.join(config.LOGS, '%s-history.jsonl' % run_tag)
    hist = open(hist_path, 'a', encoding='utf-8')

    if cli.dry_run:
        for i in range(cli.dry_run):
            case = random.choice(train_cases)
            _s, tr, term, done, turns, rec = rollout(env, strat, case)
            print('[dry %d] %s turns=%d done=%d dca=%.3f reveal=%s r_t=%s term=%.3f'
                  % (i, case['uid'], turns, done, tr.dca, sorted(env.revealed),
                     scorer(tr), term), flush=True)
        return

    summaries = [evaluate(env, strat, eval_cases,
                          os.path.join(config.LOGS, 'Record-%s-ep0.txt' % run_tag), 'ep0')]

    episodes = 0
    t0 = time.time()
    for g in range(n_groups):
        case = random.choice(train_cases)
        group_returns, group_samples, group_meta = [], [], []
        for _k in range(cli.group_k):
            samples, tr, terminal, done, turns, rec = rollout(env, strat, case)
            r = scorer(tr)
            if len(r) < turns:
                r = r + [0.0] * (turns - len(r))
            r = r[:turns]
            if cli.w_outcome and r:
                r[-1] += cli.w_outcome * terminal
            group_returns.append(returns_of(r, cfg.gamma))
            group_samples.append(samples[:turns])
            group_meta.append({'uid': case['uid'], 'done': done, 'turns': turns,
                               'dca': tr.dca, 'reveal': sorted(env.revealed),
                               'elicited': sum(1 for v in tr.reveal_elicited.values() if v),
                               'leaks': len(tr.leaks), 'r': r, 'terminal': terminal,
                               'acts': tr.acts, 'tag_misses': env.tag_misses})
            episodes += 1

        advs = advantages(group_returns, cli.advantage)
        loss = 0.0
        for samples, adv in zip(group_samples, advs):
            loss += strat.accumulate(samples, adv)
        gn = strat.maybe_step()

        rec = {'group': g, 'episodes': episodes, 'uid': case['uid'], 'loss': loss,
               'grad_norm': gn, 'elapsed_s': round(time.time() - t0, 1),
               'mean_terminal': sum(m['terminal'] for m in group_meta) / len(group_meta),
               'mean_reward_mass': sum(sum(m['r']) for m in group_meta) / len(group_meta),
               'any_reveal': sum(1 for m in group_meta if m['reveal']),
               'meta': group_meta}
        hist.write(json.dumps(rec, ensure_ascii=False) + '\n')
        hist.flush()
        if (g + 1) % 5 == 0:
            print('group %3d  ep %3d  loss %.4f  term %.3f  rewardmass %.3f  reveal %d/%d'
                  % (g, episodes, loss, rec['mean_terminal'], rec['mean_reward_mass'],
                     rec['any_reveal'], cli.group_k), flush=True)

        if cli.eval_every and episodes % cli.eval_every < cli.group_k and g:
            tag = 'ep%d' % episodes
            summaries.append(evaluate(
                env, strat, eval_cases,
                os.path.join(config.LOGS, 'Record-%s-%s.txt' % (run_tag, tag)), tag))
            strat.save(os.path.join(cli.out, run_tag, tag))

    strat.maybe_step(force=True)
    summaries.append(evaluate(
        env, strat, eval_cases,
        os.path.join(config.LOGS, 'Record-%s-final.txt' % run_tag), 'final'))
    strat.save(os.path.join(cli.out, run_tag, 'final'))

    with open(os.path.join(config.LOGS, '%s-summary.json' % run_tag), 'w',
              encoding='utf-8') as f:
        json.dump({'run': run_tag, 'args': vars(cli), 'summaries': summaries}, f, indent=1)
    hist.close()
    print('\ndone: %d episodes in %.1f min' % (episodes, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
