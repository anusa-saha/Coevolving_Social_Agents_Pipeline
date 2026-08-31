"""Evaluate a chair policy on the held-out scenarios.

Emits one record per episode in the same shape the other baselines write, so the three
arms can be scored by one script and paired per scenario. Greedy decoding throughout, so
the number is reproducible.

    python evaluate_sr.py --adapter ckpt/grpo/grpo-rm-seed1/final --tag grpo-final
    python evaluate_sr.py --adapter "" --tag base        # untrained chair, the floor
"""
import argparse
import collections
import json
import os
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
import data_csa                                      # noqa: E402
import paths                                         # noqa: E402
import prompts_sr as P                               # noqa: E402
from env_sr import SREnv                             # noqa: E402
from models_sr import POLICY, PolicyView, SharedBackbone   # noqa: E402


def run_episode(env, case, policy):
    env.reset(case)
    env.greedy = True
    done, t = 0, 0
    while not done:
        if policy is None:
            utt = env.chair_say(temperature=0.0)
        else:
            msgs = env.chair_prompt()
            text = compat.render_chat(policy.tokenizer,
                                      P.to_chat(msgs, env.names[env.dm]))
            outs, _c, _p = policy.sample(text, n=1, max_new_tokens=env.chair_budget(),
                                         greedy=True)
            utt = outs[0]
        _c, done = env.step(utt)
        t += 1
    s = env.last_score or env._finalise()
    reward = s['dca']
    goal = 1 if (s['schema_valid'] and not env.leaks
                 and s['dca'] >= env.cfg.done_tau) else -1
    return env.record(reward, goal, t), s


def summarise(recs):
    def avg(fn):
        v = [fn(r) for r in recs if fn(r) is not None]
        return sum(v) / len(v) if v else float('nan')
    sc = lambda r: r.get('score') or {}              # noqa: E731
    return {
        'n': len(recs),
        'SR': avg(lambda r: 1.0 if r['done'] == 1 else 0.0),
        'dca': avg(lambda r: sc(r).get('dca')),
        'disclosure_rate': avg(lambda r: sc(r).get('disclosure_rate')),
        'any_reveal': sum(1 for r in recs if r['revealed']),
        'elicited_frac': avg(lambda r: (sum(1 for v in r['reveal_elicited'].values() if v)
                                        / max(1, len(r['reveal_elicited'])))
                             if r['reveal_elicited'] else 0.0),
        'cbar': avg(lambda r: sc(r).get('cbar')),
        'pbar': avg(lambda r: sc(r).get('pbar')),
        'close': avg(lambda r: sc(r).get('close')),
        'cover': avg(lambda r: r.get('cover')),
        'schema_valid': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
        'halluc': avg(lambda r: sc(r).get('hallucinated_credit')),
        'leaks': sum(1 for r in recs if r['leaks']),
        'turns': avg(lambda r: r['turns']),
        'n_calls': avg(lambda r: r['n_calls']),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', default='', help='"" evaluates the untrained chair')
    p.add_argument('--split', default=config.Defaults.eval_split,
                   choices=['test', 'valid', 'train'])
    p.add_argument('--tag', default='eval')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--agent_device', default=None)
    p.add_argument('--device', default=None)
    cli = p.parse_args()

    cfg = config.Defaults
    if cli.agent_device:
        cfg.agent_device = cli.agent_device
    if cli.device:
        cfg.agent_device = cli.device

    cases = data_csa.load(cli.split)
    if cli.limit:
        cases = cases[:cli.limit]
    bb = SharedBackbone(cfg, adapters=[POLICY])
    env = SREnv(cfg, backbone=bb)
    # no adapter -> the chair speaks with the BASE weights, which is the floor
    policy = PolicyView(bb, adapter_dir=cli.adapter) if cli.adapter else None
    if policy is not None:
        policy.eval()

    out_path = os.path.join(paths.LOGS, 'Record-%s-%s.txt' % (cli.tag, cli.split))
    recs, t0 = [], time.time()
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            rec, _s = run_episode(env, case, policy)
            recs.append(rec)
            f.write('%s\n\n' % str(rec))
            if (i + 1) % 10 == 0:
                print('  %d/%d  %.1f min' % (i + 1, len(cases), (time.time() - t0) / 60),
                      flush=True)

    summ = summarise(recs)
    summ['tag'] = cli.tag
    summ['split'] = cli.split
    summ['adapter'] = cli.adapter or None
    with open(os.path.join(paths.LOGS, 'summary-%s-%s.json' % (cli.tag, cli.split)),
              'w', encoding='utf-8') as f:
        json.dump({'summary': summ,
                   'by_domain': _by(recs, 'domain'),
                   'by_num_agents': _by(recs, 'num_agents')}, f, indent=1)
    print('\nrecords -> %s' % out_path)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in summ.items()}, indent=1))


def _by(recs, key):
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r.get(key)].append(r)
    return {str(k): summarise(v) for k, v in sorted(groups.items(), key=lambda x: str(x[0]))}


if __name__ == '__main__':
    main()
