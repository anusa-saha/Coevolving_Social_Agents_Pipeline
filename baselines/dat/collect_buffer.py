"""STAGE 2a -- the offline replay buffer (Section 6.1).

Roll the self-cloned policy over training scenarios with Gaussian noise added in the
64-dimensional action space, and keep (s, u, r, s', done) for every chair turn. That is
the entire interaction budget: TD3+BC never touches the environment again, which is why
the paper can say the RL step finishes in ten minutes.

    python collect_buffer.py --episodes 400 --sigma 0.25

Two knobs matter and both follow the paper. The exploration noise is N(0, 0.25) in the
action space -- not in the prefix space, which is where it would be 7168-dimensional and
useless. And generation is greedy while collecting: "we set the LM agent's temperature to
0 ... for a less noisy signal", so the only stochasticity in a transition is the action
perturbation whose effect the critic is being asked to learn.

Budget note, stated rather than buried: the paper collects 10,000 episodes (~30,000
transitions) and Figure 4 shows the attack success rate still climbing at 80,000. The
default here is 400 episodes, ~1,200 transitions. This is the most under-resourced number
in the arm, and raising it is the first thing to do with spare GPU time.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import paths                                         # noqa: E402
import config                                        # noqa: E402
from env_dat import DATEnv                           # noqa: E402
from planner import DATPlanner                       # noqa: E402
from steering import SteeredLM                       # noqa: E402
from td3bc import ReplayBuffer                       # noqa: E402

BUFFER = os.path.join(paths.DATA, 'buffer.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=config.Defaults.buffer_episodes)
    p.add_argument('--sigma', type=float, default=config.Defaults.explore_sigma)
    p.add_argument('--planner', default=os.path.join(paths.CKPT, 'selfclone.pt'))
    p.add_argument('--around', default='selfclone', choices=['selfclone', 'dat'],
                   help='perturb the self-cloned policy (the paper) or the trained one')
    p.add_argument('--out', default=BUFFER)
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--w_outcome', type=float, default=config.Defaults.w_outcome)
    p.add_argument('--sample', action='store_true',
                   help='sample instead of greedy while collecting; the paper does not')
    p.add_argument('--device', default=config.Defaults.device)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    cli = p.parse_args()

    cfg = config.Defaults
    cfg.device = cli.device
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    gen = torch.Generator().manual_seed(cli.seed)

    if config.preflight():
        raise SystemExit('\nfix the blocking problems above, then re-run')
    if not os.path.exists(cli.planner):
        raise SystemExit('no self-cloned planner at %s; run selfclone.py first'
                         % cli.planner)

    lm = SteeredLM(cfg)
    planner = DATPlanner.load(cli.planner, map_location=lm.device).to(lm.device)
    env = DATEnv(cfg, lm=lm, planner=planner)
    env.greedy = not cli.sample and cfg.collect_greedy

    cases = config.load_csa(cli.split)
    rows, episodes = [], []
    t0 = time.time()
    log = open(os.path.join(paths.LOGS, 'dat-buffer-episodes.jsonl'), 'a',
               encoding='utf-8')
    for i in range(cli.episodes):
        case = cases[i % len(cases)]
        env.reset(case, arm=cli.around)
        done, turns = 0, 0
        while not done:
            _c, done = env.step(sigma=cli.sigma, generator=gen)
            turns += 1
        tr = env.transitions(w_outcome=cli.w_outcome)
        rows += tr
        s = env.last_score
        meta = {'uid': case['uid'], 'turns': turns, 'dca': s['dca'],
                'disclosure': s['disclosure_rate'], 'schema_valid': s['schema_valid'],
                'leaks': len(env.leaks), 'terminal': env.terminal_reward(),
                'rewards': [t['r'] for t in tr]}
        episodes.append(meta)
        log.write(json.dumps(meta) + '\n')
        log.flush()
        if (i + 1) % 10 == 0:
            nz = sum(1 for t in rows if t['r'] != 0)
            print('  %d/%d episodes, %d transitions, %d with reward, %.1f min'
                  % (i + 1, cli.episodes, len(rows), nz, (time.time() - t0) / 60),
                  flush=True)
    log.close()

    buf = ReplayBuffer.from_transitions(rows)
    buf.save(cli.out)
    summary = {'episodes': cli.episodes, 'split': cli.split, 'sigma': cli.sigma,
               'around': cli.around, 'greedy': env.greedy,
               'transitions': len(buf), 'reward': buf.reward_summary(),
               'mean_dca': float(np.mean([e['dca'] for e in episodes])),
               'mean_disclosure': float(np.mean([e['disclosure'] for e in episodes])),
               'any_reveal': int(sum(1 for e in episodes if e['disclosure'] > 0)),
               'minutes': round((time.time() - t0) / 60, 1)}
    with open(os.path.join(paths.LOGS, 'dat-buffer-summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print('\nbuffer -> %s' % cli.out)
    print(json.dumps(summary, indent=1))
    if summary['reward']['nonzero_frac'] < 0.05:
        print('\nWARNING: fewer than 5%% of transitions carry any reward. TD3+BC will')
        print('fit a critic that predicts zero everywhere. Raise --episodes, or check')
        print('that the chair is eliciting at all (mean_disclosure above).')


if __name__ == '__main__':
    main()
