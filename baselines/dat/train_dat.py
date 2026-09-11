"""STAGE 2b -- TD3+BC on the offline buffer (Section 5.2).

No language model is loaded. Everything this reads is already in buffer.npz, which is why
the paper can call the RL step cheap: minutes on one card, and it will run on a CPU here
if you are only checking the plumbing.

    python train_dat.py --steps 4000

What to look at when it finishes:

  policy_shift.mean_l2_to_buffer   0 means TD3+BC gave back the self-cloned policy and
                                   the `dat` arm will be the `selfclone` arm with extra
                                   steps. Very large means the actor has walked out of
                                   the data, where an offline critic's Q values are
                                   fiction.
  q_mean vs the buffer's rewards   a critic predicting far outside the reward range has
                                   diverged, whatever the loss curve says.
  bc_loss                          how hard the BC term is fighting the Q term. If it is
                                   flat at zero the actor never moved.
"""
import argparse
import json
import os
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
from planner import DATPlanner                       # noqa: E402
from td3bc import ReplayBuffer, TD3BC                # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', default=os.path.join(paths.DATA, 'buffer.npz'))
    p.add_argument('--planner', default=os.path.join(paths.CKPT, 'selfclone.pt'))
    p.add_argument('--out', default=os.path.join(paths.CKPT, 'dat.pt'))
    p.add_argument('--steps', type=int, default=config.Defaults.td3_steps)
    p.add_argument('--batch', type=int, default=config.Defaults.td3_batch)
    p.add_argument('--lr', type=float, default=config.Defaults.td3_lr)
    p.add_argument('--alpha', type=float, default=config.Defaults.td3_alpha)
    p.add_argument('--gamma', type=float, default=config.Defaults.td3_gamma)
    p.add_argument('--reward_weight', type=float,
                   default=config.Defaults.td3_reward_weight)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    cli = p.parse_args()

    cfg = config.Defaults
    cfg.td3_lr, cfg.td3_alpha = cli.lr, cli.alpha
    cfg.td3_gamma, cfg.td3_reward_weight = cli.gamma, cli.reward_weight
    torch.manual_seed(cli.seed)
    rng = np.random.default_rng(cli.seed)

    if not os.path.exists(cli.buffer):
        raise SystemExit('no buffer at %s; run collect_buffer.py first' % cli.buffer)
    if not os.path.exists(cli.planner):
        raise SystemExit('no self-cloned planner at %s; run selfclone.py first'
                         % cli.planner)

    buf = ReplayBuffer.load(cli.buffer)
    planner = DATPlanner.load(cli.planner, map_location=cli.device)
    # The normalisation Appendix B's residual head reads is a property of the states the
    # policy will actually see, so it comes from the buffer rather than from the
    # self-clone corpus. Setting it here also means a re-collected buffer re-normalises.
    mu, sd = buf.state_stats()
    planner.set_state_stats(mu, sd)

    print('buffer     : %d transitions from %s' % (len(buf), cli.buffer))
    print('reward     : %s' % json.dumps(buf.reward_summary()))
    print('planner    : %s' % json.dumps(planner.meta()))
    print('max_action : %.4f' % float(planner.max_action))

    agent = TD3BC(planner, cfg, device=cli.device)
    hist_path = os.path.join(paths.LOGS, 'dat-td3bc-history.jsonl')
    hist = open(hist_path, 'a', encoding='utf-8')
    t0 = time.time()
    last = {}
    for step in range(1, cli.steps + 1):
        out = agent.train_step(buf.sample(cli.batch, cli.device, rng))
        out.update({'step': step, 'elapsed_s': round(time.time() - t0, 1)})
        hist.write(json.dumps(out) + '\n')
        last = out
        if step % max(1, cli.steps // 20) == 0:
            hist.flush()
            print('  step %5d  critic %.4f  actor %s  bc %s  q %.3f  %.1f min'
                  % (step, out['critic_loss'],
                     ('%.4f' % out['actor_loss']) if 'actor_loss' in out else '-',
                     ('%.4f' % out['bc_loss']) if 'bc_loss' in out else '-',
                     out['q_mean'], (time.time() - t0) / 60), flush=True)
    hist.close()

    shift = agent.policy_shift(buf)
    print('\npolicy shift: %s' % json.dumps({k: round(v, 4) for k, v in shift.items()}))
    if shift['mean_action_norm'] < 1e-4:
        print('the RL head is still at zero -- the `dat` arm will be identical to')
        print('`selfclone`. Check that the buffer carries a non-zero reward anywhere.')

    agent.planner.save(cli.out)
    with open(os.path.join(paths.LOGS, 'dat-td3bc-summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'buffer': cli.buffer, 'transitions': len(buf),
                   'reward': buf.reward_summary(), 'args': vars(cli),
                   'final': last, 'policy_shift': shift,
                   'minutes': round((time.time() - t0) / 60, 1)}, f, indent=1)
    print('planner -> %s' % cli.out)


if __name__ == '__main__':
    main()
