"""Evaluate the three DAT conditions over the held-out scenarios.

    python run_dat.py --arms unsteered selfclone dat      # the paper's three rows
    python run_dat.py --compare                           # score what is on disk

The three arms load ONE planner checkpoint and ONE language model. `selfclone` is not a
different checkpoint from `dat` -- it is the same one with the RL head forced to zero, so
the two arms provably differ by nothing except pi_phi_rl. Loading two files would let them
differ by a stale pi_phi as well, and the comparison would stop meaning what it says.

Records are written in the shared schema, so

    cd analysis && python compute_extended_metrics.py

picks the arm up from dat/logs/ with no argument, and

    python ppdpp/compute_all_metrics.py --records dat/logs --glob 'Record-dat-*.txt'

runs the A-D/H/I catalogue over it.
"""
import argparse
import ast
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

import paths                                         # noqa: E402
import config                                        # noqa: E402
import metrics_dat as M                              # noqa: E402
from env_dat import ARMS, DATEnv                     # noqa: E402
from planner import DATPlanner                       # noqa: E402
from steering import SteeredLM                       # noqa: E402


def record_path(arm, split):
    return os.path.join(paths.LOGS, 'Record-dat-%s-%s.txt' % (arm, split))


def load_records(path):
    out = []
    if not os.path.exists(path):
        return out
    text = open(path, encoding='utf-8', errors='replace').read()
    buf = ''
    for part in text.split('\n\n'):
        buf = (buf + '\n\n' + part) if buf else part
        s = buf.strip()
        if not s.startswith('{'):
            buf = ''
            continue
        try:
            out.append(ast.literal_eval(s))
            buf = ''
        except Exception:                            # noqa: BLE001
            continue
    return out


def run_arm(env, cases, arm, split, limit=0):
    out_path = record_path(arm, split)
    recs, t0 = [], time.time()
    todo = cases[:limit] if limit else cases
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, case in enumerate(todo):
            env.reset(case, arm=arm)
            done, t = 0, 0
            while not done:
                _c, done = env.step()
                t += 1
            rec = env.record(t)
            recs.append(rec)
            f.write('%s\n\n' % str(rec))
            f.flush()
            if (i + 1) % 5 == 0:
                s = M.summarise(recs)
                print('  [%s] %d/%d  dca %.3f  discl %.3f  SR %.3f  elicit %.2f  %.1f min'
                      % (arm, i + 1, len(todo), s['dca'], s['disclosure_rate'], s['SR'],
                         s['elicit_rate'], (time.time() - t0) / 60), flush=True)
    return recs, out_path


def report(by_arm):
    order = [a for a in ARMS if a in by_arm]
    summ = {a: M.summarise(by_arm[a]) for a in order}

    print('\n' + '=' * 104)
    print('%-11s %5s %6s %8s %6s %6s %6s %8s %7s %6s %7s'
          % ('arm', 'n', 'dca', 'discl', 'SR', 'cbar', 'pbar', 'schema', 'elicit',
             'leaks', 'calls'))
    print('-' * 104)
    for a in order:
        s = summ[a]
        print('%-11s %5d %6.3f %8.3f %6.3f %6.3f %6.3f %8.3f %7.3f %6d %7.1f'
              % (a, s['n'], s['dca'], s['disclosure_rate'], s['SR'], s['cbar'],
                 s['pbar'], s['schema_valid'], s['elicit_rate'], s['leaks'],
                 s['n_calls']))

    print('\nsteering  (empty for `unsteered`: there is no action to report)')
    print('%-11s %10s %10s %12s %12s %10s'
          % ('arm', '|a| mean', '|a| sd', '|residual|', 'cos(a_t,a_t+1)', 'fwd/ep'))
    for a in order:
        s = summ[a]
        print('%-11s %10s %10s %12s %12s %10s'
              % (a, _f(s['action_norm_mean']), _f(s['action_norm_sd']),
                 _f(s['residual_norm_mean']), _f(s['action_cos_mean']),
                 _f(s['planner_forwards'], 1)))

    print('\nlanguage health  (the claim DAT makes about freezing the LM)')
    print('%-11s %8s %10s %10s %10s'
          % ('arm', 'turns', 'distinct-1', 'distinct-2', 'len mean'))
    for a in order:
        s = summ[a]
        print('%-11s %8d %10s %10s %10s'
              % (a, s['n_chair_turns'], _f(s['distinct_1'], 3), _f(s['distinct_2'], 3),
                 _f(s['utterance_len_mean'], 1)))

    base = 'unsteered' if 'unsteered' in by_arm else order[0]
    print('\npaired vs "%s", per scenario (two-sided sign test)' % base)
    print('%-11s %-16s %5s %5s %5s %11s %9s'
          % ('arm', 'metric', 'win', 'tie', 'loss', 'mean delta', 'p'))
    print('-' * 72)
    effects = {}
    for a in order:
        if a == base:
            continue
        eff = M.steering_effect(by_arm[a], by_arm[base])
        effects[a] = eff
        for key, r in eff.items():
            print('%-11s %-16s %5d %5d %5d %11s %9s'
                  % (a, key, r['win'], r['tie'], r['loss'], _f(r['mean_delta'], 3),
                     '<0.001' if r['p'] < 0.001 else _f(r['p'], 3)))
    print('=' * 104)
    print('Read `selfclone` vs `unsteered` first. Stage 1 is supposed to PRESERVE')
    print('behaviour, so a large gap there is a bug in the clone, not a result -- and it')
    print('would make the `dat` row uninterpretable.')
    return summ, effects


def _f(x, d=4):
    if x is None:
        return '-'
    if isinstance(x, float):
        return '-' if x != x else ('%.' + str(d) + 'f') % x
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    p.add_argument('--split', default=config.Defaults.eval_split,
                   choices=['test', 'valid', 'train'])
    p.add_argument('--planner', default='',
                   help='default: ckpt/dat.pt if it exists, else ckpt/selfclone.pt')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default=config.Defaults.device)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    p.add_argument('--compare', action='store_true',
                   help='score records already on disk, run nothing')
    p.add_argument('--out', default=os.path.join(paths.LOGS, 'summary.json'))
    cli = p.parse_args()

    cfg = config.Defaults
    cfg.device = cli.device
    by_arm = {}

    if cli.compare:
        for arm in ARMS:
            recs = load_records(record_path(arm, cli.split))
            if recs:
                by_arm[arm] = recs
        if not by_arm:
            raise SystemExit('no records found in %s' % paths.LOGS)
    else:
        if config.preflight():
            raise SystemExit('\nfix the blocking problems above, then re-run')
        torch.manual_seed(cli.seed)
        cases = config.load_csa(cli.split)
        lm = SteeredLM(cfg)

        planner = None
        if set(cli.arms) - {'unsteered'}:
            ckpt = cli.planner or next(
                (c for c in (os.path.join(paths.CKPT, 'dat.pt'),
                             os.path.join(paths.CKPT, 'selfclone.pt'))
                 if os.path.exists(c)), '')
            if not ckpt:
                raise SystemExit('no planner checkpoint; run selfclone.py (and '
                                 'train_dat.py) first, or evaluate --arms unsteered')
            planner = DATPlanner.load(ckpt, map_location=lm.device).to(lm.device)
            print('planner: %s  %s' % (ckpt, json.dumps(planner.meta())))

        env = DATEnv(cfg, lm=lm, planner=planner)    # loaded ONCE, shared by every arm
        print('scenarios: %d   arms: %s' % (len(cases), ', '.join(cli.arms)))
        for arm in cli.arms:
            print('\n--- %s ---' % arm, flush=True)
            recs, path = run_arm(env, cases, arm, cli.split, cli.limit)
            by_arm[arm] = recs
            print('  wrote %s' % path)

    summ, effects = report(by_arm)
    with open(cli.out, 'w', encoding='utf-8') as f:
        json.dump({'split': cli.split, 'model': cfg.model, 'summary': summ,
                   'paired': effects}, f, indent=1, default=str)
    print('summary -> %s' % cli.out)


if __name__ == '__main__':
    main()
