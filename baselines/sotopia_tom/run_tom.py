"""Run one or more prompting arms over the held-out scenarios.

No training. Every arm loads the same frozen Qwen2.5-7B-Instruct once and differs only in
how the chair's turn is prompted, so a difference in the numbers is attributable to the
strategy and nothing else.

    python run_tom.py --strategies stripped basic            # the one-hour probe
    python run_tom.py --strategies stripped basic cot tom_coach tom_belief
    python run_tom.py --compare                              # score what is on disk

Records match the other baselines' schema, so ppdpp_csa/compute_all_metrics.py runs on
them and every arm can be paired per scenario.
"""
import argparse
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
import metrics_tom as M                              # noqa: E402
import paths                                         # noqa: E402
import prompts_tom as P                              # noqa: E402
from env_tom import ToMEnv                           # noqa: E402


def record_path(strategy, split):
    return os.path.join(paths.LOGS, 'Record-tom-%s-%s.txt' % (strategy, split))


def load_records(path):
    import ast
    out = []
    if not os.path.exists(path):
        return out
    for blk in open(path, encoding='utf-8').read().split('\n\n'):
        blk = blk.strip()
        if blk:
            try:
                out.append(ast.literal_eval(blk))
            except Exception:                        # noqa: BLE001
                pass
    return out


def run_arm(env, cases, strategy, split, limit=0):
    out_path = record_path(strategy, split)
    recs, t0 = [], time.time()
    todo = cases[:limit] if limit else cases
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, case in enumerate(todo):
            env.reset(case, strategy=strategy)
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
                print('  [%s] %d/%d  DA %.3f IA %.3f EFF %.3f InfoMgmt3 %.3f  %.1f min'
                      % (strategy, i + 1, len(todo), s['DA'], s['IA'], s['EFF'],
                         s['InfoMgmt3'], (time.time() - t0) / 60), flush=True)
    return recs, out_path


def report(by_arm):
    order = [s for s in P.STRATEGIES if s in by_arm]
    print('\n' + '=' * 96)
    print('%-12s %5s %6s %6s %6s %10s %6s %6s %8s %7s %6s'
          % ('arm', 'n', 'DA', 'IA', 'EFF', 'InfoMgmt3', 'SR', 'dca', 'discl', 'calls',
             'leaks'))
    print('-' * 96)
    summ = {}
    for arm in order:
        s = M.summarise(by_arm[arm])
        summ[arm] = s
        print('%-12s %5d %6.3f %6.3f %6.3f %10.3f %6.3f %6.3f %8.3f %7.1f %6d'
              % (arm, s['n'], s['DA'], s['IA'], s['EFF'], s['InfoMgmt3'], s['SR'],
                 s['dca'], s['disclosure_rate'], s['n_calls'], s['leaks']))
    print('=' * 96)
    print('CPV: not applicable on CSA (no private channel, nothing to withhold).')
    print('InfoMgmt3 is a THREE-way mean and reads higher than the paper\'s four-way')
    print('InfoMgmt. Use it to rank these arms, not to compare against the paper.\n')

    base = 'stripped' if 'stripped' in by_arm else order[0]
    print('paired vs "%s", per scenario (two-sided sign test)' % base)
    print('%-12s %-11s %5s %5s %5s %9s' % ('arm', 'metric', 'win', 'tie', 'loss', 'p'))
    print('-' * 56)
    for arm in order:
        if arm == base:
            continue
        for key in ('DA', 'IA', 'EFF', 'InfoMgmt3'):
            r = M.paired(by_arm[arm], by_arm[base], key)
            pf = '<0.001' if r['p'] < 0.001 else ('%.3f' % r['p'])
            print('%-12s %-11s %5d %5d %5d %9s'
                  % (arm, key, r['win'], r['tie'], r['loss'], pf))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--strategies', nargs='+', default=['stripped', 'basic'],
                   choices=list(P.STRATEGIES))
    p.add_argument('--split', default=config.Defaults.eval_split,
                   choices=['test', 'valid', 'train'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--compare', action='store_true',
                   help='score records already on disk, run nothing')
    p.add_argument('--out', default=os.path.join(paths.LOGS, 'summary.json'))
    cli = p.parse_args()

    cfg = config.Defaults
    if cli.device:
        cfg.device = cli.device

    cases = data_csa.load(cli.split)
    by_arm = {}

    if cli.compare:
        for arm in P.STRATEGIES:
            recs = load_records(record_path(arm, cli.split))
            if recs:
                by_arm[arm] = recs
        if not by_arm:
            raise SystemExit('no records found in %s' % paths.LOGS)
    else:
        if compat.report():
            raise SystemExit('\nfix the blocking problems above, then re-run')
        torch.manual_seed(cfg.seed)
        print('scenarios: %d   arms: %s' % (len(cases), ', '.join(cli.strategies)))
        env = ToMEnv(cfg)                            # loaded ONCE, shared by every arm
        for arm in cli.strategies:
            print('\n--- %s ---' % arm, flush=True)
            recs, path = run_arm(env, cases, arm, cli.split, cli.limit)
            by_arm[arm] = recs
            print('  wrote %s' % path)

    summ = report(by_arm)
    with open(cli.out, 'w', encoding='utf-8') as f:
        json.dump({'split': cli.split, 'model': cfg.model, 'summary': summ}, f, indent=1)
    print('summary -> %s' % cli.out)


if __name__ == '__main__':
    main()
