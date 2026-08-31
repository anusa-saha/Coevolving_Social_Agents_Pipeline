"""STAGE 1 -- collect self-play episodes.

Sotopia-RL clones GPT-4o self-play and filters with a GPT-4 goal-completion rating. Here
the chair is the same frozen Qwen agent that plays the advisors, and the filter is the
dataset's executable checks -- weaker demonstrations, far better-grounded selection.

The chair speaks UNPROMPTED. No act, no strategy, no planner: whatever it does is what
gets cloned, and what the attributor later scores.

k rollouts per scenario are ranked WITHIN the scenario and the top few kept, so hard
scenarios still contribute rather than the filter quietly selecting only easy ones.

    python collect_episodes.py --split train --k 6 --keep 2
"""
import argparse
import collections
import json
import os
import statistics as st
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import attribution                                   # noqa: E402
import config                                        # noqa: E402
import data_csa                                      # noqa: E402
import paths                                         # noqa: E402
from env_sr import SREnv                             # noqa: E402


def rollout(env, case):
    env.reset(case)
    done = 0
    while not done:
        utt = env.chair_say(temperature=env.cfg.rollout_temperature)
        _c, done = env.step(utt)
    return env.episode(), env.last_score


def rank_key(s):
    """Rank within a scenario. cbar first because it is the densest competence signal,
    dca as the tiebreak because it is what the benchmark is actually about."""
    return (s['dca'], s['cbar'], s['disclosure_rate'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--k', type=int, default=config.Defaults.rollout_k)
    p.add_argument('--keep', type=int, default=config.Defaults.rollout_keep)
    p.add_argument('--out', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--agent_device', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--restart', action='store_true')
    cli = p.parse_args()

    cfg = config.Defaults
    if cli.agent_device or cli.device:
        cfg.agent_device = cli.agent_device or cli.device
    if cli.temperature is not None:
        cfg.rollout_temperature = cli.temperature

    out_path = cli.out or os.path.join(paths.DATA, 'episodes-%s.jsonl' % cli.split)
    rows = data_csa.load(cli.split)
    if cli.limit:
        rows = rows[:cli.limit]

    done_uids = set()
    if os.path.exists(out_path) and not cli.restart:
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_uids.add(json.loads(line)['uid'])
                except Exception:                    # noqa: BLE001
                    pass
        if done_uids:
            print('resuming: %d scenarios already collected' % len(done_uids), flush=True)

    env = SREnv(cfg)
    stats = collections.Counter()
    kept_scores, t0 = [], time.time()

    with open(out_path, 'a' if done_uids else 'w', encoding='utf-8') as f:
        for i, case in enumerate(rows):
            if case['uid'] in done_uids:
                continue
            cands = []
            for _j in range(cli.k):
                ep, s = rollout(env, case)
                ep['score'] = {k: v for k, v in s.items()
                               if k not in ('content', 'provenance', 'settlement_resolved')}
                ep['rank'] = rank_key(s)
                cands.append(ep)
            cands.sort(key=lambda e: e['rank'], reverse=True)
            spread = max(c['rank'][0] for c in cands) - min(c['rank'][0] for c in cands)
            stats['flat' if spread == 0 else 'varied'] += 1
            for c in cands[:cli.keep]:
                c.pop('rank', None)
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
                kept_scores.append(c['score']['dca'])
            f.flush()
            if (i + 1) % 5 == 0:
                print('  %3d/%d  kept dca mean %.3f  %.1f min elapsed'
                      % (i + 1, len(rows), sum(kept_scores) / max(1, len(kept_scores)),
                         (time.time() - t0) / 60), flush=True)

    print('\nwrote %s' % out_path)
    if kept_scores:
        print('  kept episodes: %d   dca mean %.3f median %.3f max %.3f'
              % (len(kept_scores), sum(kept_scores) / len(kept_scores),
                 st.median(kept_scores), max(kept_scores)))
    print('  scenarios where all %d rollouts scored the same: %d/%d'
          % (cli.k, stats['flat'], stats['flat'] + stats['varied']))
    if stats['flat'] > stats['varied']:
        print('  WARNING: the ranking filter has little to work with. Raise --temperature '
              'or --k, or accept that selection is mostly noise.')

    # a first look at what the attributor will see
    eps = [json.loads(l) for l in open(out_path, encoding='utf-8')]
    cases = data_csa.case_index()
    raws = []
    for e in eps:
        c = cases.get(e['uid'])
        if c:
            raw, _d = attribution.attribute(c, e['dialog'], e.get('settlement'))
            raws.append(raw)
    if raws:
        norm = attribution.Normaliser().fit(raws)
        rt = [x for raw in raws for x in norm.apply(raw)]
        print('\nprojected reward labels: %s' % attribution.describe(rt))
        for d in attribution.DIMS:
            v = [x for raw in raws for x in raw[d]]
            print('  %-6s %s' % (d, attribution.describe(v)))


if __name__ == '__main__':
    main()
