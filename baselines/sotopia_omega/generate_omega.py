"""STAGE 1 -- generate the corpus.

Omega's contribution is the intervention, not the dialogue: let the expert talk, detect
that it has stalled, switch it into a structured mode, keep what results. The corpus is
then filtered and fine-tuned on.

    # the probe -- run this FIRST, it decides whether anything else is worth doing
    python generate_omega.py --probe 10

    # corpus B: same model as the student, isolates strategy injection
    python generate_omega.py --split train --k 6 --keep 2 --out data/corpus-B.jsonl

    # corpus C: frontier expert, restores Omega's distillation gradient
    export OPENAI_API_KEY=...
    python generate_omega.py --split train --expert api --expert_model <id> \
                             --out data/corpus-C.jsonl

Corpus A is what you already have from the other arms' plain self-play. B vs A isolates
Omega's mechanism; C vs B isolates teacher strength. C vs A is the headline and confounds
both -- which is Omega's own confound, made visible here rather than hidden.
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import config                                        # noqa: E402
from csa_core import data_csa as data_csa                                      # noqa: E402
import experts                                       # noqa: E402
import paths                                         # noqa: E402
from csa_core import runlog                          # noqa: E402
from env_om import OmegaEnv                          # noqa: E402


def pick_opponent(case, mode, rng):
    if mode == 'none':
        return None
    adv = sorted({a['agent_id'] for a in case['agents']} - {case['decision_maker']})
    return rng.choice(adv)


def rollout(env, case, opponent=None, force_mode=None):
    env.reset(case, opponent=opponent, force_mode=force_mode)
    done = 0
    while not done:
        _c, done = env.step()
    return env.episode()


def rank_key(ep):
    """Rank within a scenario. dca first -- it is what the benchmark is about -- with
    cbar as the tiebreak because it is the denser competence signal."""
    s = ep['score']
    return (s['dca'], s['cbar'], s['disclosure_rate'])


# ------------------------------------------------------------------ probe
def probe(env, cases, n, rng, rollouts):
    """Force each scenario through BOTH modes and compare the chair turns.

    The whole method rests on one assumption: a 7B with the scaffold produces better
    turns than the same 7B without. If that is false the corpus is no better than plain
    self-play and the SFT is pointless. This costs minutes; the corpus pass costs hours.
    """
    print('probe: %d scenarios, forced fast vs forced slow\n' % n)
    rows = []
    for i, case in enumerate(cases[:n]):
        fast = rollout(env, case, force_mode='fast')
        slow = rollout(env, case, force_mode='slow')
        for mode, ep in (('fast', fast), ('slow', slow)):
            rollouts.add(case, ep, step=i, candidate=mode, reward=ep['score']['dca'],
                         force_mode=mode)
        rows.append((case['uid'], fast, slow))
        print('--- %s ---' % case['uid'])
        for lbl, ep in (('fast', fast), ('slow', slow)):
            chair = [t['content'] for t in ep['dialog'] if t['speaker'] == 'sys']
            print('  %s  dca %.3f  disclosed %s  addressed %s'
                  % (lbl, ep['score']['dca'], ep['revealed'] or '[]', ep['addressed']))
            for c in chair[:2]:
                print('       %s' % c[:150].replace('\n', ' '))
        print()

    def agg(key, idx):
        return sum(r[idx]['score'][key] if key in r[idx]['score'] else 0.0
                   for r in rows) / max(1, len(rows))

    print('=' * 70)
    print('%-22s %8s %8s' % ('', 'fast', 'slow'))
    for key in ('dca', 'disclosure_rate', 'cbar'):
        print('%-22s %8.3f %8.3f' % (key, agg(key, 1), agg(key, 2)))
    anyrev = lambda i: sum(1 for r in rows if r[i]['revealed'])   # noqa: E731
    print('%-22s %8d %8d' % ('episodes w/ a reveal', anyrev(1), anyrev(2)))
    print('%-22s %8.1f %8.1f' % ('calls per episode',
                                 sum(r[1]['n_calls'] for r in rows) / len(rows),
                                 sum(r[2]['n_calls'] for r in rows) / len(rows)))
    print('=' * 70)
    d = agg('disclosure_rate', 2) - agg('disclosure_rate', 1)
    if d <= 0.01:
        print('\nSTOP. Slow mode did not improve disclosure (%+.3f). The scaffold is not\n'
              'earning its calls on this model, and a full corpus pass would inherit\n'
              'that. Try a stronger expert (--expert api) before generating.' % d)
    else:
        print('\nSlow mode improves disclosure by %+.3f. Worth generating the corpus.' % d)
    return rows


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    p.add_argument('--k', type=int, default=config.Defaults.rollout_k)
    p.add_argument('--keep', type=int, default=config.Defaults.rollout_keep)
    p.add_argument('--out', default='')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--probe', type=int, default=0,
                   help='run the fast-vs-slow probe on N scenarios and exit')
    p.add_argument('--expert', default=config.Defaults.expert, choices=['local', 'api'])
    p.add_argument('--expert_model', default=config.Defaults.expert_model)
    p.add_argument('--api_base', default=config.Defaults.api_base)
    p.add_argument('--opponent', default=config.Defaults.opponent,
                   choices=['none', 'withhold'])
    p.add_argument('--device', default=config.Defaults.device)
    p.add_argument('--temperature', type=float, default=config.Defaults.temperature)
    p.add_argument('--stall_after', type=int, default=config.Defaults.stall_after)
    p.add_argument('--stall_patience', type=int, default=config.Defaults.stall_patience)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    p.add_argument('--restart', action='store_true')
    cli = p.parse_args()

    cfg = config.Defaults
    for k in ('expert', 'expert_model', 'api_base', 'device', 'temperature',
              'stall_after', 'stall_patience'):
        setattr(cfg, k, getattr(cli, k))

    rng = random.Random(cli.seed)
    cases = data_csa.load(cli.split)
    if cli.limit:
        cases = cases[:cli.limit]

    expert = experts.build(cfg)
    env = OmegaEnv(cfg, expert)
    print('expert: %s (%s)   scenarios: %d   opponent: %s'
          % (cfg.expert_model, cfg.expert, len(cases), cli.opponent))

    tag = 'B' if cfg.expert == 'local' else 'C'
    if cli.probe:
        rollouts = runlog.RolloutLog(paths.LOGS, 'probe-%s-%s' % (tag, cli.split),
                                     algo='omega-probe')
        probe(env, cases, cli.probe, rng, rollouts)
        rollouts.close()
        return
    out_path = cli.out or os.path.join(paths.DATA, 'corpus-%s-%s.jsonl' % (tag, cli.split))
    done_uids = set()
    if os.path.exists(out_path) and not cli.restart:
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_uids.add(json.loads(line)['uid'])
                except Exception:                    # noqa: BLE001
                    pass
        if done_uids:
            print('resuming: %d scenarios already generated' % len(done_uids), flush=True)

    rollouts = runlog.RolloutLog(paths.LOGS, 'corpus-%s-%s' % (tag, cli.split),
                                 algo='omega-selfplay-%s' % tag, append=bool(done_uids))
    stats = collections.Counter()
    kept, t0 = [], time.time()
    with open(out_path, 'a' if done_uids else 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            if case['uid'] in done_uids:
                continue
            opp = pick_opponent(case, cli.opponent, rng)
            cands = []
            for _j in range(cli.k):
                ep = rollout(env, case, opponent=opp)
                ep['rank'] = rank_key(ep)
                ep['_j'] = _j
                cands.append(ep)
                stats['stalled' if ep['stalled_at'] is not None else 'never_stalled'] += 1
            cands.sort(key=lambda e: e['rank'], reverse=True)
            spread = cands[0]['rank'][0] - cands[-1]['rank'][0]
            stats['flat' if spread == 0 else 'varied'] += 1
            # every rollout is logged, not only the kept ones
            for pos, c in enumerate(cands):
                c.pop('rank', None)
                rollouts.add(case, c, step=i, candidate=c.pop('_j', None),
                             reward=c['score']['dca'], rank=pos, kept=pos < cli.keep,
                             opponent=opp, stalled_at=c['stalled_at'])
            for c in cands[:cli.keep]:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
                kept.append(c)
            f.flush()
            if (i + 1) % 5 == 0:
                print('  %3d/%d  kept dca %.3f  stalled %d/%d rollouts  %.1f min'
                      % (i + 1, len(cases),
                         sum(k['score']['dca'] for k in kept) / max(1, len(kept)),
                         stats['stalled'], stats['stalled'] + stats['never_stalled'],
                         (time.time() - t0) / 60), flush=True)

    rollouts.close()
    print('\nwrote %s  (%d episodes)' % (out_path, len(kept)))
    if kept:
        dca = [k['score']['dca'] for k in kept]
        print('  dca      mean %.3f median %.3f max %.3f' % (sum(dca) / len(dca),
                                                             st.median(dca), max(dca)))
        print('  disclosure mean %.3f'
              % (sum(k['score']['disclosure_rate'] for k in kept) / len(kept)))
        slow_turns = sum(m == 'slow' for k in kept for m in k['modes'])
        all_turns = sum(len(k['modes']) for k in kept)
        print('  chair turns in slow mode: %d/%d (%.0f%%)'
              % (slow_turns, all_turns, 100 * slow_turns / max(1, all_turns)))
    tot = stats['stalled'] + stats['never_stalled']
    print('  rollouts that stalled: %d/%d (%.0f%%)'
          % (stats['stalled'], tot, 100 * stats['stalled'] / max(1, tot)))
    if stats['stalled'] == 0:
        print('  WARNING: nothing ever stalled, so the intervention never fired and this '
              'corpus is plain self-play. Lower --stall_after / --stall_patience.')
    elif stats['stalled'] == tot:
        print('  NOTE: everything stalled, so slow mode is effectively always on and the '
              'adaptive part is untested. Raise --stall_after to make it selective.')
    if getattr(expert, 'n_failed', 0):
        print('  expert calls that failed after retries: %d' % expert.n_failed)


if __name__ == '__main__':
    main()
