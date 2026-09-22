"""STAGE 3 -- evaluate the student on the held-out scenarios.

The trained chair runs inside an unchanged environment: advisors speak with the BASE
weights (adapter off), turn order and detectors match every other arm, and decoding is
greedy so the number is reproducible.

The chair runs in FAST mode by default. Omega's slow scaffold is a generation-time device
for building the corpus; the student is supposed to have absorbed it. Evaluating with the
scaffold still on would measure the scaffold rather than the training -- `--eval_mode
adaptive` exists to quantify exactly that gap, and must be reported separately if used.

    python evaluate_om.py --adapter ckpt/sft-B --tag omega-B
    python evaluate_om.py --adapter "" --tag base          # untrained, the floor
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import config                                        # noqa: E402
from csa_core import data_csa as data_csa                                      # noqa: E402
import paths                                         # noqa: E402
from csa_core import runlog                          # noqa: E402
from env_om import OmegaEnv                          # noqa: E402
from student import Student, StudentExpert           # noqa: E402


def summarise(recs):
    if not recs:
        return {}

    def avg(fn):
        v = [fn(r) for r in recs if fn(r) is not None]
        return sum(v) / len(v) if v else float('nan')

    sc = lambda r: (r.get('score') or {})            # noqa: E731
    return {
        'n': len(recs),
        'SR': avg(lambda r: 1.0 if r.get('done') == 1 else 0.0),
        'dca': avg(lambda r: sc(r).get('dca')),
        'dca_norm': avg(lambda r: r.get('dca_norm')),
        'ceiling': avg(lambda r: r.get('ceiling')),
        'disclosure_rate': avg(lambda r: sc(r).get('disclosure_rate')),
        'any_reveal': sum(1 for r in recs if r.get('revealed')),
        'elicited_frac': avg(lambda r: (sum(1 for v in r['reveal_elicited'].values() if v)
                                        / max(1, len(r['reveal_elicited'])))
                             if r.get('reveal_elicited') else 0.0),
        'cbar': avg(lambda r: sc(r).get('cbar')),
        'pbar': avg(lambda r: sc(r).get('pbar')),
        'schema_valid': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
        'cover': avg(lambda r: r.get('cover')),
        'leaks': sum(1 for r in recs if r.get('leaks')),
        'opponent_leaks': sum(1 for r in recs if r.get('opponent_leaks')),
        'stalled': sum(1 for r in recs if r.get('stalled_at') is not None),
        'turns': avg(lambda r: r.get('turns')),
        'n_calls': avg(lambda r: r.get('n_calls')),
    }


def _by(recs, key):
    g = collections.defaultdict(list)
    for r in recs:
        g[r.get(key)].append(r)
    return {str(k): summarise(v) for k, v in sorted(g.items(), key=lambda x: str(x[0]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', default='', help='"" evaluates the untrained student')
    p.add_argument('--split', default=config.Defaults.eval_split,
                   choices=['test', 'valid', 'train'])
    p.add_argument('--tag', default='omega')
    p.add_argument('--eval_mode', default='fast', choices=['fast', 'adaptive'],
                   help='fast = no scaffold, measures what the student learned; '
                        'adaptive = scaffold still on, measures the scaffold')
    p.add_argument('--opponent', default='none', choices=['none', 'withhold'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default=config.Defaults.device)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    cli = p.parse_args()

    cfg = config.Defaults
    cfg.device = cli.device
    cfg.temperature = 0.0                            # greedy: reproducible
    torch.manual_seed(cli.seed)
    rng = random.Random(cli.seed)

    cases = data_csa.load(cli.split)
    if cli.limit:
        cases = cases[:cli.limit]

    student = Student(cfg, adapter_dir=(cli.adapter or None))
    student.eval()

    out_path = os.path.join(paths.LOGS, 'Record-%s-%s.txt' % (cli.tag, cli.split))
    recs, t0 = [], time.time()
    conv = runlog.EvalLog(paths.LOGS, '%s-%s' % (cli.tag, cli.split), tag=cli.tag)
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases):
            chair = next(a['name'] for a in case['agents']
                         if a['agent_id'] == case['decision_maker'])
            env = OmegaEnv(cfg, StudentExpert(student, chair))
            opp = None
            if cli.opponent != 'none':
                adv = sorted({a['agent_id'] for a in case['agents']}
                             - {case['decision_maker']})
                opp = rng.choice(adv)
            env.reset(case, opponent=opp,
                      force_mode=('fast' if cli.eval_mode == 'fast' else None))
            done = 0
            while not done:
                _c, done = env.step()
            rec = env.episode()
            rec['tag'], rec['eval_mode'] = cli.tag, cli.eval_mode
            recs.append(rec)
            f.write('%s\n\n' % str(rec))
            conv.add(case, rec)
            if (i + 1) % 10 == 0:
                print('  %d/%d  %.1f min' % (i + 1, len(cases),
                                             (time.time() - t0) / 60), flush=True)

    headline = conv.close()
    summ = summarise(recs)
    summ.update({'tag': cli.tag, 'split': cli.split, 'eval_mode': cli.eval_mode,
                 'opponent': cli.opponent, 'adapter': cli.adapter or None})
    with open(os.path.join(paths.LOGS, 'summary-%s-%s.json' % (cli.tag, cli.split)),
              'w', encoding='utf-8') as f:
        json.dump({'summary': summ, 'headline': headline,
                   'by_domain': _by(recs, 'domain'),
                   'by_num_agents': _by(recs, 'num_agents')}, f, indent=1)

    print('\nrecords -> %s' % out_path)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in summ.items()}, indent=1))
    if cli.opponent != 'none':
        print('\nNOTE: an opponent was active, so the dca ceiling is below 1.0 (mean '
              '%.3f here). Compare dca_norm, and only against runs with the same '
              '--opponent setting.' % summ['ceiling'])
    if cli.eval_mode == 'adaptive':
        print('\nNOTE: the stall scaffold was ON during evaluation. This number includes '
              'the scaffold and is not comparable to the other arms; report it as a '
              'separate row.')


if __name__ == '__main__':
    main()
