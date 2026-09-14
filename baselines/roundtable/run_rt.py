"""Run the round-table baseline over a split.

    python run_rt.py --decide chair --split test
    python run_rt.py --decide vote  --split test
    python run_rt.py --backend api --api_model gpt-5.4-luna --decide chair --split test

Writes logs/Record-rt-<decide>-<backend>-<split>.txt in the same schema every other arm
uses, so `cd ../analysis && python compute_extended_metrics.py` scores it alongside them.
"""
import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import _compat as compat
import _data_csa as data_csa
import paths

import config
import prompts_rt as P
from backends import build
from env_rt import RoundTableEnv


def record_path(decide, backend, split):
    return os.path.join(paths.LOGS, 'Record-rt-%s-%s-%s.txt' % (decide, backend, split))


def summarise(recs):
    if not recs:
        return {}
    sc = lambda r: (r.get('score') or {})            # noqa: E731

    def avg(fn):
        v = [fn(r) for r in recs if isinstance(fn(r), (int, float))]
        return sum(v) / len(v) if v else float('nan')

    return {'n': len(recs),
            'dca': avg(lambda r: sc(r).get('dca')),
            'disclosure_rate': avg(lambda r: sc(r).get('disclosure_rate')),
            'cbar': avg(lambda r: sc(r).get('cbar')),
            'pbar': avg(lambda r: sc(r).get('pbar')),
            'schema_valid': avg(lambda r: 1.0 if sc(r).get('schema_valid') else 0.0),
            'SR': avg(lambda r: 1.0 if r.get('done') == 1 else 0.0),
            'leaks': sum(1 for r in recs if r.get('leaks')),
            'n_calls': avg(lambda r: r.get('n_calls')),
            'turns': avg(lambda r: r.get('turns'))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default=None, choices=['local', 'api'])
    p.add_argument('--model', default=None, help='local model id')
    p.add_argument('--api_model', default=None, help='API model id, e.g. gpt-5.4-luna')
    p.add_argument('--api_base', default=None, help='OpenAI-compatible base URL')
    p.add_argument('--decide', default=None, choices=list(P.DECIDERS))
    p.add_argument('--rounds', type=int, default=None,
                   help='passes around the table before settling')
    p.add_argument('--converge_rounds', type=int, default=None)
    p.add_argument('--max_new_tokens', type=int, default=None)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--split', default=config.Defaults.eval_split,
                   choices=['train', 'valid', 'test'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=None)
    cli = p.parse_args()

    cfg = config.apply_cli(cli)

    if compat.report():
        raise SystemExit('\nfix the blocking problems above, then re-run')

    cases = data_csa.load(cli.split)
    if cli.limit:
        cases = cases[:cli.limit]
    model_name = cfg.model if cfg.backend == 'local' else cfg.api_model
    print('round table | backend %s (%s) | decide %s | rounds %d | %d scenarios'
          % (cfg.backend, model_name, cfg.decide, cfg.rounds, len(cases)))

    backend = build(cfg)
    env = RoundTableEnv(cfg, backend)

    out = record_path(cfg.decide, cfg.backend, cli.split)
    recs = []
    with open(out, 'w', encoding='utf-8') as f:
        for i, case in enumerate(cases, 1):
            rec = env.run(case)
            recs.append(rec)
            f.write(repr(rec) + '\n\n')
            f.flush()                    # a long run should survive being interrupted
            s = rec['score']
            print('  [%3d/%3d] %-42s dca %.3f  disc %.3f  calls %3d'
                  % (i, len(cases), rec['uid'], s['dca'], s['disclosure_rate'],
                     rec['n_calls']))

    print('\nwrote %s' % out)
    summ = summarise(recs)
    print('%-18s %s' % ('metric', 'value'))
    for k in ('n', 'dca', 'disclosure_rate', 'cbar', 'pbar', 'schema_valid', 'SR',
              'leaks', 'n_calls', 'turns'):
        v = summ.get(k)
        print('%-18s %s' % (k, ('%.4f' % v) if isinstance(v, float) else v))
    if getattr(backend, 'n_failed', 0):
        print('\nNOTE %d generation call(s) failed and were recorded as empty turns.'
              % backend.n_failed)


if __name__ == '__main__':
    main()
