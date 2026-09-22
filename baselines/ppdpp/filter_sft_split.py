"""Copy the archived planner SFT data into data_sft/, keeping RL-train scenarios only.

data_sft/ holds the annotated chair turns that PPDPP's planner (sft.py) and EPO's
strategist (make_strategies.py -> sft_epo.py) are supervised from. It was built under the
original 3-domain split. Re-derived under the current benchmark, some of its scenarios now
sit in the RL valid and test splits -- on the archive, 11 of the 75 training scenarios
(83 labelled chair turns) are RL-test scenarios, so training on it as-is trains on the
test set.

Every conversation whose scenario is in the RL TRAIN split is kept, in all three files;
everything else is dropped. The classifier's own valid/test files stay held out from its
training, and no RL evaluation scenario is seen by any stage. Nothing is re-annotated, so
the domains the archive never covered stay uncovered.

    python filter_sft_split.py                 # archive -> ./data_sft
    python filter_sft_split.py --src DIR       # explicit source directory
"""
import argparse
import collections
import io
import json
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from csa_core import data_csa, paths                 # noqa: E402

SPLITS = ('train', 'valid', 'test')


def find_source(out_dir):
    """The archive first; the output directory itself only as a last resort."""
    cands = []
    if os.environ.get('CSA_ARTIFACTS_DIR'):
        cands.append(os.path.join(os.environ['CSA_ARTIFACTS_DIR'], 'ppdpp', 'data_sft'))
    cands.append(os.path.join(os.path.dirname(ROOT), 'csa-artifacts', 'ppdpp', 'data_sft'))
    ref = paths.find_reference('data_sft/csa-train.txt')
    if ref:
        cands.append(os.path.dirname(ref))
    out = os.path.abspath(out_dir)
    for c in cands:
        if os.path.isfile(os.path.join(c, 'csa-train.txt')) and os.path.abspath(c) != out:
            return c
    # Re-filtering in place is safe: the filter only ever removes rows.
    return out if os.path.isfile(os.path.join(out, 'csa-train.txt')) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=None)
    p.add_argument('--out_dir', default=os.path.join(HERE, 'data_sft'))
    cli = p.parse_args()

    src = cli.src or find_source(cli.out_dir)
    if not src:
        raise SystemExit('no archived data_sft/ found. Set CSA_ARTIFACTS_DIR, pass --src, '
                         'or rebuild it (PPDPP README, "Building the training material").')
    rl = {r['uid']: name for name in SPLITS for r in data_csa.load(name)}

    kept, stats = {}, {}
    for s in SPLITS:
        path = os.path.join(src, 'csa-%s.txt' % s)
        lines = []
        if os.path.isfile(path):
            with io.open(path, encoding='utf-8') as f:
                lines = [l.rstrip('\n') + '\n' for l in f if l.strip()]
        keep, dropped, turns = [], collections.Counter(), collections.Counter()
        for line in lines:
            conv = json.loads(line)
            where = rl.get(conv['uid'], 'outside the benchmark')
            n = sum(1 for t in conv.get('dialog') or () if t.get('strategy'))
            if where == 'train':
                keep.append(line)
                turns['kept'] += n
            else:
                dropped[where] += 1
                turns['dropped'] += n
        kept[s] = keep
        stats[s] = {'conversations_in': len(lines), 'conversations_kept': len(keep),
                    'scenarios_kept': len({json.loads(l)['uid'] for l in keep}),
                    'dropped_by_rl_split': dict(dropped),
                    'labelled_turns_kept': turns['kept'],
                    'labelled_turns_dropped': turns['dropped']}

    os.makedirs(cli.out_dir, exist_ok=True)
    # data_reader.py caches tokenised features beside the data files. A cache built from
    # unfiltered files would be read back silently, undoing this.
    for name in os.listdir(cli.out_dir):
        if name.startswith('sft_csa_'):
            os.remove(os.path.join(cli.out_dir, name))
    for s in SPLITS:
        with io.open(os.path.join(cli.out_dir, 'csa-%s.txt' % s), 'w', encoding='utf-8',
                     newline='') as f:
            f.writelines(kept[s])
    with open(os.path.join(cli.out_dir, 'filter_stats.json'), 'w', encoding='utf-8') as f:
        json.dump({'source': os.path.abspath(src),
                   'rule': 'keep conversations whose scenario is in the RL train split',
                   'splits': stats}, f, indent=1)

    print('source %s' % os.path.abspath(src))
    for s in SPLITS:
        st = stats[s]
        print('  %-5s kept %3d/%3d conversations (%d scenarios, %d labelled turns); '
              'dropped %s'
              % (s, st['conversations_kept'], st['conversations_in'], st['scenarios_kept'],
                 st['labelled_turns_kept'], st['dropped_by_rl_split'] or 'none'))
    if not kept['train']:
        raise SystemExit('nothing left to train on')
    print('-> %s' % os.path.abspath(cli.out_dir))


if __name__ == '__main__':
    main()
