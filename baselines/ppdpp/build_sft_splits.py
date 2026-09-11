"""Split annotated conversations 80/10/10 into the format sft.py consumes.

The split is over SCENARIOS, not turns: every conversation from one scenario lands in
exactly one split. Splitting by turn would put turns from the same conversation in both
train and test, and the classifier would be scored on near-duplicates of what it saw.

Note this is a different split from the RL experiment's 99/9/42. The 42 RL-test
scenarios have no generated conversations at all, so they stay untouched here and the
classifier never sees them.

    python build_sft_splits.py --in conversations/conversations-train-labelled.jsonl \
                               conversations/conversations-valid-labelled.jsonl
"""
import argparse
import collections
import io
import json
import os
import random
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infiles', nargs='+', required=True)
    p.add_argument('--out_dir', default='data_sft')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_frac', type=float, default=0.8)
    p.add_argument('--test_frac', type=float, default=0.1)
    args = p.parse_args()

    convs = []
    for f in args.infiles:
        if not os.path.exists(f):
            raise SystemExit('missing input: %s' % f)
        convs += [json.loads(l) for l in open(f, encoding='utf-8')]

    by_uid = collections.defaultdict(list)
    for c in convs:
        by_uid[c['uid']].append(c)
    uids = sorted(by_uid)

    # stratify on domain so each split spans all three
    buckets = collections.defaultdict(list)
    for u in uids:
        buckets[u.split('::')[0]].append(u)

    rng = random.Random(args.seed)
    train, test, valid = [], [], []
    for dom in sorted(buckets):
        b = sorted(buckets[dom])
        rng.shuffle(b)
        n = len(b)
        n_tr = int(args.train_frac * n)
        n_te = max(1, int(args.test_frac * n))
        train += b[:n_tr]
        test += b[n_tr:n_tr + n_te]
        valid += b[n_tr + n_te:]

    assert not (set(train) & set(test)) and not (set(train) & set(valid)) \
        and not (set(test) & set(valid)), 'scenario leaked across splits'

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}
    for name, uid_list in (('train', train), ('test', test), ('valid', valid)):
        path = os.path.join(args.out_dir, 'csa-%s.txt' % name)
        counts = collections.Counter()
        n_turns = n_conv = 0
        with open(path, 'w', encoding='utf-8') as f:
            for u in uid_list:
                for c in by_uid[u]:
                    dialog = []
                    for t in c['dialog']:
                        e = {'role': t['role'], 'content': t['content'],
                             'speaker': t['speaker']}
                        # only annotated turns carry a label sft.py will train on
                        if t.get('annotate') and t.get('strategy') in LABELS:
                            e['strategy'] = t['strategy']
                            counts[t['strategy']] += 1
                            n_turns += 1
                        dialog.append(e)
                    f.write(json.dumps({'uid': c['uid'], 'score': c.get('score'),
                                        'dialog': dialog}, ensure_ascii=False) + '\n')
                    n_conv += 1
        stats[name] = {'scenarios': len(uid_list), 'conversations': n_conv,
                       'labelled_turns': n_turns, 'by_label': dict(counts)}
        print('%-6s scenarios %3d  conversations %3d  labelled turns %4d  %s -> %s'
              % (name, len(uid_list), n_conv, n_turns,
                 dict(sorted(counts.items())), path))

    with open(os.path.join(args.out_dir, 'split_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1)

    total = sum(s['labelled_turns'] for s in stats.values())
    print('\ntotal labelled turns: %d' % total)
    maj = collections.Counter()
    for s in stats.values():
        maj.update(s['by_label'])
    if maj:
        top, n = maj.most_common(1)[0]
        print('majority-class baseline: predict "%s" every time -> %.1f%% accuracy'
              % (top, 100 * n / total))
        print('(any trained model must beat that to be worth anything)')


if __name__ == '__main__':
    main()
