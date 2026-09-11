"""Where in a meeting does each intent occur, and what would raise the `followup` rate?

`followup` is 6.5% of training turns and the planner never predicts it. Before spending
GPU hours generating more, this asks whether the scarcity is structural: a chair cannot
follow up on turn 1, because nothing has been answered yet. If `followup` density rises
with turn position, then generating LONGER conversations raises the followup rate rather
than merely scaling every class equally -- a far cheaper fix than 3x more rollouts.

This touches no model and no test data; it only describes the labelled corpus.

    python analyse_label_structure.py --in conversations/conversations-train-labelled.jsonl \
                                           conversations/conversations-valid-labelled.jsonl
"""
import argparse
import collections
import json
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())


def bar(frac, width=28):
    n = int(round(frac * width))
    return '#' * n + '.' * (width - n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infiles', nargs='+', required=True)
    args = p.parse_args()

    convs = []
    for f in args.infiles:
        convs += [json.loads(l) for l in open(f, encoding='utf-8')]

    by_pos = collections.defaultdict(collections.Counter)   # chair-turn index -> labels
    by_dom = collections.defaultdict(collections.Counter)
    by_nag = collections.defaultdict(collections.Counter)
    overall = collections.Counter()
    chair_turns_per_conv = []
    first_followup_pos = []
    transitions = collections.Counter()

    for c in convs:
        labs = [t.get('strategy') for t in c['dialog']
                if t.get('annotate') and t.get('strategy') in LABELS]
        chair_turns_per_conv.append(len(labs))
        dom = (c.get('domain') or c.get('uid', '::').split('::')[0])
        nag = c.get('num_agents')
        for i, lab in enumerate(labs):
            overall[lab] += 1
            by_pos[i][lab] += 1
            by_dom[dom][lab] += 1
            if nag:
                by_nag[nag][lab] += 1
        for a, b in zip(labs, labs[1:]):
            transitions[(a, b)] += 1
        fu = [i for i, l in enumerate(labs) if l == 'followup']
        if fu:
            first_followup_pos.append(fu[0])

    tot = sum(overall.values())
    print('corpus: %d conversations, %d labelled chair turns' % (len(convs), tot))
    print('chair turns per conversation: mean %.2f  min %d  max %d\n'
          % (sum(chair_turns_per_conv) / len(chair_turns_per_conv),
             min(chair_turns_per_conv), max(chair_turns_per_conv)))

    print('=== overall label distribution ===')
    for l in LABELS:
        print('  %-9s %4d  %5.1f%%  %s'
              % (l, overall[l], 100 * overall[l] / tot, bar(overall[l] / tot)))

    print('\n=== label distribution BY CHAIR-TURN POSITION ===')
    print('(this is the question: does followup need long conversations?)\n')
    print('  %-5s %6s  %s' % ('pos', 'n', ''.join('%10s' % l for l in LABELS)))
    for i in sorted(by_pos):
        row = by_pos[i]
        n = sum(row.values())
        if n < 5:
            continue
        print('  %-5d %6d  %s' % (i, n,
              ''.join('%9.1f%%' % (100 * row[l] / n) for l in LABELS)))

    print('\n  followup rate by position:')
    for i in sorted(by_pos):
        row = by_pos[i]
        n = sum(row.values())
        if n < 5:
            continue
        fr = row['followup'] / n
        print('    pos %-3d n=%-5d %5.1f%%  %s' % (i, n, 100 * fr, bar(fr, 40)))

    if first_followup_pos:
        print('\n  conversations containing >=1 followup : %d/%d (%.0f%%)'
              % (len(first_followup_pos), len(convs),
                 100 * len(first_followup_pos) / len(convs)))
        print('  mean position of FIRST followup       : %.2f'
              % (sum(first_followup_pos) / len(first_followup_pos)))

    print('\n=== by domain ===')
    for d in sorted(by_dom):
        row = by_dom[d]
        n = sum(row.values())
        print('  %-12s n=%-5d %s' % (d, n,
              ''.join('%s %4.1f%%  ' % (l, 100 * row[l] / n) for l in LABELS)))

    if by_nag:
        print('\n=== by number of agents ===')
        for k in sorted(by_nag):
            row = by_nag[k]
            n = sum(row.values())
            print('  %-3s agents n=%-5d %s' % (k, n,
                  ''.join('%s %4.1f%%  ' % (l, 100 * row[l] / n) for l in LABELS)))

    print('\n=== most common label transitions (what follows what) ===')
    ttot = sum(transitions.values())
    for (a, b), n in transitions.most_common(10):
        print('  %-9s -> %-9s %4d  %4.1f%%' % (a, b, n, 100 * n / ttot))

    print('\n=== projection: turns needed for a target followup count ===')
    fr = overall['followup'] / tot
    print('  current followup rate: %.3f (%d of %d)' % (fr, overall['followup'], tot))
    for target in (100, 150, 200, 300):
        need = target / fr
        print('  to reach %3d followup turns at this rate: %5.0f total labelled turns '
              '(%.1fx current)' % (target, need, need / tot))


if __name__ == '__main__':
    main()
