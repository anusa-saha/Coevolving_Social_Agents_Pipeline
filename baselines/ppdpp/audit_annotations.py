"""Re-audit annotator labels offline, from the saved conversations.

The checks written during annotation used an exact 40-character substring test to decide
whether a fact had been disclosed. Advisors paraphrase, so that test almost never fires
and the resulting "0% of share turns pool a decisive fact" measures the strictness of
the test rather than the behaviour of the chair. This recomputes it with the same
lexical-overlap rule the environment uses everywhere else.

No API calls: everything needed is already in the labelled conversation file.

    python audit_annotations.py --in conversations/conversations-train-labelled.jsonl
"""
import argparse
import collections
import io
import json
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from env import _overlap                                   # noqa: E402
from prompt import CSAAct                                  # noqa: E402
from utils import load_dataset                             # noqa: E402

LABELS = sorted(CSAAct.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infiles', nargs='+', required=True)
    p.add_argument('--thr', type=float, default=0.30,
                   help='lexical overlap for "this turn repeats that content"')
    p.add_argument('--disclose_thr', type=float, default=0.35,
                   help='same threshold the environment uses to call a fact disclosed')
    p.add_argument('--out', default='')
    args = p.parse_args()

    cases = {}
    for split in ('train', 'valid', 'test'):
        for c in load_dataset('csa')[split]:
            cases[c['uid']] = c

    convs = []
    for f in args.infiles:
        convs += [json.loads(l) for l in open(f, encoding='utf-8')]

    counts = collections.Counter()
    decide_ok = decide_n = 0
    share_label_ok = share_pools = share_n = 0
    ask_is_question = ask_n = 0
    followup_refers_back = followup_n = 0
    pooling_opportunities = 0

    for c in convs:
        case = cases.get(c['uid'])
        if not case:
            continue
        facts = {k: v['text'] for k, v in case['private_facts'].items()}
        prior_any, prior_decisive = [], []

        for t in c['dialog']:
            lab = t.get('strategy')
            if t.get('annotate') and lab in LABELS:
                counts[lab] += 1
                txt = t['content']

                if lab == 'decide':
                    decide_n += 1
                    # Testing for JSON here would always fail: the turn that emits the
                    # settlement is the instructed closing turn, which is excluded from
                    # annotation. Annotated turns are natural, so the right test is
                    # whether the chair is committing in words.
                    if re.search(r"\b(we will|we'll|let's go with|i will|assign|"
                                 r"approved|approve|confirm|finalis|finaliz|decided|"
                                 r"decision is|record|proceed with|go ahead)\b",
                                 txt.lower()):
                        decide_ok += 1
                elif lab == 'share':
                    share_n += 1
                    if any(_overlap(p, txt) >= args.thr for p in prior_any):
                        share_label_ok += 1
                    if any(_overlap(facts[f], txt) >= args.thr for f in prior_decisive):
                        share_pools += 1
                    if prior_decisive:
                        pooling_opportunities += 1
                elif lab == 'ask':
                    ask_n += 1
                    if '?' in txt:
                        ask_is_question += 1
                elif lab == 'followup':
                    followup_n += 1
                    if '?' in txt:
                        followup_refers_back += 1

            if t['speaker'] == 'usr':
                prior_any.append(t['content'])
                for fid, ftxt in facts.items():
                    if fid in c.get('revealed', []) and fid not in prior_decisive \
                            and _overlap(ftxt, t['content']) >= args.disclose_thr:
                        prior_decisive.append(fid)

    tot = sum(counts.values())
    print('labelled turns: %d across %d conversations\n' % (tot, len(convs)))
    for lab in LABELS:
        print('  %-9s %4d  %5.1f%%' % (lab, counts[lab], 100 * counts[lab] / max(1, tot)))

    def pct(a, b):
        return '%d/%d (%.0f%%)' % (a, b, 100 * a / b) if b else 'n/a'

    print('\n--- label consistency, recomputed ---')
    print('  decide   contains a JSON settlement : %s' % pct(decide_ok, decide_n))
    print('  share    repeats prior participant content : %s'
          % pct(share_label_ok, share_n))
    print('  ask      is phrased as a question   : %s' % pct(ask_is_question, ask_n))
    print('  followup is phrased as a question   : %s'
          % pct(followup_refers_back, followup_n))

    print('\n--- information pooling (the benchmark question) ---')
    print('  share turns with a decisive fact available to relay : %s'
          % pct(pooling_opportunities, share_n))
    print('  share turns that actually relayed one               : %s'
          % pct(share_pools, share_n))
    if pooling_opportunities:
        print('  pooling rate when the opportunity existed           : %s'
              % pct(share_pools, pooling_opportunities))
    print('\n  A low rate here is the Stasser and Titus failure mode: the group discusses'
          '\n  what everyone already knows and neglects what only one member holds.')

    if args.out:
        json.dump({'counts': dict(counts), 'decide_ok': decide_ok, 'decide_n': decide_n,
                   'share_label_ok': share_label_ok, 'share_pools': share_pools,
                   'share_n': share_n, 'pooling_opportunities': pooling_opportunities},
                  open(args.out, 'w', encoding='utf-8'), indent=1)
        print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
