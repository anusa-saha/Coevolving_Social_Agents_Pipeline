"""Why does the planner never predict `followup`?

Two very different causes produce the same 0.000 F1, and they need different fixes:

  (a) the model learned NOTHING about the class -- followup sits at the bottom of the
      probability ranking, indistinguishable from noise. Only more/better data helps.

  (b) the model learned the class but never wins the argmax -- followup ranks 2nd
      consistently, beaten by `ask` because `ask` is 7x more frequent and unweighted
      cross-entropy rewards betting on the prior. Class weighting or a decision
      threshold fixes this outright.

This reports the rank and probability mass assigned to the true class, which separates
the two. It also reports top-2 accuracy: if followup is usually 2nd, the representation
is there and only the decision rule is wrong.

    python diagnose_followup.py --checkpoint sft/csa/roberta/best_checkpoint \
                                --data data_sft/csa-test.txt
"""
import argparse
import collections
import json
import os
import sys

import torch
from transformers import RobertaTokenizer, RobertaConfig

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from agent import PPDPP                                    # noqa: E402
from eval_planner import Args, load_examples               # noqa: E402
from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='sft/csa/roberta/best_checkpoint')
    p.add_argument('--data', default='data_sft/csa-test.txt')
    p.add_argument('--focus', default='followup')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    cli = p.parse_args()

    args = Args()
    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path,
                                           cache_dir=args.cache_dir)
    cfg = RobertaConfig.from_pretrained(args.model_name_or_path,
                                        cache_dir=args.cache_dir)
    model = PPDPP(args, cfg, tok)
    sd = torch.load(os.path.join(cli.checkpoint, 'pytorch_model.bin'),
                    map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.to(cli.device).eval()

    ex = load_examples(cli.data)
    rows = []
    for e in ex:
        ids = []
        for s in e['prefix'][::-1]:
            enc = tok.encode(s)
            if len(ids) + len(enc) > args.max_seq_length:
                break
            ids = enc[1:] + ids
        ids = enc[:1] + ids
        with torch.no_grad():
            t = torch.tensor([ids[-args.max_seq_length + 1:]]).long().to(cli.device)
            logits = model.classifier(model.dropout(model.policy(t)[1]))
            pr = torch.softmax(logits, dim=1)[0].cpu().tolist()
        order = sorted(range(len(pr)), key=lambda i: -pr[i])
        rank = {LABELS[j]: k + 1 for k, j in enumerate(order)}
        rows.append({'true': e['label'], 'probs': dict(zip(LABELS, pr)),
                     'pred': LABELS[order[0]], 'rank_of_true': rank[e['label']],
                     'utt': e['prefix'][-1][:110]})

    n = len(rows)
    top1 = sum(r['pred'] == r['true'] for r in rows) / n
    top2 = sum(r['rank_of_true'] <= 2 for r in rows) / n
    print('overall   top-1 %.3f    top-2 %.3f    (n=%d)\n' % (top1, top2, n))

    print('--- mean predicted probability of each class, over ALL turns ---')
    for lab in LABELS:
        m = sum(r['probs'][lab] for r in rows) / n
        mx = max(r['probs'][lab] for r in rows)
        print('  %-9s mean %.3f   max %.3f   argmax-wins %d'
              % (lab, m, mx, sum(1 for r in rows if r['pred'] == lab)))

    foc = [r for r in rows if r['true'] == cli.focus]
    print('\n--- the %d true `%s` turns ---' % (len(foc), cli.focus))
    if foc:
        rk = collections.Counter(r['rank_of_true'] for r in foc)
        print('  rank of the correct class:  %s'
              % '  '.join('rank%d x%d' % (k, rk[k]) for k in sorted(rk)))
        print('  mean prob assigned to `%s` on these turns : %.3f'
              % (cli.focus, sum(r['probs'][cli.focus] for r in foc) / len(foc)))
        print('  mean prob assigned to `%s` on other turns  : %.3f'
              % (cli.focus, sum(r['probs'][cli.focus] for r in rows
                                if r['true'] != cli.focus) / max(1, n - len(foc))))
        print('  what it predicted instead : %s'
              % dict(collections.Counter(r['pred'] for r in foc)))
        print('\n  per-turn detail (p_focus = probability given to the correct class):')
        for r in foc:
            print('    p_%s=%.3f rank%d -> pred %-8s | %s'
                  % (cli.focus, r['probs'][cli.focus], r['rank_of_true'],
                     r['pred'], r['utt'].replace('\n', ' ')))

    # If the class were merely losing the argmax race, boosting it should recover it.
    print('\n--- would a decision-threshold boost recover it? ---')
    for boost in (1, 2, 3, 5, 8, 12):
        pred = []
        for r in rows:
            sc = dict(r['probs'])
            sc[cli.focus] *= boost
            pred.append(max(sc, key=sc.get))
        acc = sum(p == r['true'] for p, r in zip(pred, rows)) / n
        tp = sum(1 for p, r in zip(pred, rows) if p == cli.focus and r['true'] == cli.focus)
        fp = sum(1 for p, r in zip(pred, rows) if p == cli.focus and r['true'] != cli.focus)
        fn = sum(1 for p, r in zip(pred, rows) if p != cli.focus and r['true'] == cli.focus)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print('  x%-3d  overall acc %.3f   %s: prec %.2f rec %.2f F1 %.3f  (tp %d fp %d)'
              % (boost, acc, cli.focus, prec, rec, f1, tp, fp))


if __name__ == '__main__':
    main()
