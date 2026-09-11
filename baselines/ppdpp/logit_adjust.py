"""Recover the rare classes post-hoc, without retraining.

The planner scores `followup` well (ROC-AUC 0.861) but never selects it (F1 0.000):
under a 7:1 imbalance the argmax always prefers `ask`. That is a decision-rule failure,
so the fix belongs at the decision rule.

Logit adjustment (Menon et al., 2021, "Long-tail learning via logit adjustment") shifts
each class by its log training prior:

    adjusted_k = log p_k  -  tau * log prior_k

tau = 0 leaves the model untouched; tau = 1 fully removes the prior. One scalar, fitted
on VALIDATION and then applied unchanged to test -- fitting it on test and reporting test
would be circular, which is the whole reason this script keeps the two splits apart.

Priors come from the TRAINING split only.

    python logit_adjust.py --valid data_sft/csa-valid.txt --test data_sft/csa-test.txt
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaConfig

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from sklearn.metrics import f1_score, precision_recall_fscore_support   # noqa: E402

from agent import PPDPP                                    # noqa: E402
from eval_planner import Args, load_examples               # noqa: E402
from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())
IDX = {l: i for i, l in enumerate(LABELS)}


def probs_for(model, tok, ex, max_len, device):
    out = []
    for e in ex:
        ids = []
        for s in e['prefix'][::-1]:
            enc = tok.encode(s)
            if len(ids) + len(enc) > max_len:
                break
            ids = enc[1:] + ids
        ids = enc[:1] + ids
        with torch.no_grad():
            t = torch.tensor([ids[-max_len + 1:]]).long().to(device)
            lg = model.classifier(model.dropout(model.policy(t)[1]))
            out.append(torch.softmax(lg, dim=1)[0].cpu().numpy())
    return np.vstack(out)


def apply_tau(P, logprior, tau):
    return (np.log(np.maximum(P, 1e-12)) - tau * logprior).argmax(1)


def report(y, yp, title):
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, yp, labels=range(len(LABELS)), zero_division=0)
    acc = float((y == yp).mean())
    print('\n  %s' % title)
    print('  %-9s %8s %8s %8s %8s' % ('label', 'prec', 'recall', 'F1', 'support'))
    for i, l in enumerate(LABELS):
        print('  %-9s %8.3f %8.3f %8.3f %8d' % (l, pr[i], rc[i], f1[i], sup[i]))
    print('  accuracy %.4f   macro F1 %.4f' % (acc, f1_score(y, yp, average='macro',
                                                             zero_division=0)))
    return acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='sft/csa/roberta/best_checkpoint')
    p.add_argument('--train', default='data_sft/csa-train.txt')
    p.add_argument('--valid', default='data_sft/csa-valid.txt')
    p.add_argument('--test', default='data_sft/csa-test.txt')
    p.add_argument('--out', default='logs/logit_adjust.json')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    cli = p.parse_args()

    # priors from TRAIN only
    cnt = collections.Counter()
    for line in open(cli.train, encoding='utf-8'):
        for t in json.loads(line)['dialog']:
            if t.get('strategy') in IDX:
                cnt[t['strategy']] += 1
    ntr = sum(cnt.values())
    prior = np.array([cnt[l] / ntr for l in LABELS])
    logprior = np.log(prior)
    print('training priors: %s' % {l: round(prior[i], 4) for i, l in enumerate(LABELS)})
    print('tau=1 shifts followup vs ask by %.2f nats (a %.1fx odds boost)'
          % (logprior[IDX['ask']] - logprior[IDX['followup']],
             prior[IDX['ask']] / prior[IDX['followup']]))

    args = Args()
    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    cfg = RobertaConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = PPDPP(args, cfg, tok)
    model.load_state_dict(torch.load(os.path.join(cli.checkpoint, 'pytorch_model.bin'),
                                     map_location='cpu', weights_only=True))
    model.to(cli.device).eval()

    ev, et = load_examples(cli.valid), load_examples(cli.test)
    Pv = probs_for(model, tok, ev, args.max_seq_length, cli.device)
    Pt = probs_for(model, tok, et, args.max_seq_length, cli.device)
    yv = np.array([IDX[e['label']] for e in ev])
    yt = np.array([IDX[e['label']] for e in et])
    print('\nvalidation %d turns, test %d turns' % (len(yv), len(yt)))

    # ------------------------------------------------ sweep tau on VALIDATION
    print('\n' + '=' * 72)
    print('TAU SWEEP -- fitted on VALIDATION only')
    print('=' * 72)
    print('  %6s %9s %9s %11s %9s %9s' %
          ('tau', 'acc', 'macroF1', 'followupF1', 'shareF1', 'decideF1'))
    taus = np.round(np.arange(0, 1.51, 0.05), 2)
    rows = []
    for tau in taus:
        yp = apply_tau(Pv, logprior, tau)
        _, _, f1, _ = precision_recall_fscore_support(yv, yp, labels=range(len(LABELS)),
                                                      zero_division=0)
        rows.append({'tau': float(tau), 'acc': float((yv == yp).mean()),
                     'macro_f1': float(f1_score(yv, yp, average='macro', zero_division=0)),
                     'followup_f1': float(f1[IDX['followup']]),
                     'share_f1': float(f1[IDX['share']]),
                     'decide_f1': float(f1[IDX['decide']])})
        r = rows[-1]
        mark = ''
        print('  %6.2f %9.4f %9.4f %11.4f %9.4f %9.4f%s'
              % (tau, r['acc'], r['macro_f1'], r['followup_f1'], r['share_f1'],
                 r['decide_f1'], mark))

    best_macro = max(rows, key=lambda r: r['macro_f1'])
    best_fu = max(rows, key=lambda r: r['followup_f1'])
    best_acc = max(rows, key=lambda r: r['acc'])
    print('\n  selected by macro F1 on valid : tau = %.2f (macro F1 %.4f, acc %.4f)'
          % (best_macro['tau'], best_macro['macro_f1'], best_macro['acc']))
    print('  (for reference) best followup  : tau = %.2f (followup F1 %.4f, acc %.4f)'
          % (best_fu['tau'], best_fu['followup_f1'], best_fu['acc']))
    print('  (for reference) best accuracy  : tau = %.2f (acc %.4f)'
          % (best_acc['tau'], best_acc['acc']))

    tau = best_macro['tau']

    # ------------------------------------------------------- apply to TEST
    print('\n' + '=' * 72)
    print('TEST -- tau = %.2f applied unchanged from validation' % tau)
    print('=' * 72)
    yp0 = Pt.argmax(1)
    ypa = apply_tau(Pt, logprior, tau)
    a0, f0 = report(yt, yp0, 'BEFORE (tau = 0, the reported Stage-1 model)')
    a1, f1a = report(yt, ypa, 'AFTER  (tau = %.2f)' % tau)

    print('\n  %-14s %10s %10s %9s' % ('', 'before', 'after', 'delta'))
    print('  %-14s %10.4f %10.4f %+9.4f' % ('accuracy', a0, a1, a1 - a0))
    m0 = f1_score(yt, yp0, average='macro', zero_division=0)
    m1 = f1_score(yt, ypa, average='macro', zero_division=0)
    print('  %-14s %10.4f %10.4f %+9.4f' % ('macro F1', m0, m1, m1 - m0))
    for i, l in enumerate(LABELS):
        print('  %-14s %10.4f %10.4f %+9.4f' % ('F1 ' + l, f0[i], f1a[i], f1a[i] - f0[i]))

    print('\n  predicted distribution before: %s'
          % {LABELS[i]: int((yp0 == i).sum()) for i in range(len(LABELS))})
    print('  predicted distribution after : %s'
          % {LABELS[i]: int((ypa == i).sum()) for i in range(len(LABELS))})

    json.dump({'tau_selected': tau, 'sweep_valid': rows,
               'priors': {l: float(prior[i]) for i, l in enumerate(LABELS)},
               'test_before': {'accuracy': a0, 'macro_f1': float(m0),
                               'per_class_f1': {l: float(f0[i]) for i, l in enumerate(LABELS)}},
               'test_after': {'accuracy': a1, 'macro_f1': float(m1),
                              'per_class_f1': {l: float(f1a[i]) for i, l in enumerate(LABELS)}}},
              open(cli.out, 'w', encoding='utf-8'), indent=1)
    print('\nwrote %s' % cli.out)


if __name__ == '__main__':
    main()
