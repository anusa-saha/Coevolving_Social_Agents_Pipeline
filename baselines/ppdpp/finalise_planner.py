"""Final Stage-1 result: bootstrap-selected logit adjustment, reported once on test.

Selecting tau by a single argmax over the validation sweep is fragile -- validation is
164 turns and the macro-F1 curve is nearly flat between tau 0.6 and 1.0, so the argmax
moves with noise. Instead tau is chosen by resampling validation many times, taking the
best tau within each resample, and using the MEDIAN. That is a stable point estimate and
it still never touches test.

Test is then scored exactly once, with bootstrap intervals and paired significance tests
against both the majority baseline and the unadjusted model.

    python finalise_planner.py
"""
import argparse
import collections
import json
import math
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

from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)                # noqa: E402

from agent import PPDPP                                    # noqa: E402
from eval_planner import Args, load_examples               # noqa: E402
from logit_adjust import probs_for                         # noqa: E402
from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())
IDX = {l: i for i, l in enumerate(LABELS)}


def adjust(P, logprior, tau):
    """Return renormalised probabilities, so NLL/Brier/ECE stay meaningful."""
    z = np.log(np.maximum(P, 1e-12)) - tau * logprior
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def mcnemar(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, chi2, math.erfc(math.sqrt(chi2 / 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='sft/csa/roberta/best_checkpoint')
    p.add_argument('--train', default='data_sft/csa-train.txt')
    p.add_argument('--valid', default='data_sft/csa-valid.txt')
    p.add_argument('--test', default='data_sft/csa-test.txt')
    p.add_argument('--boot_tau', type=int, default=2000)
    p.add_argument('--boot_ci', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/final_test_results.json')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    cli = p.parse_args()
    rng = np.random.default_rng(cli.seed)

    cnt = collections.Counter()
    for line in open(cli.train, encoding='utf-8'):
        for t in json.loads(line)['dialog']:
            if t.get('strategy') in IDX:
                cnt[t['strategy']] += 1
    ntr = sum(cnt.values())
    prior = np.array([cnt[l] / ntr for l in LABELS])
    logprior = np.log(prior)

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

    # ------------------------------------------- bootstrap tau on VALIDATION
    taus = np.round(np.arange(0, 1.51, 0.05), 2)
    Yp_v = {t: adjust(Pv, logprior, t).argmax(1) for t in taus}
    picks = []
    nv = len(yv)
    for _ in range(cli.boot_tau):
        s = rng.integers(0, nv, nv)
        best_t, best_v = 0.0, -1
        for t in taus:
            v = f1_score(yv[s], Yp_v[t][s], average='macro', zero_division=0)
            if v > best_v:
                best_v, best_t = v, t
        picks.append(best_t)
    picks = np.array(picks)
    tau = float(np.median(picks))
    lo_t, hi_t = np.percentile(picks, [2.5, 97.5])

    print('=' * 78)
    print('TAU SELECTION (validation only, %d bootstrap resamples)' % cli.boot_tau)
    print('=' * 78)
    print('  median tau            %.2f' % tau)
    print('  95%% of resamples in   [%.2f, %.2f]' % (lo_t, hi_t))
    print('  mode                  %.2f' % collections.Counter(picks).most_common(1)[0][0])
    print('  fraction picking 0    %.3f   (i.e. "no adjustment is best")'
          % float((picks == 0).mean()))

    Pt_adj = adjust(Pt, logprior, tau)
    yp0, yp1 = Pt.argmax(1), Pt_adj.argmax(1)
    n, K = len(yt), len(LABELS)
    Y1 = np.eye(K)[yt]
    ok0, ok1 = (yt == yp0), (yt == yp1)

    def block(y, yp, P):
        pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=range(K),
                                                          zero_division=0)
        return {'acc': float((y == yp).mean()),
                'bal': float(balanced_accuracy_score(y, yp)),
                'macro': float(f1_score(y, yp, average='macro', zero_division=0)),
                'weighted': float(f1_score(y, yp, average='weighted', zero_division=0)),
                'kappa': float(cohen_kappa_score(y, yp)),
                'mcc': float(matthews_corrcoef(y, yp)),
                'nll': float(log_loss(y, P, labels=list(range(K)))),
                'brier': float(np.mean(np.sum((P - Y1) ** 2, axis=1))),
                'pr': pr, 'rc': rc, 'f1': f1, 'sup': sup}
    B, A = block(yt, yp0, Pt), block(yt, yp1, Pt_adj)

    maj_i = collections.Counter(yt).most_common(1)[0][0]
    maj = float((yt == maj_i).mean())

    print('\n' + '=' * 78)
    print('FINAL TEST RESULTS  --  %d turns from %d scenarios'
          % (n, len({e['uid'] for e in et})))
    print('tau = %.2f, fixed on validation and applied unchanged' % tau)
    print('=' * 78)
    print('\n  %-24s %12s %12s %10s' % ('metric', 'SFT', 'SFT+adjust', 'delta'))
    print('  ' + '-' * 62)
    for key, lab in (('acc', 'accuracy'), ('bal', 'balanced accuracy'),
                     ('macro', 'macro F1'), ('weighted', 'weighted F1'),
                     ('kappa', 'Cohen kappa'), ('mcc', 'Matthews corr'),
                     ('nll', 'log loss (lower=better)'), ('brier', 'Brier (lower=better)')):
        print('  %-24s %12.4f %12.4f %+10.4f' % (lab, B[key], A[key], A[key] - B[key]))
    print('  %-24s %12.4f %12s %10s' % ('majority baseline', maj, '-', '-'))

    print('\n  per-class F1')
    print('  %-12s %12s %12s %10s %8s' % ('label', 'SFT', 'SFT+adjust', 'delta', 'support'))
    print('  ' + '-' * 58)
    for i, l in enumerate(LABELS):
        print('  %-12s %12.4f %12.4f %+10.4f %8d'
              % (l, B['f1'][i], A['f1'][i], A['f1'][i] - B['f1'][i], B['sup'][i]))

    print('\n  adjusted model, full per-class detail')
    print('  %-12s %9s %9s %9s %9s %9s' % ('label', 'prec', 'recall', 'F1', 'support', 'ROC-AUC'))
    for i, l in enumerate(LABELS):
        yi = (yt == i).astype(int)
        auc = roc_auc_score(yi, Pt_adj[:, i]) if 0 < yi.sum() < n else float('nan')
        print('  %-12s %9.3f %9.3f %9.3f %9d %9.3f'
              % (l, A['pr'][i], A['rc'][i], A['f1'][i], A['sup'][i], auc))

    # ------------------------------------------------------------ intervals
    print('\n  bootstrap 95%% CI (%d resamples)' % cli.boot_ci)
    st = collections.defaultdict(list)
    for _ in range(cli.boot_ci):
        s = rng.integers(0, n, n)
        st['acc'].append((yt[s] == yp1[s]).mean())
        st['macro'].append(f1_score(yt[s], yp1[s], average='macro', zero_division=0))
        st['d_macro'].append(f1_score(yt[s], yp1[s], average='macro', zero_division=0)
                             - f1_score(yt[s], yp0[s], average='macro', zero_division=0))
        st['d_maj'].append((yt[s] == yp1[s]).mean() - (yt[s] == maj_i).mean())
    for k_, lab in (('acc', 'accuracy'), ('macro', 'macro F1'),
                    ('d_macro', 'macro F1 gain vs SFT'),
                    ('d_maj', 'accuracy gain vs majority')):
        lo, hi = np.percentile(st[k_], [2.5, 97.5])
        flag = '' if lo > 0 or k_ in ('acc', 'macro') else '  <- includes zero'
        print('  %-26s [%7.4f, %7.4f]%s' % (lab, lo, hi, flag))

    print('\n  paired significance (McNemar)')
    for name, other in (('vs majority baseline', (yt == maj_i)),
                        ('vs unadjusted SFT', ok0)):
        w, l_, chi2, pv = mcnemar(ok1, other)
        print('  %-26s win %3d  loss %3d  chi2 %6.2f  p = %.3e %s'
              % (name, w, l_, chi2, pv,
                 '***' if pv < 0.001 else '**' if pv < 0.01 else
                 '*' if pv < 0.05 else 'ns'))

    cm = confusion_matrix(yt, yp1, labels=range(K))
    print('\n  confusion matrix, adjusted (rows = true, cols = predicted)')
    print('  %-11s %s' % ('', ' '.join('%9s' % l for l in LABELS)))
    for i, l in enumerate(LABELS):
        print('  %-11s %s' % (l, ' '.join('%9d' % v for v in cm[i])))

    json.dump({
        'tau': tau, 'tau_ci': [float(lo_t), float(hi_t)],
        'n_test': n, 'majority_baseline': maj,
        'sft': {k: B[k] for k in ('acc', 'bal', 'macro', 'weighted', 'kappa', 'mcc',
                                  'nll', 'brier')},
        'sft_adjusted': {k: A[k] for k in ('acc', 'bal', 'macro', 'weighted', 'kappa',
                                           'mcc', 'nll', 'brier')},
        'per_class_f1': {l: {'sft': float(B['f1'][i]), 'adjusted': float(A['f1'][i]),
                             'support': int(B['sup'][i])} for i, l in enumerate(LABELS)},
        'confusion_adjusted': cm.tolist(), 'labels': LABELS,
    }, open(cli.out, 'w', encoding='utf-8'), indent=1)
    print('\nwrote %s' % cli.out)


if __name__ == '__main__':
    main()
