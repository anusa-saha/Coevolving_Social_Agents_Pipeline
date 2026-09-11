"""Exhaustive evaluation of the trained planner on a held-out split.

eval_planner.py reports the handful of numbers needed to decide whether the model beats
its baseline. This reports everything a reviewer might ask for, grouped so the important
claims come first and the diagnostics follow.

The distinction that matters throughout: accuracy is dominated by the majority class, so
every headline figure is paired with a class-balanced counterpart (balanced accuracy,
macro F1, kappa, MCC) and with an interval, because 135 test turns is a small sample and
a bare point estimate invites over-reading.

    python full_metrics.py --checkpoint sft/csa/roberta/best_checkpoint \
                           --data data_sft/csa-test.txt --tag test --figure
"""
import argparse
import collections
import json
import math
import os
import random
import sys

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaConfig

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)                # noqa: E402

from agent import PPDPP                                    # noqa: E402
from eval_planner import Args                              # noqa: E402
from prompt import CSAAct                                  # noqa: E402
from utils import load_dataset                             # noqa: E402

LABELS = sorted(CSAAct.keys())
IDX = {l: i for i, l in enumerate(LABELS)}


def load_examples_meta(path, cases):
    """Same example construction as eval_planner, plus metadata for the breakdowns."""
    out = []
    for line in open(path, encoding='utf-8'):
        s = json.loads(line)
        state, pos = [], 0
        for turn in s['dialog']:
            if turn.get('speaker') == 'sys' and turn.get('strategy') and state:
                c = cases.get(s['uid'], {})
                out.append({'prefix': list(state), 'label': turn['strategy'],
                            'uid': s['uid'], 'pos': pos,
                            'domain': c.get('domain') or s['uid'].split('::')[0],
                            'num_agents': c.get('num_agents'),
                            'utt': ' '.join(turn['content'].split())[:120]})
                pos += 1
            state.append('%s: %s' % (turn['role'], turn['content']))
    return out


def predict(model, tok, ex, max_len, device):
    probs = []
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
            logits = model.classifier(model.dropout(model.policy(t)[1]))
            probs.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
    return np.vstack(probs)


def h(title):
    print('\n' + '=' * 76)
    print(title)
    print('=' * 76)


def ci(vals, lo=2.5, hi=97.5):
    return np.percentile(vals, lo), np.percentile(vals, hi)


def mcnemar(a_ok, b_ok):
    """Compare two prediction vectors on the same items. Returns (b, c, chi2, p)."""
    b = int(np.sum(a_ok & ~b_ok))          # model right, other wrong
    c = int(np.sum(~a_ok & b_ok))          # model wrong, other right
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)          # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2))              # chi2 sf, 1 dof
    return b, c, chi2, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='sft/csa/roberta/best_checkpoint')
    p.add_argument('--data', default='data_sft/csa-test.txt')
    p.add_argument('--tag', default='test')
    p.add_argument('--out', default='')
    p.add_argument('--figure', action='store_true')
    p.add_argument('--boot', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    cli = p.parse_args()
    rng = np.random.default_rng(cli.seed)

    cases = {}
    for sp in ('train', 'valid', 'test'):
        for c in load_dataset('csa')[sp]:
            cases[c['uid']] = c

    args = Args()
    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    cfg = RobertaConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = PPDPP(args, cfg, tok)
    model.load_state_dict(torch.load(os.path.join(cli.checkpoint, 'pytorch_model.bin'),
                                     map_location='cpu', weights_only=True))
    model.to(cli.device).eval()

    ex = load_examples_meta(cli.data, cases)
    P = predict(model, tok, ex, args.max_seq_length, cli.device)
    y = np.array([IDX[e['label']] for e in ex])
    yp = P.argmax(1)
    n, K = len(y), len(LABELS)
    Y1 = np.eye(K)[y]

    print('checkpoint : %s' % cli.checkpoint)
    print('split      : %s (%s)' % (cli.tag, cli.data))
    print('examples   : %d turns from %d scenarios' % (n, len({e['uid'] for e in ex})))

    # ---------------------------------------------------------------- headline
    h('1. HEADLINE')
    acc = float((y == yp).mean())
    bal = balanced_accuracy_score(y, yp)
    maj_i = collections.Counter(y).most_common(1)[0][0]
    maj_acc = float((y == maj_i).mean())
    print('  accuracy                 %.4f' % acc)
    print('  balanced accuracy        %.4f   (mean per-class recall; imbalance-proof)' % bal)
    print('  macro F1                 %.4f' % f1_score(y, yp, average='macro', zero_division=0))
    print('  micro F1                 %.4f   (= accuracy for single-label)'
          % f1_score(y, yp, average='micro', zero_division=0))
    print('  weighted F1              %.4f' % f1_score(y, yp, average='weighted', zero_division=0))
    print()
    print('  majority baseline        %.4f   (always "%s")' % (maj_acc, LABELS[maj_i]))
    print('  improvement over it      %+.4f' % (acc - maj_acc))
    print('  error reduction          %.1f%%' % (100 * (acc - maj_acc) / (1 - maj_acc)))

    # --------------------------------------------------------------- per class
    h('2. PER-CLASS')
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=range(K), zero_division=0)
    cm = confusion_matrix(y, yp, labels=range(K))
    print('  %-9s %8s %8s %8s %8s %8s %8s' %
          ('label', 'prec', 'recall', 'F1', 'support', 'spec', 'pred_n'))
    for i, l in enumerate(LABELS):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if tn + fp else 0.0
        print('  %-9s %8.3f %8.3f %8.3f %8d %8.3f %8d'
              % (l, pr[i], rc[i], f1[i], sup[i], spec, int((yp == i).sum())))
    print('\n  prevalence  true: %s' % {l: int((y == i).sum()) for i, l in enumerate(LABELS)})
    print('  prevalence  pred: %s' % {l: int((yp == i).sum()) for i, l in enumerate(LABELS)})

    # --------------------------------------------------------- chance-adjusted
    h('3. CHANCE-CORRECTED AGREEMENT')
    kap = cohen_kappa_score(y, yp)
    kapw = cohen_kappa_score(y, yp, weights='linear')
    mcc = matthews_corrcoef(y, yp)
    print('  Cohen kappa              %.4f   (0 = chance, 1 = perfect)' % kap)
    print('  Cohen kappa (linear w)   %.4f' % kapw)
    print('  Matthews corr coef       %.4f   (balanced; robust to skew)' % mcc)
    print('\n  kappa reading: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,')
    print('                 0.61-0.80 substantial, >0.80 almost perfect (Landis-Koch)')

    # ---------------------------------------------------------------- ranking
    h('4. RANKING / TOP-K')
    order = np.argsort(-P, axis=1)
    rank_of_true = np.array([int(np.where(order[i] == y[i])[0][0]) + 1 for i in range(n)])
    for k in (1, 2, 3):
        print('  top-%d accuracy            %.4f' % (k, float((rank_of_true <= k).mean())))
    print('  mean reciprocal rank     %.4f' % float((1 / rank_of_true).mean()))
    print('  mean rank of true class  %.3f  (1 = perfect, %.1f = random)'
          % (rank_of_true.mean(), (K + 1) / 2))
    print('\n  rank distribution: %s'
          % {int(k): int(v) for k, v in zip(*np.unique(rank_of_true, return_counts=True))})

    # ---------------------------------------------------------- probabilistic
    h('5. PROBABILISTIC QUALITY')
    ll = log_loss(y, P, labels=list(range(K)))
    brier = float(np.mean(np.sum((P - Y1) ** 2, axis=1)))
    print('  log loss (NLL)           %.4f   (uniform = %.4f, lower better)'
          % (ll, math.log(K)))
    print('  Brier score (multiclass) %.4f   (uniform = %.4f, lower better)'
          % (brier, 1 - 1.0 / K))
    try:
        auc_m = roc_auc_score(Y1, P, average='macro', multi_class='ovr')
        auc_w = roc_auc_score(Y1, P, average='weighted', multi_class='ovr')
        print('  ROC-AUC macro (OvR)      %.4f   (0.5 = chance)' % auc_m)
        print('  ROC-AUC weighted (OvR)   %.4f' % auc_w)
    except Exception as e:
        auc_m = auc_w = float('nan')
        print('  ROC-AUC unavailable: %s' % e)
    print()
    print('  %-9s %10s %10s' % ('label', 'ROC-AUC', 'avg prec'))
    per_auc = {}
    for i, l in enumerate(LABELS):
        yi = (y == i).astype(int)
        if yi.sum() in (0, n):
            print('  %-9s %10s %10s' % (l, 'n/a', 'n/a'))
            continue
        a = roc_auc_score(yi, P[:, i])
        ap = average_precision_score(yi, P[:, i])
        per_auc[l] = {'roc_auc': float(a), 'avg_precision': float(ap),
                      'base_rate': float(yi.mean())}
        print('  %-9s %10.4f %10.4f   (base rate %.3f)' % (l, a, ap, yi.mean()))
    print('\n  NOTE: a class can have high ROC-AUC and zero F1 -- AUC measures whether the')
    print('  score RANKS positives above negatives, F1 measures whether argmax picks it.')

    # ---------------------------------------------------------- calibration
    h('6. CALIBRATION')
    conf = P.max(1)
    correct = (y == yp)
    nb = 10
    edges = np.linspace(0, 1, nb + 1)
    ece = mce = 0.0
    print('  %-14s %6s %9s %9s %8s' % ('confidence bin', 'n', 'conf', 'acc', 'gap'))
    for b in range(nb):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if not m.any():
            continue
        c_, a_ = conf[m].mean(), correct[m].mean()
        gap = abs(a_ - c_)
        ece += m.mean() * gap
        mce = max(mce, gap)
        print('  (%.1f, %.1f]     %6d %9.3f %9.3f %8.3f'
              % (edges[b], edges[b + 1], m.sum(), c_, a_, gap))
    print('\n  ECE (expected calib error) %.4f   lower is better' % ece)
    print('  MCE (max calib error)      %.4f' % mce)
    print('  mean confidence            %.4f' % conf.mean())
    print('  mean accuracy              %.4f' % correct.mean())
    over = conf.mean() - correct.mean()
    print('  overconfidence             %+.4f   (%s)'
          % (over, 'overconfident' if over > 0 else 'underconfident'))

    # ------------------------------------------------------------- confidence
    h('7. DECISIVENESS')
    ent = -np.sum(np.where(P > 0, P * np.log2(np.maximum(P, 1e-12)), 0), axis=1)
    print('  mean entropy             %.4f bits  (uniform = %.4f)' % (ent.mean(), math.log2(K)))
    print('  median entropy           %.4f bits' % np.median(ent))
    print('  entropy p10 / p90        %.4f / %.4f' % (np.percentile(ent, 10),
                                                      np.percentile(ent, 90)))
    print('  mean max-probability     %.4f' % conf.mean())
    for t in (0.5, 0.7, 0.9):
        m = conf >= t
        print('  turns with conf >= %.1f    %4d (%4.1f%%)   accuracy there %.3f'
              % (t, m.sum(), 100 * m.mean(), correct[m].mean() if m.any() else float('nan')))

    # ------------------------------------------------- risk / coverage curve
    h('8. RISK-COVERAGE (selective prediction)')
    print('  If the planner deferred its least-confident turns, how good is the rest?\n')
    idx = np.argsort(-conf)
    print('  %-10s %8s %10s' % ('coverage', 'n', 'accuracy'))
    for cov in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        k = max(1, int(round(cov * n)))
        print('  %-10.0f%% %8d %10.4f' % (100 * cov, k, correct[idx[:k]].mean()))

    # ------------------------------------------------------------- bootstrap
    h('9. BOOTSTRAP 95%% CONFIDENCE INTERVALS (%d resamples)' % cli.boot)
    stats = collections.defaultdict(list)
    for _ in range(cli.boot):
        s = rng.integers(0, n, n)
        ys, yps = y[s], yp[s]
        stats['accuracy'].append((ys == yps).mean())
        stats['macro_f1'].append(f1_score(ys, yps, average='macro', zero_division=0))
        stats['balanced_acc'].append(balanced_accuracy_score(ys, yps))
        stats['kappa'].append(cohen_kappa_score(ys, yps))
        stats['mcc'].append(matthews_corrcoef(ys, yps))
        stats['delta_vs_majority'].append((ys == yps).mean() - (ys == maj_i).mean())
    print('  %-20s %9s   %-22s %8s' % ('metric', 'point', '95% CI', 'width'))
    point = {'accuracy': acc, 'macro_f1': f1_score(y, yp, average='macro', zero_division=0),
             'balanced_acc': bal, 'kappa': kap, 'mcc': mcc,
             'delta_vs_majority': acc - maj_acc}
    boot_ci = {}
    for k_ in ('accuracy', 'balanced_acc', 'macro_f1', 'kappa', 'mcc', 'delta_vs_majority'):
        lo, hi = ci(stats[k_])
        boot_ci[k_] = [float(lo), float(hi)]
        print('  %-20s %9.4f   [%7.4f, %7.4f] %8.4f' % (k_, point[k_], lo, hi, hi - lo))
    d_lo, d_hi = boot_ci['delta_vs_majority']
    print('\n  the improvement over the majority baseline %s zero'
          % ('EXCLUDES' if d_lo > 0 else 'INCLUDES'))
    print('  -> %s' % ('significant at the 5%% level' if d_lo > 0 else
                       'NOT significant: the gain could be sampling noise'))

    # ---------------------------------------------------------- significance
    h('10. SIGNIFICANCE vs BASELINES (McNemar, paired on the same turns)')
    rs = random.Random(cli.seed)
    prior = np.array([(y == i).mean() for i in range(K)])
    baselines = {
        'majority': np.full(n, maj_i),
        'uniform random': np.array([rs.randrange(K) for _ in range(n)]),
        'prior-matched random': np.array(rs.choices(range(K), weights=prior, k=n)),
    }
    print('  %-22s %9s %6s %6s %8s %10s' % ('baseline', 'its acc', 'win', 'loss', 'chi2', 'p'))
    sig = {}
    for name, bp in baselines.items():
        b_ok = (bp == y)
        w, l_, chi2, pv = mcnemar(correct, b_ok)
        sig[name] = {'baseline_acc': float(b_ok.mean()), 'win': w, 'loss': l_,
                     'chi2': float(chi2), 'p': float(pv)}
        print('  %-22s %9.4f %6d %6d %8.2f %10.2e %s'
              % (name, b_ok.mean(), w, l_, chi2, pv, '***' if pv < 0.001 else
                 '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'))
    print('\n  win  = turns the model got right and the baseline did not')
    print('  loss = turns the baseline got right and the model did not')

    # ------------------------------------------------------------- confusion
    h('11. CONFUSION MATRIX')
    print('  rows = true, cols = predicted (raw counts)')
    print('  %-10s %s %8s' % ('', ' '.join('%9s' % l for l in LABELS), 'total'))
    for i, l in enumerate(LABELS):
        print('  %-10s %s %8d' % (l, ' '.join('%9d' % v for v in cm[i]), cm[i].sum()))
    print('  %-10s %s %8d' % ('total', ' '.join('%9d' % v for v in cm.sum(0)), cm.sum()))
    print('\n  row-normalised (recall per row)')
    print('  %-10s %s' % ('', ' '.join('%9s' % l for l in LABELS)))
    for i, l in enumerate(LABELS):
        r = cm[i] / max(1, cm[i].sum())
        print('  %-10s %s' % (l, ' '.join('%8.1f%%' % (100 * v) for v in r)))

    h('12. MOST COSTLY CONFUSIONS')
    pairs = [((LABELS[i], LABELS[j]), int(cm[i, j]))
             for i in range(K) for j in range(K) if i != j and cm[i, j]]
    pairs.sort(key=lambda x: -x[1])
    tot_err = int((y != yp).sum())
    print('  %d errors in total\n' % tot_err)
    for (a, b), v in pairs[:8]:
        print('  true %-9s -> pred %-9s %4d   %4.1f%% of all errors'
              % (a, b, v, 100 * v / max(1, tot_err)))

    # ------------------------------------------------------------ breakdowns
    h('13. BREAKDOWNS')
    def group(keyfn, title):
        g = collections.defaultdict(list)
        for i, e in enumerate(ex):
            g[keyfn(e)].append(i)
        print('\n  --- by %s ---' % title)
        print('  %-22s %6s %9s %9s %9s' % (title, 'n', 'accuracy', 'macroF1', 'maj base'))
        for k_ in sorted(g, key=lambda z: str(z)):
            ii = np.array(g[k_])
            if len(ii) < 3:
                continue
            sub_maj = collections.Counter(y[ii]).most_common(1)[0][0]
            print('  %-22s %6d %9.4f %9.4f %9.4f'
                  % (str(k_), len(ii), (y[ii] == yp[ii]).mean(),
                     f1_score(y[ii], yp[ii], average='macro', zero_division=0),
                     (y[ii] == sub_maj).mean()))
    group(lambda e: e['domain'], 'domain')
    group(lambda e: e['num_agents'], 'num_agents')
    group(lambda e: e['pos'], 'chair-turn position')
    group(lambda e: e['label'], 'true label')

    per_scen = collections.defaultdict(list)
    for i, e in enumerate(ex):
        per_scen[e['uid']].append(correct[i])
    sa = np.array([np.mean(v) for v in per_scen.values()])
    print('\n  --- per-scenario accuracy ---')
    print('  scenarios %d   mean %.4f   sd %.4f   min %.3f   max %.3f'
          % (len(sa), sa.mean(), sa.std(), sa.min(), sa.max()))
    print('  scenarios at 100%%: %d    below 50%%: %d'
          % (int((sa == 1).sum()), int((sa < 0.5).sum())))

    # --------------------------------------------------------------- errors
    h('14. HIGH-CONFIDENCE ERRORS (worst failures)')
    err = np.where(~correct)[0]
    err = err[np.argsort(-conf[err])][:10]
    for i in err:
        print('  conf %.3f  true %-9s pred %-9s pos%d %-11s | %s'
              % (conf[i], LABELS[y[i]], LABELS[yp[i]], ex[i]['pos'],
                 ex[i]['domain'][:11], ex[i]['utt'][:78]))

    # ----------------------------------------------------------------- dump
    res = {
        'checkpoint': cli.checkpoint, 'split': cli.tag, 'n': n,
        'accuracy': acc, 'balanced_accuracy': float(bal),
        'macro_f1': float(f1_score(y, yp, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, yp, average='weighted', zero_division=0)),
        'majority_baseline': maj_acc, 'delta_vs_majority': acc - maj_acc,
        'cohen_kappa': float(kap), 'mcc': float(mcc),
        'top2_accuracy': float((rank_of_true <= 2).mean()),
        'mrr': float((1 / rank_of_true).mean()),
        'log_loss': float(ll), 'brier': brier,
        'roc_auc_macro': float(auc_m), 'ece': float(ece), 'mce': float(mce),
        'mean_entropy_bits': float(ent.mean()), 'mean_confidence': float(conf.mean()),
        'per_class': {l: {'precision': float(pr[i]), 'recall': float(rc[i]),
                          'f1': float(f1[i]), 'support': int(sup[i]),
                          **per_auc.get(l, {})} for i, l in enumerate(LABELS)},
        'bootstrap_ci95': boot_ci, 'significance': sig,
        'confusion_matrix': cm.tolist(), 'labels': LABELS,
    }
    out = cli.out or 'logs/full_metrics_%s.json' % cli.tag
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)
    print('\nwrote %s' % out)

    if cli.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 10))
        cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
        im = ax[0, 0].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        ax[0, 0].set_xticks(range(K)); ax[0, 0].set_xticklabels(LABELS, rotation=45)
        ax[0, 0].set_yticks(range(K)); ax[0, 0].set_yticklabels(LABELS)
        ax[0, 0].set_xlabel('predicted'); ax[0, 0].set_ylabel('true')
        ax[0, 0].set_title('Confusion (row-normalised)')
        for i in range(K):
            for j in range(K):
                ax[0, 0].text(j, i, '%d\n%.0f%%' % (cm[i, j], 100 * cmn[i, j]),
                              ha='center', va='center', fontsize=8,
                              color='white' if cmn[i, j] > 0.5 else 'black')
        fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

        xs = np.arange(K)
        ax[0, 1].bar(xs - 0.25, pr, 0.25, label='precision')
        ax[0, 1].bar(xs, rc, 0.25, label='recall')
        ax[0, 1].bar(xs + 0.25, f1, 0.25, label='F1')
        ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels(LABELS)
        ax[0, 1].set_ylim(0, 1); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3, axis='y')
        ax[0, 1].set_title('Per-class scores (n=%d)' % n)
        for i, s_ in enumerate(sup):
            ax[0, 1].text(i, 0.95, 'n=%d' % s_, ha='center', fontsize=8)

        bx, by, bn = [], [], []
        for b in range(nb):
            m = (conf > edges[b]) & (conf <= edges[b + 1])
            if m.any():
                bx.append(conf[m].mean()); by.append(correct[m].mean()); bn.append(m.sum())
        ax[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
        ax[1, 0].plot(bx, by, 'o-', label='model')
        for x_, y_, n_ in zip(bx, by, bn):
            ax[1, 0].annotate(str(n_), (x_, y_), textcoords='offset points',
                              xytext=(4, -10), fontsize=7)
        ax[1, 0].set_xlabel('confidence'); ax[1, 0].set_ylabel('accuracy')
        ax[1, 0].set_title('Reliability (ECE = %.3f)' % ece)
        ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

        covs = np.linspace(0.05, 1, 40)
        accs = [correct[idx[:max(1, int(c_ * n))]].mean() for c_ in covs]
        ax[1, 1].plot(100 * covs, accs, lw=2)
        ax[1, 1].axhline(acc, ls='--', c='grey', label='full-coverage acc %.3f' % acc)
        ax[1, 1].axhline(maj_acc, ls=':', c='red', label='majority %.3f' % maj_acc)
        ax[1, 1].set_xlabel('coverage (%)'); ax[1, 1].set_ylabel('accuracy')
        ax[1, 1].set_title('Risk-coverage'); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

        fig.suptitle('PPDPP-CSA planner (RoBERTa-large) -- %s split, %d turns'
                     % (cli.tag, n), fontsize=13)
        fig.tight_layout()
        fp = 'logs/metrics_%s.png' % cli.tag
        fig.savefig(fp, dpi=140)
        print('wrote %s' % fp)


if __name__ == '__main__':
    main()
