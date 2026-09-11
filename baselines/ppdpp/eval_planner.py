"""Evaluate the trained planner on held-out annotated turns.

Accuracy alone is misleading here: if one class is 40% of the data, predicting it every
time scores 40% while learning nothing. So the headline numbers are per-class precision,
recall and F1, macro-F1, the confusion matrix, and the majority-class baseline the model
must beat.

Prediction entropy is reported too. The earlier nine-act planner finished at 3.01 bits
against a 3.17 uniform maximum -- it never formed a preference at all -- so knowing
whether this one commits matters as much as whether it is right.

    python eval_planner.py --checkpoint sft/csa/roberta/best_checkpoint \
                           --data data_sft/csa-test.txt
"""
import argparse
import collections
import io
import json
import math
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
from prompt import CSAAct                                  # noqa: E402

LABELS = sorted(CSAAct.keys())


class Args:
    model_name = 'roberta'
    model_name_or_path = 'roberta-large'
    cache_dir = './hfcache'
    max_seq_length = 512
    learning_rate = 1e-6
    data_name = 'csa'


def load_examples(path):
    """Rebuild exactly what data_reader.convert_to_features would produce."""
    out = []
    for line in open(path, encoding='utf-8'):
        sample = json.loads(line)
        state = []
        for turn in sample['dialog']:
            if turn.get('speaker') == 'sys' and turn.get('strategy') and state:
                out.append({'prefix': list(state), 'label': turn['strategy'],
                            'uid': sample['uid']})
            state.append('%s: %s' % (turn['role'], turn['content']))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='sft/csa/roberta/best_checkpoint')
    p.add_argument('--data', default='data_sft/csa-test.txt')
    p.add_argument('--out', default='')
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
    print('evaluating %d held-out turns from %d scenarios\n'
          % (len(ex), len({e['uid'] for e in ex})))

    y_true, y_pred, probs_all = [], [], []
    for e in ex:
        # same truncation as agent.build_input: walk backwards to max_seq_length
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
        probs_all.append(pr)
        y_pred.append(LABELS[int(max(range(len(pr)), key=lambda i: pr[i]))])
        y_true.append(e['label'])

    n = len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / n
    support = collections.Counter(y_true)
    maj_label, maj_n = support.most_common(1)[0]
    maj_acc = maj_n / n

    print('%-10s %8s %8s %8s %8s' % ('label', 'prec', 'recall', 'F1', 'support'))
    f1s = []
    for lab in LABELS:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print('%-10s %8.3f %8.3f %8.3f %8d' % (lab, prec, rec, f1, support[lab]))
    macro = sum(f1s) / len(f1s)

    print('\naccuracy            : %.3f' % acc)
    print('macro F1            : %.3f' % macro)
    print('majority baseline   : %.3f  (always predict "%s")' % (maj_acc, maj_label))
    print('beats baseline by   : %+.3f' % (acc - maj_acc))

    # does the planner commit, or is it near-uniform?
    ent = [-sum(q * math.log(q, 2) for q in pr if q > 0) for pr in probs_all]
    print('\nmean prediction entropy: %.3f bits  (uniform = %.3f)'
          % (sum(ent) / n, math.log(len(LABELS), 2)))
    print('predicted distribution : %s'
          % dict(sorted(collections.Counter(y_pred).items())))
    print('true distribution      : %s' % dict(sorted(support.items())))

    print('\nconfusion (rows = true, cols = predicted)')
    print('%-10s %s' % ('', ' '.join('%9s' % l for l in LABELS)))
    for a in LABELS:
        row = [sum(1 for x, y in zip(y_true, y_pred) if x == a and y == b)
               for b in LABELS]
        print('%-10s %s' % (a, ' '.join('%9d' % v for v in row)))

    if cli.out:
        with open(cli.out, 'w', encoding='utf-8') as f:
            json.dump({'n': n, 'accuracy': acc, 'macro_f1': macro,
                       'majority_baseline': maj_acc, 'majority_label': maj_label,
                       'mean_entropy_bits': sum(ent) / n,
                       'per_class': {lab: {'f1': f1s[i], 'support': support[lab]}
                                     for i, lab in enumerate(LABELS)},
                       'pred_dist': dict(collections.Counter(y_pred)),
                       'true_dist': dict(support)}, f, indent=1)
        print('\nwrote %s' % cli.out)


if __name__ == '__main__':
    main()
