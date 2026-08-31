"""STAGE 2.2 -- distil the attributed labels into an online reward model.

Input  : (state, action) -- transcript up to turn t, plus the utterance
Label  : the attributed scalar r_t
Loss   : MSE (Eq. 5)

Why a reward model exists at all: the attributor sees the FINISHED episode, the RM sees
only the prefix. Training it turns hindsight labels into something that can score a fresh
utterance online, which is the only way GRPO can evaluate a candidate that has never been
generated before.

Note this arm is optional on CSA in a way it is not on SOTOPIA: the attribution is free
and deterministic here, so `--reward_source lookahead` in train_grpo.py skips the RM
entirely. Training it anyway is what lets you MEASURE the distillation error.

Selection is on within-state ranking, not MSE alone. GRPO standardises rewards inside
each group, so what matters is whether the RM orders candidates at the same state
correctly -- a model with lower MSE and worse ranking is worse for our purposes.

    python train_rm.py --data data/rm-train.jsonl --epochs 4
"""
import argparse
import collections
import json
import os
import random
import sys

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import compat                                        # noqa: E402
import config                                        # noqa: E402
import data_csa                                      # noqa: E402
import paths                                         # noqa: E402
import prompts_sr as P                               # noqa: E402
from models_sr import REWARD, RewardView, SharedBackbone   # noqa: E402


def prompt_text(rm, row, cases):
    case = cases[row['uid']]
    dm_name = next(a['name'] for a in case['agents']
                   if a['agent_id'] == case['decision_maker'])
    return compat.render_chat(rm.tokenizer, P.to_chat(row['prompt_messages'], dm_name))


def pair_ranking_accuracy(preds, rows):
    """Over pairs of turns from the SAME episode with different labels, how often does the
    RM order them correctly. This is the quantity GRPO actually consumes."""
    by_uid = collections.defaultdict(list)
    for p, r in zip(preds, rows):
        by_uid[r['uid']].append((p, r['label']))
    ok = tot = 0
    for items in by_uid.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (pi, li), (pj, lj) = items[i], items[j]
                if abs(li - lj) < 1e-9:
                    continue
                tot += 1
                ok += int((pi - pj) * (li - lj) > 0)
    return (ok / tot if tot else float('nan')), tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(paths.DATA, 'rm-train.jsonl'))
    p.add_argument('--out', default=os.path.join(paths.CKPT, 'rm'))
    p.add_argument('--epochs', type=int, default=config.Defaults.rm_epochs)
    p.add_argument('--lr', type=float, default=config.Defaults.rm_lr)
    p.add_argument('--accum', type=int, default=config.Defaults.rm_accum)
    p.add_argument('--max_len', type=int, default=config.Defaults.max_len)
    p.add_argument('--holdout', type=float, default=0.1,
                   help='fraction of SCENARIOS held out; splitting by turn would put '
                        'near-duplicates on both sides')
    p.add_argument('--device', default=None)
    p.add_argument('--grad_checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    cli = p.parse_args()

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    cfg = config.Defaults
    if cli.device:
        cfg.agent_device = cli.device

    if not os.path.exists(cli.data):
        raise SystemExit('missing %s -- run make_rm_data.py first' % cli.data)
    rows = [json.loads(l) for l in open(cli.data, encoding='utf-8')]
    cases = data_csa.case_index()
    rows = [r for r in rows if r['uid'] in cases]

    uids = sorted({r['uid'] for r in rows})
    random.Random(cli.seed).shuffle(uids)
    n_hold = max(1, int(cli.holdout * len(uids)))
    hold = set(uids[:n_hold])
    train = [r for r in rows if r['uid'] not in hold]
    valid = [r for r in rows if r['uid'] in hold]
    print('rows %d  train %d  valid %d  (%d scenarios held out)'
          % (len(rows), len(train), len(valid), len(hold)))
    nz = sum(1 for r in train if r['label'] > 1e-9)
    print('non-zero labels in train: %d/%d (%.1f%%)' % (nz, len(train),
                                                        100 * nz / max(1, len(train))))
    if nz / max(1, len(train)) < 0.1:
        print('WARNING: labels are almost all zero; the RM will collapse to predicting 0.')

    bb = SharedBackbone(cfg, adapters=[REWARD],
                        grad_checkpointing=cli.grad_checkpointing)
    rm = RewardView(bb, cfg=cfg)
    opt = torch.optim.AdamW(rm.params, lr=cli.lr, eps=1e-6, weight_decay=0.0)
    steps = max(1, (len(train) * cli.epochs) // cli.accum)
    sched = compat.linear_schedule(opt, int(0.05 * steps), steps)
    print('optimizer steps: %d' % steps)

    def evaluate(rows_):
        rm.eval()
        preds, se = [], 0.0
        with torch.no_grad():
            for r in rows_:
                ids = rm.encode(prompt_text(rm, r, cases), r['completion'], cli.max_len)
                v = float(rm.forward_ids(ids).item())
                preds.append(v)
                se += (v - r['label']) ** 2
        acc, npair = pair_ranking_accuracy(preds, rows_)
        base = sum(x['label'] for x in rows_) / max(1, len(rows_))
        base_se = sum((base - x['label']) ** 2 for x in rows_)
        return se / max(1, len(rows_)), acc, npair, base_se / max(1, len(rows_))

    rng = random.Random(cli.seed)
    best = None
    for ep in range(cli.epochs):
        rm.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        run, n = 0.0, 0
        for j, idx in enumerate(order):
            r = train[idx]
            ids = rm.encode(prompt_text(rm, r, cases), r['completion'], cli.max_len)
            pred = rm.forward_ids(ids)
            loss = torch.nn.functional.mse_loss(
                pred.float(), torch.tensor([r['label']], device=rm.device,
                                           dtype=torch.float))
            (loss / cli.accum).backward()
            run += float(loss); n += 1
            if (j + 1) % cli.accum == 0:
                torch.nn.utils.clip_grad_norm_(rm.params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if (j + 1) % 200 == 0:
                print('  ep%d %4d/%d  mse %.5f' % (ep, j + 1, len(order), run / n),
                      flush=True)
                run, n = 0.0, 0

        mse, acc, npair, base = evaluate(valid) if valid else (float('nan'),) * 4
        print('epoch %d  valid mse %.5f (predict-the-mean %.5f)  pair-rank %.3f over %d '
              'pairs' % (ep, mse, base, acc, npair), flush=True)
        if valid and (best is None or acc > best[0]):
            best = (acc, ep)
            rm.save(cli.out)
            print('  new best by ranking accuracy -> saved')

    if not valid:
        rm.save(cli.out)
    with open(os.path.join(cli.out, 'rm_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'rows': len(rows), 'train': len(train), 'valid': len(valid),
                   'epochs': cli.epochs, 'lr': cli.lr,
                   'best_epoch': best[1] if best else None,
                   'best_pair_rank': best[0] if best else None,
                   'holdout_scenarios': sorted(hold)}, f, indent=1)


if __name__ == '__main__':
    main()
