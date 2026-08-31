"""STAGE 2.1 -- behaviour cloning for the chair policy.

Input  : the chair's view plus the transcript up to a turn
Label  : the utterance that was actually said, token by token
Loss   : cross-entropy on the COMPLETION only; every prompt token is masked to -100

Getting the mask wrong trains the model to reproduce scenario descriptions, which is the
quietest way this whole pipeline fails, so the boundary is asserted rather than trusted.

The prompt is truncated from the LEFT: the tail is the most recent dialogue, which is
what the utterance responds to.

    python train_sft.py --episodes data/episodes-train.jsonl --epochs 3
"""
import argparse
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
from models_sr import POLICY, PolicyView, SharedBackbone   # noqa: E402

IGNORE = -100


def build_examples(episodes, cases, min_dca=None):
    out = []
    for e in episodes:
        case = cases.get(e['uid'])
        if not case:
            continue
        if min_dca is not None and (e.get('score') or {}).get('dca', 0.0) < min_dca:
            continue
        for i, t in enumerate(e['dialog']):
            if t.get('speaker') != 'sys':
                continue
            prefix = [{'role': x['role'], 'content': x['content']}
                      for x in e['dialog'][:i]]
            out.append({'uid': e['uid'],
                        'messages': P.chair_messages(case, prefix, settling=False),
                        'target': t['content'],
                        'dca': (e.get('score') or {}).get('dca', 0.0)})
    return out


def encode(tok, ex, dm_name, max_len):
    prompt = compat.render_chat(tok, P.to_chat(ex['messages'], dm_name))
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(ex['target'] + tok.eos_token, add_special_tokens=False).input_ids
    room = max_len - len(c_ids)
    if room < 16:
        c_ids = c_ids[:max_len - 16]
        room = 16
    p_ids = p_ids[-room:]
    ids = p_ids + c_ids
    labels = [IGNORE] * len(p_ids) + list(c_ids)
    assert len(ids) == len(labels)
    assert all(x == IGNORE for x in labels[:len(p_ids)]), 'prompt leaked into the loss'
    assert any(x != IGNORE for x in labels), 'nothing to learn from'
    return ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', nargs='+',
                   default=[os.path.join(paths.DATA, 'episodes-train.jsonl')])
    p.add_argument('--valid', default=os.path.join(paths.DATA, 'episodes-valid.jsonl'))
    p.add_argument('--out', default=os.path.join(paths.CKPT, 'sft'))
    p.add_argument('--epochs', type=int, default=config.Defaults.sft_epochs)
    p.add_argument('--lr', type=float, default=config.Defaults.sft_lr)
    p.add_argument('--accum', type=int, default=config.Defaults.sft_accum)
    p.add_argument('--max_len', type=int, default=config.Defaults.max_len)
    p.add_argument('--min_dca', type=float, default=None,
                   help='clone only episodes at or above this dca. The demonstrations '
                        'here are mediocre; filtering harder is usually right.')
    p.add_argument('--device', default=None)
    p.add_argument('--grad_checkpointing', action='store_true',
                   help='~6x less activation memory for ~30%% slower steps')
    p.add_argument('--seed', type=int, default=0)
    cli = p.parse_args()

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    cfg = config.Defaults
    if cli.device:
        cfg.agent_device = cli.device

    cases = data_csa.case_index()
    eps = []
    for path in cli.episodes:
        if not os.path.exists(path):
            raise SystemExit('missing %s -- run collect_episodes.py first' % path)
        eps += [json.loads(l) for l in open(path, encoding='utf-8')]
    train = build_examples(eps, cases, cli.min_dca)
    valid = []
    if os.path.exists(cli.valid):
        valid = build_examples([json.loads(l) for l in open(cli.valid, encoding='utf-8')],
                               cases, cli.min_dca)
    if not train:
        raise SystemExit('no training examples (min_dca too strict?)')
    print('BC examples: train %d  valid %d  (from %d episodes)'
          % (len(train), len(valid), len(eps)))
    print('mean dca of cloned episodes: %.3f'
          % (sum(x['dca'] for x in train) / len(train)))

    bb = SharedBackbone(cfg, adapters=[POLICY],
                        grad_checkpointing=cli.grad_checkpointing)
    policy = PolicyView(bb)
    tok = policy.tokenizer
    opt = torch.optim.AdamW(policy.params, lr=cli.lr, eps=1e-6, weight_decay=0.0)
    steps = max(1, (len(train) * cli.epochs) // cli.accum)
    sched = compat.linear_schedule(opt, int(0.05 * steps), steps)
    print('optimizer steps: %d' % steps)

    rng = random.Random(cli.seed)
    for ep in range(cli.epochs):
        policy.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        run, n = 0.0, 0
        for j, idx in enumerate(order):
            ex = train[idx]
            case = cases[ex['uid']]
            dm_name = next(a['name'] for a in case['agents']
                           if a['agent_id'] == case['decision_maker'])
            ids, labels = encode(tok, ex, dm_name, cli.max_len)
            out = policy.forward_lm(torch.tensor([ids], device=policy.device),
                                    torch.tensor([labels], device=policy.device))
            (out.loss / cli.accum).backward()
            run += float(out.loss); n += 1
            if (j + 1) % cli.accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if (j + 1) % 100 == 0:
                print('  ep%d %4d/%d  loss %.4f' % (ep, j + 1, len(order), run / n),
                      flush=True)
                run, n = 0.0, 0

        if valid:
            policy.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for ex in valid[:200]:
                    case = cases[ex['uid']]
                    dm_name = next(a['name'] for a in case['agents']
                                   if a['agent_id'] == case['decision_maker'])
                    ids, labels = encode(tok, ex, dm_name, cli.max_len)
                    o = policy.forward_lm(
                        torch.tensor([ids], device=policy.device),
                        torch.tensor([labels], device=policy.device))
                    tot += float(o.loss); cnt += 1
            print('epoch %d  valid loss %.4f' % (ep, tot / max(1, cnt)), flush=True)

    policy.save(cli.out)
    with open(os.path.join(cli.out, 'sft_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'examples': len(train), 'epochs': cli.epochs, 'lr': cli.lr,
                   'accum': cli.accum, 'min_dca': cli.min_dca}, f, indent=1)


if __name__ == '__main__':
    main()
