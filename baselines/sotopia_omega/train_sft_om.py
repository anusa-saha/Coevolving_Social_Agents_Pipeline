"""STAGE 2 -- fine-tune the student on the Omega corpus.

Input  : the chair's view plus the transcript up to a turn
Label  : the utterance that was actually said, token by token
Loss   : cross-entropy on the COMPLETION only; every prompt token is masked to -100

Getting the mask wrong trains the model to reproduce scenario descriptions, which is the
quietest way this whole pipeline fails, so the boundary is asserted rather than trusted.

The prompt is truncated from the LEFT: the tail is the most recent dialogue, which is
what the utterance responds to.

    python train_sft_om.py --episodes data/episodes-train.jsonl --epochs 3
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

from csa_core import compat as compat                                        # noqa: E402
import config                                        # noqa: E402
from csa_core import data_csa as data_csa                                      # noqa: E402
import paths                                         # noqa: E402
import prompts_om as P                               # noqa: E402
from student import Student                          # noqa: E402

IGNORE = -100


def build_examples(episodes, cases, min_dca=None, mode_filter='all'):
    """(chair prompt, chair utterance) pairs.

    `mode_filter` is the design decision Omega leaves open. Every episode contains turns
    from before the stall and after it, and the pre-stall turns are the ones that CAUSED
    the deadlock -- cloning the whole trajectory teaches the failure alongside the
    recovery. Omega's default is to keep everything and record the mode; 'slow' isolates
    the intervention. The mode array makes this a filter rather than a re-generation.

    Note the scaffolding itself is never a label: only the utterance the chair actually
    spoke is cloned, so the student learns to produce it without the three reasoning
    stages. That is where the distillation happens.
    """
    out = []
    for e in episodes:
        case = cases.get(e['uid'])
        if not case:
            continue
        if min_dca is not None and (e.get('score') or {}).get('dca', 0.0) < min_dca:
            continue
        modes = e.get('modes') or []
        chair_i = -1
        for i, t in enumerate(e['dialog']):
            if t.get('speaker') != 'sys':
                continue
            chair_i += 1
            mode = modes[chair_i] if chair_i < len(modes) else 'fast'
            if mode_filter != 'all' and mode != mode_filter:
                continue
            prefix = [{'role': x['role'], 'content': x['content']}
                      for x in e['dialog'][:i]]
            out.append({'uid': e['uid'],
                        'messages': P.chair_messages(case, prefix, settling=False),
                        'target': t['content'],
                        'mode': mode,
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
                   default=[os.path.join(paths.DATA, 'corpus-B-train.jsonl')])
    p.add_argument('--valid', default=os.path.join(paths.DATA, 'corpus-B-valid.jsonl'))
    p.add_argument('--out', default=os.path.join(paths.CKPT, 'sft'))
    p.add_argument('--epochs', type=int, default=config.Defaults.sft_epochs)
    p.add_argument('--lr', type=float, default=config.Defaults.sft_lr)
    p.add_argument('--accum', type=int, default=config.Defaults.sft_accum)
    p.add_argument('--max_len', type=int, default=config.Defaults.max_len)
    p.add_argument('--mode_filter', default='all', choices=['all', 'slow', 'fast'],
                   help="which chair turns to clone. 'all' matches Omega; 'slow' clones "
                        'only post-intervention turns and drops the ones that caused the '
                        'stall')
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
    train = build_examples(eps, cases, cli.min_dca, cli.mode_filter)
    valid = []
    if os.path.exists(cli.valid):
        valid = build_examples([json.loads(l) for l in open(cli.valid, encoding='utf-8')],
                               cases, cli.min_dca, cli.mode_filter)
    if not train:
        raise SystemExit('no training examples (min_dca too strict?)')
    print('BC examples: train %d  valid %d  (from %d episodes)'
          % (len(train), len(valid), len(eps)))
    import collections as _c
    print('mean dca of cloned episodes: %.3f'
          % (sum(x['dca'] for x in train) / len(train)))
    print('turns by mode: %s   (filter=%s)'
          % (dict(_c.Counter(x['mode'] for x in train)), cli.mode_filter))

    policy = Student(cfg, grad_checkpointing=cli.grad_checkpointing)
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
                   'accum': cli.accum, 'min_dca': cli.min_dca,
                   'mode_filter': cli.mode_filter}, f, indent=1)


if __name__ == '__main__':
    main()
