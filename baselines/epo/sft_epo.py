"""STAGE 2 -- SFT warm-start for LLM_s.

Vanilla EPO treats this as optional and reports pure RL doing better. That finding is at
~2050 SOTOPIA episodes; here the whole RL budget is 700, and it cannot be spent teaching
the model to emit a well-formed "tau: sigma" line. So the warm-start is mandatory, and
the pure-RL arm is kept as an ablation rather than the default.

Loss is on the COMPLETION tokens only. Every prompt token is masked to -100; getting
that wrong trains the model to reproduce scenario descriptions, which is the quietest
way this port fails.

    python sft_epo.py --train data/strategies-train.jsonl \
                      --valid data/strategies-valid.jsonl --epochs 3
"""
import argparse
import collections
import json
import math
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
import prompt_epo as pe                              # noqa: E402
from strategist import EPOStrategist                 # noqa: E402

IGNORE = -100


def build_example(tok, case, row, max_len):
    """-> (input_ids, labels) with the prompt masked out."""
    msgs = pe.strategist_messages(case, row['prefix'])
    # compat.render_chat, not apply_chat_template directly: this prompt must be
    # byte-identical to what EPOStrategist.build_prompt produces at rollout, or the
    # warm-start trains on a distribution the policy never sees.
    prompt = compat.render_chat(tok, msgs)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(row['target'] + tok.eos_token, add_special_tokens=False).input_ids

    # Truncate the PROMPT from the left, never the completion: the tail of the prompt
    # is the most recent dialogue, which is what the strategy responds to.
    room = max_len - len(c_ids)
    if room < 16:
        c_ids = c_ids[:max_len - 16]
        room = 16
    p_ids = p_ids[-room:]

    ids = p_ids + c_ids
    labels = [IGNORE] * len(p_ids) + list(c_ids)
    return ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default=os.path.join(config.DATA, 'strategies-train.jsonl'))
    p.add_argument('--valid', default=os.path.join(config.DATA, 'strategies-valid.jsonl'))
    p.add_argument('--out', default=os.path.join(config.CKPT, 'sft'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--max_len', type=int, default=1536)
    p.add_argument('--class_balance', default='sqrt_inverse',
                   choices=['none', 'sqrt_inverse', 'inverse'],
                   help='the 7:1 ask:followup ratio drove PPDPP followup F1 to 0.000')
    p.add_argument('--seed', type=int, default=0)
    cli = p.parse_args()

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    cfg = config.Defaults
    cfg.lr = cli.lr
    strat = EPOStrategist(cfg)
    tok = strat.tokenizer
    cases = config.case_index()

    def load(path):
        if not os.path.exists(path):
            return []
        return [json.loads(l) for l in open(path, encoding='utf-8')]

    train_rows = [r for r in load(cli.train) if r['uid'] in cases]
    valid_rows = [r for r in load(cli.valid) if r['uid'] in cases]
    if not train_rows:
        raise SystemExit('no training rows -- run make_strategies.py first')

    counts = collections.Counter(r['tau'] for r in train_rows)
    print('train %d rows  %s' % (len(train_rows), dict(sorted(counts.items()))))

    # Sampling weights rather than loss weights: with batch size 1 the two are
    # equivalent in expectation, and resampling keeps every gradient a real example.
    if cli.class_balance == 'none':
        weights = [1.0] * len(train_rows)
    else:
        pw = {}
        for k, v in counts.items():
            pw[k] = (1.0 / math.sqrt(v)) if cli.class_balance == 'sqrt_inverse' else (1.0 / v)
        norm = sum(pw.values()) / len(pw)
        pw = {k: v / norm for k, v in pw.items()}
        print('class weights: %s' % {k: round(v, 3) for k, v in sorted(pw.items())})
        weights = [pw[r['tau']] for r in train_rows]

    n_steps = max(1, (len(train_rows) * cli.epochs) // cli.accum)
    strat.set_schedule(n_steps)
    print('optimizer steps: %d' % n_steps)

    dev = strat.policy.device
    rng = random.Random(cli.seed)
    seen = 0
    for ep in range(cli.epochs):
        order = rng.choices(range(len(train_rows)), weights=weights, k=len(train_rows))
        strat.policy.train()
        run_loss, n = 0.0, 0
        for j, idx in enumerate(order):
            row = train_rows[idx]
            ids, labels = build_example(tok, cases[row['uid']], row, cli.max_len)
            t_ids = torch.tensor([ids], device=dev)
            t_lab = torch.tensor([labels], device=dev)
            out = strat.policy(input_ids=t_ids, labels=t_lab)
            (out.loss / cli.accum).backward()
            run_loss += float(out.loss); n += 1; seen += 1
            if (j + 1) % cli.accum == 0:
                torch.nn.utils.clip_grad_norm_(strat.params, 1.0)
                strat.optimizer.step()
                if strat.scheduler is not None:
                    strat.scheduler.step()
                strat.optimizer.zero_grad(set_to_none=True)
            if (j + 1) % 100 == 0:
                print('  ep%d %4d/%d  loss %.4f' % (ep, j + 1, len(order), run_loss / n),
                      flush=True)
                run_loss, n = 0.0, 0

        if valid_rows:
            strat.policy.eval()
            tot, cnt, tag_ok = 0.0, 0, 0
            with torch.no_grad():
                for row in valid_rows:
                    ids, labels = build_example(tok, cases[row['uid']], row, cli.max_len)
                    out = strat.policy(input_ids=torch.tensor([ids], device=dev),
                                       labels=torch.tensor([labels], device=dev))
                    tot += float(out.loss); cnt += 1
                # cheap behavioural check: does it emit a parseable tag at all?
                for row in valid_rows[:40]:
                    text, _ = strat.act(cases[row['uid']], row['prefix'], is_test=True)
                    _tau, _sig, ok = pe.parse_strategy(text)
                    tag_ok += int(ok)
            print('epoch %d  valid loss %.4f  parseable tag %d/40'
                  % (ep, tot / max(1, cnt), tag_ok), flush=True)

    strat.save(cli.out)
    with open(os.path.join(cli.out, 'sft_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'train_rows': len(train_rows), 'epochs': cli.epochs, 'lr': cli.lr,
                   'accum': cli.accum, 'class_balance': cli.class_balance,
                   'by_act': dict(counts), 'examples_seen': seen}, f, indent=1)


if __name__ == '__main__':
    main()
