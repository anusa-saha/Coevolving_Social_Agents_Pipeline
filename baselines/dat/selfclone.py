"""STAGE 1 -- self-cloning: train pi_phi and W from scratch (Section 5.1).

Two steps, both here because neither is useful alone:

  collect   run the UNSTEERED chair over training scenarios and keep every
            (prompt, utterance) pair it produced. This is the paper's corpus
            {p_1, q_1, ... p_N, q_N} for M dialogues.
  train     fit (phi, W) so that the steered model reproduces those utterances:

                L = - sum_i sum_j  log f_theta( q_j^i | pi_phi(g_theta(e_j^i)) W || e_j^i )

            Equation 5, with the language model frozen. The gradient reaches the planner
            only through the two prefix embeddings.

The paper is explicit that this brings no performance gain -- it exists so that stage 2
starts from a policy that behaves like the base model, which converts the problem from
"discover language" to "adjust a 64-dimensional control", and that is the whole point of
DAT. The number to look at afterwards is the gap between the steered and the unsteered
NLL of the same held-out utterances: it should be small, and it is printed at the end.

    python selfclone.py                       # collect, then train
    python selfclone.py --collect_only
    python selfclone.py --train_only
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                # noqa: BLE001
        pass

import paths                                         # noqa: E402
import config                                        # noqa: E402
from env_dat import DATEnv                           # noqa: E402
from planner import DATPlanner, init_up_from_embeddings   # noqa: E402
from steering import SteeredLM                       # noqa: E402

CORPUS = os.path.join(paths.DATA, 'selfclone-corpus.jsonl')
STATES = os.path.join(paths.DATA, 'selfclone-states.npz')
CKPT = os.path.join(paths.CKPT, 'selfclone.pt')


# ------------------------------------------------------------------ collect
def collect(cfg, lm, cases, n_episodes, out_path):
    """Unsteered rollouts, kept for their chair turns.

    Sampled rather than greedy: the clone target is the base policy's DISTRIBUTION, and
    a corpus of greedy argmaxes teaches the planner to reproduce one point of it. The
    evaluation runs are still greedy, as in every other arm.
    """
    env = DATEnv(cfg, lm=lm, planner=None)
    env.greedy = False
    env.collect_clone_pairs = True
    rows, t0 = [], time.time()
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(n_episodes):
            case = cases[i % len(cases)]
            env.reset(case, arm='unsteered')
            done = 0
            while not done:
                _c, done = env.step()
            for pair in env.clone_pairs:
                if pair['target'].strip():
                    rows.append(pair)
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            f.flush()
            if (i + 1) % 10 == 0:
                print('  collected %d/%d episodes, %d utterances, %.1f min'
                      % (i + 1, n_episodes, len(rows), (time.time() - t0) / 60),
                      flush=True)
    print('corpus -> %s  (%d utterances)' % (out_path, len(rows)))
    return rows


def load_corpus(path):
    rows = []
    if not os.path.exists(path):
        raise SystemExit('no corpus at %s; run without --train_only first' % path)
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------ train
def encode_pair(lm, cfg, row):
    tok = lm.tokenizer
    p = tok([row['prompt']], return_tensors='pt', add_special_tokens=False).input_ids
    if cfg.selfclone_max_ctx and p.shape[1] > cfg.selfclone_max_ctx:
        p = p[:, -cfg.selfclone_max_ctx:]
    t = tok([row['target']], return_tensors='pt', add_special_tokens=False).input_ids
    if t.shape[1] > cfg.selfclone_max_target:
        t = t[:, :cfg.selfclone_max_target]
    return p.to(lm.device), t.to(lm.device)


def cache_states(lm, cfg, rows, path=None):
    """g_theta(e) for every prompt, computed once.

    The clone loop needs the state on every epoch and it never changes -- the prompts are
    fixed strings. Recomputing it would double the forward passes for nothing.
    """
    out = []
    t0 = time.time()
    for i, row in enumerate(rows):
        p, _t = encode_pair(lm, cfg, row)
        out.append(lm.state(p).cpu().numpy())
        if (i + 1) % 50 == 0:
            print('  states %d/%d  %.1f min' % (i + 1, len(rows),
                                                (time.time() - t0) / 60), flush=True)
    S = np.stack(out).astype(np.float32)
    if path:
        np.savez_compressed(path, states=S)
    return S


def train(cfg, lm, rows, states, planner, epochs, lr, accum, log_path):
    planner.train()
    opt = torch.optim.AdamW(planner.clone_parameters(), lr=lr, weight_decay=0.0)
    if cfg.selfclone_grad_checkpointing:
        lm.enable_gradient_checkpointing()

    order = list(range(len(rows)))
    hist = open(log_path, 'a', encoding='utf-8')
    step, t0 = 0, time.time()
    for ep in range(epochs):
        random.shuffle(order)
        run, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, i in enumerate(order):
            p, t = encode_pair(lm, cfg, rows[i])
            if t.shape[1] == 0:
                continue
            s = torch.as_tensor(states[i], device=planner_device(planner))
            prefix = planner.prefix(planner.base_action(s))
            loss = lm.clone_loss(p, t, prefix) / accum
            loss.backward()
            run += float(loss.detach()) * accum
            seen += 1
            if (k + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(planner.clone_parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                hist.write(json.dumps({'stage': 'selfclone', 'epoch': ep, 'step': step,
                                       'loss': run / max(1, seen),
                                       'grad_norm': float(gn),
                                       'elapsed_s': round(time.time() - t0, 1)}) + '\n')
                hist.flush()
                if step % 10 == 0:
                    print('  epoch %d step %d  clone NLL %.4f  %.1f min'
                          % (ep, step, run / max(1, seen), (time.time() - t0) / 60),
                          flush=True)
                run, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
    hist.close()
    if cfg.selfclone_grad_checkpointing:
        lm.disable_gradient_checkpointing()
    planner.eval()
    return planner


def planner_device(planner):
    return next(planner.parameters()).device


@torch.no_grad()
def fidelity(lm, cfg, rows, states, planner, n=32):
    """Steered vs unsteered NLL on the same utterances. Stage 1's only real claim."""
    steered, plain = [], []
    for i in range(min(n, len(rows))):
        p, t = encode_pair(lm, cfg, rows[i])
        if t.shape[1] == 0:
            continue
        s = torch.as_tensor(states[i], device=planner_device(planner))
        prefix = planner.prefix(planner.base_action(s))
        steered.append(float(lm.clone_loss(p, t, prefix)))
        plain.append(float(lm.clone_loss(p, t, None)))
    if not steered:
        return {}
    return {'n': len(steered),
            'nll_steered': sum(steered) / len(steered),
            'nll_unsteered': sum(plain) / len(plain),
            'gap': (sum(steered) - sum(plain)) / len(steered)}


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=config.Defaults.selfclone_episodes)
    p.add_argument('--epochs', type=int, default=config.Defaults.selfclone_epochs)
    p.add_argument('--lr', type=float, default=config.Defaults.selfclone_lr)
    p.add_argument('--accum', type=int, default=config.Defaults.selfclone_accum)
    p.add_argument('--corpus', default=CORPUS)
    p.add_argument('--out', default=CKPT)
    p.add_argument('--device', default=config.Defaults.device)
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    p.add_argument('--collect_only', action='store_true')
    p.add_argument('--train_only', action='store_true')
    p.add_argument('--pca_up', action='store_true',
                   help="Appendix A: initialise W from the embedding matrix's principal "
                        'components instead of at random')
    cli = p.parse_args()

    cfg = config.Defaults
    cfg.device = cli.device
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if config.preflight():
        raise SystemExit('\nfix the blocking problems above, then re-run')

    lm = SteeredLM(cfg)
    print('d_model = %d, action_dim = %d, prefix tokens = %d'
          % (lm.d_model, cfg.action_dim, cfg.n_prefix))

    if not cli.train_only:
        cases = config.load_csa('train')
        collect(cfg, lm, cases, cli.episodes, cli.corpus)
        if cli.collect_only:
            return

    rows = load_corpus(cli.corpus)
    print('corpus: %d utterances from %d scenarios'
          % (len(rows), len({r['uid'] for r in rows})))

    if os.path.exists(STATES):
        S = np.load(STATES)['states']
        if len(S) != len(rows):
            print('cached states are stale (%d vs %d); recomputing' % (len(S), len(rows)))
            S = cache_states(lm, cfg, rows, STATES)
    else:
        S = cache_states(lm, cfg, rows, STATES)

    planner = DATPlanner(lm.d_model, action_dim=cfg.action_dim, n_prefix=cfg.n_prefix,
                         hidden=cfg.planner_hidden, layers=cfg.planner_layers,
                         residual=cfg.residual_rl).to(lm.device)
    planner.set_state_stats(S.mean(0), S.std(0))
    if cli.pca_up:
        init_up_from_embeddings(planner, lm.embed.weight)
        print('W initialised from the embedding PCA (Appendix A)')

    log_path = os.path.join(paths.LOGS, 'dat-selfclone-history.jsonl')
    before = fidelity(lm, cfg, rows, S, planner)
    print('before training: %s' % json.dumps(before))

    train(cfg, lm, rows, S, planner, cli.epochs, cli.lr, cli.accum, log_path)

    after = fidelity(lm, cfg, rows, S, planner)
    print('after training : %s' % json.dumps(after))

    # The residual's range. pi_phi's outputs have whatever scale the clone loss gave
    # them, and TD3's target smoothing needs a max_action in the same units -- so it is
    # read off the trained policy rather than guessed.
    with torch.no_grad():
        A = planner.base_action(torch.as_tensor(S, device=lm.device)).float()
    planner.set_max_action(float(A.std(0).mean()))
    print('max_action (mean per-dim sd of pi_phi over the corpus): %.4f'
          % float(planner.max_action))

    planner.save(cli.out)
    with open(os.path.join(paths.LOGS, 'dat-selfclone-summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'episodes': cli.episodes, 'utterances': len(rows),
                   'epochs': cli.epochs, 'lr': cli.lr, 'accum': cli.accum,
                   'fidelity_before': before, 'fidelity_after': after,
                   'max_action': float(planner.max_action),
                   'meta': planner.meta()}, f, indent=1)
    print('planner -> %s' % cli.out)


if __name__ == '__main__':
    main()
