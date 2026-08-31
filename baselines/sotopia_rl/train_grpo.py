"""STAGE 2.3 -- single-turn GRPO for the chair policy.

At each dialogue state the policy samples G candidate utterances, each is scored, and the
advantage is the group-standardised score:

    A_i = (s_i - mean(s)) / std(s)
    L   = - (1/G) sum_i A_i * mean_j log pi(c_ij)   + beta * KL(pi || pi_ref)

Three ways to score a candidate, because CSA offers something SOTOPIA does not:

  rm         the trained reward model. Faithful to the paper. One forward per candidate.
  lookahead  commit the candidate, let ONE advisor answer, read the disclosure detector,
             roll back. Deterministic and needs no reward model, but costs one advisor
             generation per candidate and only measures pooling.
  hybrid     lookahead where it is informative, RM otherwise, and the EXACT verifier
             score on settling turns -- a settlement can always be scored outright, with
             no rollout and no model.

Groups whose candidates all score identically produce no gradient. That is a real risk
here, so the fraction of collapsed groups is logged; if it is high you are paying for
generations that teach nothing.

    python train_grpo.py --adapter ckpt/sft --rm ckpt/rm --groups 175
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

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
from env_sr import SREnv                             # noqa: E402
from models_sr import POLICY, REWARD, PolicyView, RewardView, SharedBackbone  # noqa: E402
from verifier_sr import flipped_checks, score        # noqa: E402


class Scorer(object):
    def __init__(self, cfg, mode, rm=None):
        self.cfg, self.mode, self.rm = cfg, mode, rm
        self.n_lookahead = self.n_rm = self.n_exact = 0

    def __call__(self, env, prompt_text, candidates):
        exact = env.is_settling_turn()
        out = []
        for text in candidates:
            if exact:
                out.append(self._exact(env, text)); self.n_exact += 1
            elif self.mode == 'rm':
                out.append(self.rm.score(prompt_text, text)); self.n_rm += 1
            elif self.mode == 'lookahead':
                out.append(self._lookahead(env, text)); self.n_lookahead += 1
            else:                                    # hybrid
                v = self._lookahead(env, text); self.n_lookahead += 1
                if v == 0.0 and self.rm is not None:
                    v = 0.5 * self.rm.score(prompt_text, text); self.n_rm += 1
                out.append(v)
        return out

    def _exact(self, env, text):
        """A settlement can be scored outright: parse, check the schema, run the
        verifier. No rollout, no model, no approximation."""
        s = env._parse_json(text)
        if not s:
            return -1.0
        r = score(env.case, s, env.revealed, resolve=True)
        if not r['schema_valid']:
            return -0.5
        return r['dca'] - self.cfg.__dict__.get('halluc_pen', 0.5) * r['hallucinated_credit']

    def _lookahead(self, env, text):
        """Exact pooling credit: did this candidate actually make someone disclose?"""
        info = env.peek(text)
        if info['leaked']:
            return -1.0
        phi = flipped_checks(env.case) or ['_']
        return len(info['elicited_flips']) / len(phi)


def advantages(scores):
    if len(scores) < 2:
        return [0.0] * len(scores)
    mu = sum(scores) / len(scores)
    sd = statistics.pstdev(scores)
    if sd < 1e-8:
        return [0.0] * len(scores)                   # collapsed group: no gradient
    return [(s - mu) / sd for s in scores]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', default=os.path.join(paths.CKPT, 'sft'),
                   help='BC warm-start; pass "" for the pure-RL ablation')
    p.add_argument('--rm', default=os.path.join(paths.CKPT, 'rm'))
    p.add_argument('--out', default=os.path.join(paths.CKPT, 'grpo'))
    p.add_argument('--groups', type=int, default=config.Defaults.grpo_groups)
    p.add_argument('--group', type=int, default=config.Defaults.grpo_group)
    p.add_argument('--reward_source', default=config.Defaults.reward_source,
                   choices=['rm', 'lookahead', 'hybrid'])
    p.add_argument('--lr', type=float, default=config.Defaults.grpo_lr)
    p.add_argument('--kl_beta', type=float, default=config.Defaults.grpo_kl_beta)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--eval_every', type=int, default=0)
    p.add_argument('--agent_device', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--grad_checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=config.Defaults.seed)
    p.add_argument('--dry_run', type=int, default=0)
    cli = p.parse_args()

    if compat.report():
        raise SystemExit('\nfix the blocking problems above, then re-run')

    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    cfg = config.Defaults
    if cli.agent_device:
        cfg.agent_device = cli.agent_device
    if cli.device:
        cfg.agent_device = cli.device

    train_cases = data_csa.load('train')
    needs_rm = cli.reward_source in ('rm', 'hybrid')
    if needs_rm and not os.path.isdir(cli.rm):
        raise SystemExit('no reward model at %s. Train one, or use '
                         '--reward_source lookahead.' % cli.rm)
    # ONE backbone for all three roles: agent (base), policy, reward.
    bb = SharedBackbone(cfg, adapters=([POLICY, REWARD] if needs_rm else [POLICY]),
                        grad_checkpointing=cli.grad_checkpointing)
    env = SREnv(cfg, backbone=bb)
    policy = PolicyView(bb, adapter_dir=(cli.adapter or None))
    rm = RewardView(bb, adapter_dir=cli.rm, cfg=cfg) if needs_rm else None
    scorer = Scorer(cfg, cli.reward_source, rm)
    bb.set_trainable_adapter("policy")

    opt = torch.optim.AdamW(policy.params, lr=cli.lr, eps=1e-6, weight_decay=0.0)
    sched = compat.linear_schedule(opt, int(0.05 * cli.groups), cli.groups)

    tag = 'grpo-%s-seed%d' % (cli.reward_source, cli.seed)
    hist = open(os.path.join(paths.LOGS, '%s-history.jsonl' % tag), 'a', encoding='utf-8')
    collapsed = total = 0
    t0 = time.time()

    for g in range(cli.groups):
        case = random.choice(train_cases)
        env.reset(case)
        # walk to a random chair slot so training is not concentrated on turn 0
        target = random.randrange(env.max_turn)
        for _ in range(target):
            if env.step(env.chair_say())[1]:
                break
        if env.step_i > target:
            continue

        dm_name = env.names[env.dm]
        msgs = env.chair_prompt()
        prompt_text = compat.render_chat(policy.tokenizer, P.to_chat(msgs, dm_name))
        texts, comps, prompt_ids = policy.sample(
            prompt_text, n=cli.group, max_new_tokens=env.chair_budget(),
            temperature=cli.temperature)
        scores = scorer(env, prompt_text, texts)
        adv = advantages(scores)
        total += 1
        if all(a == 0.0 for a in adv):
            collapsed += 1

        if cli.dry_run:
            print('[dry %d] %s turn %d/%d' % (g, case['uid'], env.step_i, env.max_turn))
            for tx, sc in zip(texts, scores):
                print('    %+.3f  %s' % (sc, tx[:110].replace('\n', ' ')))
            if g + 1 >= cli.dry_run:
                return
            continue

        loss_val = 0.0
        opt.zero_grad(set_to_none=True)
        for comp, a in zip(comps, adv):
            if a == 0.0:
                continue
            lp = policy.logprob(prompt_ids, comp)
            loss = -(a * lp) / len(comps)
            if cli.kl_beta:
                ref = policy.ref_logprob(prompt_ids, comp)
                if ref is not None:
                    loss = loss + cli.kl_beta * (lp - ref) / len(comps)
            loss.backward()
            loss_val += float(loss.detach())
        gn = None
        if loss_val:
            gn = float(torch.nn.utils.clip_grad_norm_(policy.params,
                                                      config.Defaults.grpo_clip))
            opt.step()
        sched.step()

        hist.write(json.dumps({
            'group': g, 'uid': case['uid'], 'turn': env.step_i,
            'scores': [round(s, 4) for s in scores], 'adv': [round(a, 3) for a in adv],
            'loss': round(loss_val, 5), 'grad_norm': gn,
            'collapsed': all(a == 0.0 for a in adv),
            'elapsed_s': round(time.time() - t0, 1)}, ensure_ascii=False) + '\n')
        hist.flush()
        if (g + 1) % 10 == 0:
            print('group %3d/%d  loss %.4f  score mean %.3f sd %.3f  collapsed %d/%d '
                  '(%.0f%%)' % (g + 1, cli.groups, loss_val,
                                sum(scores) / len(scores),
                                statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                                collapsed, total, 100 * collapsed / max(1, total)),
                  flush=True)
        if cli.eval_every and (g + 1) % cli.eval_every == 0:
            policy.save(os.path.join(cli.out, tag, 'g%d' % (g + 1)))

    policy.save(os.path.join(cli.out, tag, 'final'))
    hist.close()
    summary = {'groups': cli.groups, 'group_size': cli.group,
               'reward_source': cli.reward_source,
               'collapsed_groups': collapsed, 'total_groups': total,
               'collapsed_frac': round(collapsed / max(1, total), 4),
               'calls': {'lookahead': scorer.n_lookahead, 'rm': scorer.n_rm,
                         'exact': scorer.n_exact},
               'minutes': round((time.time() - t0) / 60, 1)}
    with open(os.path.join(paths.LOGS, '%s-summary.json' % tag), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print('\n%s' % json.dumps(summary, indent=1))
    if summary['collapsed_frac'] > 0.5:
        print('\nWARNING: over half the groups produced no gradient. Raise --temperature, '
              'or switch --reward_source to something with more resolution at these '
              'states.')


if __name__ == '__main__':
    main()
