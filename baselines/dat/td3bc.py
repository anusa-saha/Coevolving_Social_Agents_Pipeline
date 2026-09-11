"""TD3+BC over the offline buffer. No language model is loaded by this file.

Section 5.2 picks TD3+BC (Fujimoto & Gu, 2021) and Appendix B gives the two tricks that
made it work: the residual planner architecture (implemented in planner.py) and a
weighted MSE for the Q loss. The remark in the paper is worth repeating -- the choice of
offline algorithm is not essential to DAT, and a better-tuned one would plausibly do
better. What matters is that the actor moves in R^d' and the environment is never touched
again after the buffer is collected.

TD3+BC is TD3 plus one term:

    actor loss = -lambda * Q1(s, pi(s)) + MSE(pi(s), u)      lambda = alpha / mean|Q1|

The BC term keeps the policy near the actions the buffer actually contains, which is what
makes an offline algorithm safe on ~1200 transitions: without it the actor walks into a
region of the action space the critic has never seen and the Q values there are fiction.

The action the RL stage learns is the RESIDUAL u, not the full action a. pi_phi is frozen
after self-cloning, so a = pi_phi(s) + u and u = 0 reproduces the self-cloned policy
exactly. That means the zero action is always available and always means "behave as the
unsteered model does", which is a far better anchor than an arbitrary point in R^64.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
    """(s, u, r, s', done), as float32 arrays. Small enough to hold in memory."""

    def __init__(self, s, u, r, s2, done, meta=None):
        self.s = np.asarray(s, dtype=np.float32)
        self.u = np.asarray(u, dtype=np.float32)
        self.r = np.asarray(r, dtype=np.float32).reshape(-1, 1)
        self.s2 = np.asarray(s2, dtype=np.float32)
        self.done = np.asarray(done, dtype=np.float32).reshape(-1, 1)
        self.meta = dict(meta or {})
        n = len(self.s)
        assert len(self.u) == len(self.r) == len(self.s2) == len(self.done) == n

    def __len__(self):
        return len(self.s)

    @classmethod
    def from_transitions(cls, rows, meta=None):
        if not rows:
            raise SystemExit('no transitions: the buffer would be empty')
        return cls([x['s'] for x in rows], [x['u'] for x in rows],
                   [x['r'] for x in rows], [x['s2'] for x in rows],
                   [x['done'] for x in rows], meta=meta)

    def save(self, path):
        np.savez_compressed(path, s=self.s, u=self.u, r=self.r, s2=self.s2,
                            done=self.done)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['s'], z['u'], z['r'], z['s2'], z['done'])

    def state_stats(self):
        """mu and sigma over the states actually visited -- Appendix B's normalisation."""
        return self.s.mean(0), self.s.std(0)

    def sample(self, batch, device, rng):
        idx = rng.integers(0, len(self), size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)      # noqa: E731
        return t(self.s), t(self.u), t(self.r), t(self.s2), t(self.done)

    def reward_summary(self):
        r = self.r.reshape(-1)
        return {'n': int(len(r)), 'mean': float(r.mean()), 'sd': float(r.std()),
                'min': float(r.min()), 'max': float(r.max()),
                'nonzero_frac': float((r != 0).mean())}


class Critic(nn.Module):
    """Twin Q networks over (normalised state, residual action)."""

    def __init__(self, d_state, action_dim, hidden=512):
        super().__init__()

        def head():
            return nn.Sequential(nn.Linear(d_state + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = head(), head()

    def forward(self, s, u):
        x = torch.cat([s, u], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, s, u):
        return self.q1(torch.cat([s, u], dim=-1))


class TD3BC(object):
    def __init__(self, planner, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.planner = planner.to(device)
        self.planner.freeze_clone()                  # pi_phi and W are done training
        for p in self.planner.rl_parameters():
            p.requires_grad_(True)
        if not self.planner.rl_parameters():
            raise SystemExit('this planner has no RL head; build it with residual=True')

        self.target = copy.deepcopy(self.planner).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.critic = Critic(planner.d_model, planner.action_dim,
                             hidden=cfg.planner_hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.planner.rl_parameters(), lr=cfg.td3_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.td3_lr)
        self.max_action = float(self.planner.max_action.item())
        self.it = 0

    # ------------------------------------------------------------- one update
    def train_step(self, batch):
        s, u, r, s2, done = batch
        cfg = self.cfg
        self.it += 1
        sn, s2n = self.planner.normalise(s), self.planner.normalise(s2)

        with torch.no_grad():
            noise = (torch.randn_like(u) * cfg.td3_policy_noise * self.max_action) \
                .clamp(-cfg.td3_noise_clip * self.max_action,
                       cfg.td3_noise_clip * self.max_action)
            u2 = (self.target.rl_action(s2) + noise).clamp(-self.max_action,
                                                           self.max_action)
            q1t, q2t = self.critic_target(s2n, u2)
            target_q = r + (1.0 - done) * cfg.td3_gamma * torch.min(q1t, q2t)

        q1, q2 = self.critic(sn, u)
        # Appendix B's "weighted mean squared error": the paper does not say what the
        # weights are, so this is our reading -- transitions that carry a non-zero reward
        # are upweighted. On CSA most turns earn nothing, and an unweighted MSE lets the
        # critic reach a low loss by predicting zero everywhere. td3_reward_weight = 1.0
        # turns it off.
        w = 1.0 + (cfg.td3_reward_weight - 1.0) * (r != 0).float()
        critic_loss = (w * ((q1 - target_q) ** 2 + (q2 - target_q) ** 2)).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        cgn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        out = {'critic_loss': float(critic_loss.detach()),
               'q_mean': float(q1.mean().detach()),
               'target_q_mean': float(target_q.mean()),
               'critic_grad_norm': float(cgn)}

        if self.it % cfg.td3_policy_freq == 0:
            pi = self.planner.rl_action(s)
            q = self.critic.Q1(sn, pi)
            lam = cfg.td3_alpha / q.abs().mean().detach().clamp_min(1e-6)
            bc = F.mse_loss(pi, u)
            actor_loss = -lam * q.mean() + bc

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            agn = torch.nn.utils.clip_grad_norm_(self.planner.rl_parameters(), 10.0)
            self.actor_opt.step()
            self._polyak()
            out.update({'actor_loss': float(actor_loss.detach()),
                        'bc_loss': float(bc.detach()), 'lambda': float(lam),
                        'grad_norm': float(agn)})
        # `loss` is the key analysis/compute_extended_metrics.py reads for every arm.
        out['loss'] = out['critic_loss']
        out.setdefault('grad_norm', out['critic_grad_norm'])
        return out

    def _polyak(self):
        tau = self.cfg.td3_tau
        with torch.no_grad():
            for a, b in zip(self.critic.parameters(), self.critic_target.parameters()):
                b.mul_(1 - tau).add_(tau * a)
            for a, b in zip(self.planner.parameters(), self.target.parameters()):
                b.mul_(1 - tau).add_(tau * a.to(b.dtype))

    # ------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def policy_shift(self, buffer, n=256, seed=0):
        """How far the learned residual has moved from the buffer's actions.

        Zero means TD3+BC returned the self-cloned policy and the DAT arm will be the
        selfclone arm with extra steps. Large means the actor has left the data, which on
        an offline algorithm is where Q values stop meaning anything.
        """
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(buffer), size=min(n, len(buffer)))
        s = torch.as_tensor(buffer.s[idx], device=self.device)
        u = torch.as_tensor(buffer.u[idx], device=self.device)
        pi = self.planner.rl_action(s)
        return {'mean_l2_to_buffer': float((pi - u).norm(dim=-1).mean()),
                'mean_action_norm': float(pi.norm(dim=-1).mean()),
                'buffer_action_norm': float(u.norm(dim=-1).mean())}
