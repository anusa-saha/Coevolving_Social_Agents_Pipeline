"""The planner: pi_phi, the up-mapping W, and the RL residual head.

This is the only thing in the arm that has parameters. The language model never gets a
gradient -- not in stage 1, not in stage 2, not at evaluation. That is what DAT buys, and
why "language degradation under reward optimization" (the failure the paper opens with)
cannot happen here by construction: the decoder is bit-identical to the unsteered one and
the only thing the optimiser can move is a two-token continuous prefix.

Shapes, following Section 4 of the paper:

    s   = g_theta(e)          R^d          last-layer last-token hidden state
    a   = pi_phi(s)           R^d'         d' = 64, the low-dimensional action
    z   = a W                 R^(L x d)    L = 2 dialogue action tokens
    q   ~ f_theta(. | z || e)              controlled generation

W exists so RL acts in R^64 instead of R^(2 x 3584). Predicting the prefix directly is
the same function class, but it is a 7168-dimensional action space, which no offline RL
algorithm at this budget will explore.

Appendix B's residual architecture lives here rather than in the trainer: after
self-cloning, pi_phi's output distribution is whatever the clone loss made it, which is
not normal and is awkward to fine-tune. So pi_phi is FROZEN for stage 2 and a second head
reads the per-dimension-normalised state and adds to it,

    a = pi_phi(s) + pi_phi_rl((s - mu) / (sigma + eps))

with pi_phi_rl zero-initialised at its last layer, so the RL stage starts exactly at the
self-cloned policy and every action in the replay buffer is a perturbation of it.
"""
import json
import os

import torch
import torch.nn as nn


def _mlp(d_in, hidden, d_out, layers, zero_last=False):
    mods, prev = [], d_in
    for _ in range(max(1, layers)):
        mods += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    last = nn.Linear(prev, d_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


class DATPlanner(nn.Module):
    """pi_phi, W, the RL head and the state statistics, in one saveable object."""

    def __init__(self, d_model, action_dim=64, n_prefix=2, hidden=512, layers=2,
                 residual=True):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        self.n_prefix = int(n_prefix)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.residual = bool(residual)

        self.base = _mlp(self.d_model, hidden, action_dim, layers)
        # W: R^d' -> R^(L*d). No bias: the paper's W is a matrix, and a bias would hand
        # the steered agent a constant prefix that survives even when a = 0, which would
        # make the self-clone step unfalsifiable.
        self.up = nn.Linear(action_dim, self.n_prefix * self.d_model, bias=False)
        nn.init.normal_(self.up.weight, std=1.0 / (action_dim ** 0.5))
        self.rl = _mlp(self.d_model, hidden, action_dim, layers, zero_last=True) \
            if residual else None

        # Set from the collected data, not learned. Registered as buffers so they travel
        # with the checkpoint -- a planner restored without its normalisation is a
        # different function.
        self.register_buffer('state_mu', torch.zeros(self.d_model))
        self.register_buffer('state_sd', torch.ones(self.d_model))
        self.register_buffer('max_action', torch.ones(1))

    # ------------------------------------------------------------- state
    def normalise(self, s, eps=1e-3):
        return (s - self.state_mu) / (self.state_sd + eps)

    def set_state_stats(self, mu, sd):
        with torch.no_grad():
            self.state_mu.copy_(torch.as_tensor(mu, dtype=self.state_mu.dtype))
            self.state_sd.copy_(
                torch.as_tensor(sd, dtype=self.state_sd.dtype).clamp_min(1e-6))

    def set_max_action(self, x):
        with torch.no_grad():
            self.max_action.fill_(float(max(1e-6, x)))

    # ------------------------------------------------------------- actions
    def base_action(self, s):
        """pi_phi(s). The self-cloned policy; frozen once stage 1 is done."""
        return self.base(s)

    def rl_action(self, s):
        """pi_phi_rl((s - mu) / sigma), bounded so TD3's target smoothing has a range.

        Zero at initialisation, so `action` starts equal to `base_action` and the replay
        buffer's actions really are perturbations of the self-cloned policy.
        """
        if self.rl is None:
            return torch.zeros(s.shape[:-1] + (self.action_dim,), device=s.device,
                               dtype=s.dtype)
        return self.max_action * torch.tanh(self.rl(self.normalise(s)))

    def action(self, s, residual=None):
        """The action actually used for steering. `residual` overrides pi_phi_rl."""
        a = self.base_action(s)
        if residual is None:
            residual = self.rl_action(s)
        return a + residual

    # ------------------------------------------------------------- steering
    def prefix(self, a):
        """a W, reshaped into L prefix token embeddings."""
        z = self.up(a)
        return z.reshape(z.shape[:-1] + (self.n_prefix, self.d_model))

    def prefix_from_state(self, s, residual=None):
        return self.prefix(self.action(s, residual=residual))

    # ------------------------------------------------------------- stage gating
    def clone_parameters(self):
        """What stage 1 trains: pi_phi and W together, as in Equation 5."""
        return list(self.base.parameters()) + list(self.up.parameters())

    def rl_parameters(self):
        """What stage 2 trains: pi_phi_rl alone. W stays frozen so the effective action
        space remains d'-dimensional (Section 5.1, last paragraph)."""
        return list(self.rl.parameters()) if self.rl is not None else []

    def freeze_clone(self):
        for p in self.clone_parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------- io
    def meta(self):
        return {'d_model': self.d_model, 'action_dim': self.action_dim,
                'n_prefix': self.n_prefix, 'hidden': self.hidden,
                'layers': self.layers, 'residual': self.residual}

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        torch.save({'meta': self.meta(), 'state_dict': self.state_dict()}, path)
        with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.meta(), f, indent=1)
        return path

    @classmethod
    def load(cls, path, map_location='cpu'):
        try:
            blob = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:                            # torch < 2.0 has no weights_only
            blob = torch.load(path, map_location=map_location)
        obj = cls(**blob['meta'])
        obj.load_state_dict(blob['state_dict'])
        obj.eval()
        return obj


def init_up_from_embeddings(planner, embedding_weight, sample=20000, seed=0):
    """Appendix A: W's rows as the first d' principal components of the embedding matrix.

    The paper reports this as a viable alternative that removes the need for stage 1
    entirely -- the action vector is mapped into the shape of a single word embedding and
    repeated L times. Implemented because it is a handful of lines and it is the obvious
    thing to reach for when the self-clone loss will not come down; not the default,
    because the reported experiments use self-cloning.
    """
    with torch.no_grad():
        E = embedding_weight.detach().float()
        if E.shape[0] > sample:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(E.shape[0], generator=g)[:sample]
            E = E[idx]
        E = E - E.mean(0, keepdim=True)
        _u, _s, v = torch.pca_lowrank(E, q=min(planner.action_dim, E.shape[1]))
        comp = v[:, :planner.action_dim].t()                  # (d', d)
        if comp.shape[0] < planner.action_dim:                # pad if the sample was tiny
            comp = torch.cat([comp,
                              torch.zeros(planner.action_dim - comp.shape[0],
                                          comp.shape[1])], 0)
        W = comp.repeat(1, planner.n_prefix)                  # (d', L*d)
        planner.up.weight.copy_(W.t().contiguous())
    return planner
