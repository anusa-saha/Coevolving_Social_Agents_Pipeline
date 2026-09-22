"""ttpo.py - token-level selection, after TTPO (arXiv:2608.27448).

WHAT WE TAKE, AND WHY IT APPLIES HERE

TTPO is a test-time-training method: no labels, so it manufactures a pseudo-label by
majority vote and then has to survive that label being wrong ~85% of the time. We are not
in that setting - env.TerminalVerifier gives us an exact reward - so the majority-vote
machinery is irrelevant to us and we do not take it.

Two other things in the paper apply directly, and both address failures already measured
in train_V1 and check/OPD_check_one:

  1. THE ASYMMETRIC SPLIT (paper 3.2). Do not apply the same signal to every rollout in a
     group. Distil the ones that did well; penalise the ones that did badly. TTPO's reason
     is pseudo-label noise; ours is different and stronger. Our privileged teacher is a
     legitimate target for a rollout that ALREADY did well - the record it is looking at
     is one the student itself produced - and a much worse one for a rollout that failed,
     where "what would a teacher who knows all the secrets have done" is exactly the
     signal that collapsed onlyOPD (reveal rate 0.832 -> 0.134, success -> 0.000).
     main.py routes above-average rollouts into distillation and below-average ones into
     RL for that reason.

  2. TOKEN-LEVEL SELECTION (paper 3.3, 3.4, Table 3). This is the part that maps onto a
     failure we can point at:

     * DISTILLATION WEIGHTING. check/OPD_check_one's insight KL fell from 0.076 to 0.055
       over its last ~90 optimiser steps and bought exactly nothing - checks went 0.351
       (batches 21-30) to 0.135 (batches 81-90). It was distilling tokens the student had
       already mastered: JSON punctuation, schema slot names, the fixed skeleton of a slot
       table. TTPO's w(t) drives the weight on precisely those positions to ~0.

           w(t) = H(t) + D(t) - H(t) * D(t)          (Soft-OR, both min-max normalised)

       High when the student is uncertain (H) OR when it disagrees with the teacher (D);
       ~0 only when it is confident AND already aligned. Both terms are free: we already
       compute the student log-softmax and the teacher divergence.

     * NEGATIVE-SAMPLE MASKING. A settlement span is ~191 tokens, most of them JSON
       boilerplate the model emits at logp ~= 0. Applying one negative advantage uniformly
       across them penalises punctuation as hard as the wrong field value - TTPO calls
       this the False Penalties on Negative Samples problem, and notes that in a
       group-relative objective there are no positive-advantage gradients on those same
       tokens to cancel it. The mask keeps the top half by

           s(t) = -log pi(y_t) * (1 - H(t))

       i.e. tokens the model produced CONFIDENTLY (low entropy) and yet were UNLIKELY
       (high surprisal): confident errors. Locally-correct high-probability tokens score
       low and are excluded, which is the whole point.

     Per the paper's ablation, masking applies to NEGATIVE samples only. Positive rollouts
     are reinforced across their whole span - there is no collateral damage to limit when
     the update direction is already the one you want.

Everything here is pure tensor arithmetic on quantities the callers already have. Nothing
in this file costs an extra forward pass.
"""

from __future__ import annotations

import torch


def _minmax(x, eps: float = 1e-8):
    """Per-sample min-max normalisation to [0, 1]. A span whose values are all equal
    normalises to zeros, which is the right answer: nothing in it stands out."""
    lo = x.min()
    hi = x.max()
    rng = hi - lo
    if float(rng) < eps:
        return torch.zeros_like(x)
    return (x - lo) / rng


def entropy_from_logprobs(logprobs, chunk: int = 64):
    """H(t) = -sum_v p(v) log p(v) over the vocab, from a [L, V] log-softmax.

    Chunked over the sequence for the same reason the KL is: a [320, 152k] float32
    intermediate is ~195 MB.
    """
    outs = []
    for i in range(0, logprobs.shape[0], chunk):
        lp = logprobs[i:i + chunk]
        outs.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(outs, dim=0)


def distil_weights(entropy, divergence):
    """TTPO eq. 4:  w(t) = H_hat(t) + D_hat(t) - H_hat(t) * D_hat(t).

    A Soft-OR of "the student is unsure here" and "the student and teacher disagree here".
    Returns a detached [L] tensor in [0, 1]; the caller multiplies its per-token KL by it.

    Weights are NOT renormalised to sum to 1. Renormalising would restore the very
    property this removes - a span of fully-mastered tokens would be scaled back up to
    the same total loss as a span of hard ones. Letting the total shrink is the intent:
    an already-learned position should contribute little, not be re-inflated.
    """
    with torch.no_grad():
        h = _minmax(entropy.float())
        d = _minmax(divergence.float())
        return h + d - h * d


def confident_error_mask(logprobs, entropy, keep: float = 0.5):
    """TTPO eqs. 6-7: keep the top `keep` fraction of tokens by

        s(t) = -log pi(y_t) * (1 - H_hat(t))

    Returns a detached [L] float mask of 1.0/0.0.

    The design choice that matters is that -log pi is UNNORMALISED, so it dominates the
    ranking: high-probability tokens (the JSON skeleton, the schema key names, the closing
    braces) can never enter the top half however their entropy falls. What survives is the
    genuinely anomalous - the model confidently emitted something unlikely, which on this
    task is a wrong field value rather than a mis-typed delimiter.

    A span of 1-2 tokens (a route decision) is returned unmasked: there is no
    "concentrate the penalty" to do when there is only one place for it to go.
    """
    n = logprobs.shape[0]
    if n <= 2:
        return torch.ones_like(logprobs)
    with torch.no_grad():
        s = (-logprobs.float()) * (1.0 - _minmax(entropy.float()))
        k = max(1, int(round(n * keep)))
        thresh = torch.topk(s, k).values.min()
        return (s >= thresh).float()
