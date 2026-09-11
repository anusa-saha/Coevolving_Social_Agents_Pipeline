"""Shared benchmark contract for every CSA baseline.

The split, the disclosure detector and the verifier are the *measuring instrument*. If
two baselines score with different copies of them, their numbers are not comparable --
so there is exactly one copy here and every arm imports it.

What lives here is only what MUST be identical across arms:

    data_csa    the scenario split, re-derived from a fixed procedure
    detectors   lexical disclosure / leak / addressing rules, threshold frozen at 0.35
    verifier    deterministic scoring of a settlement against the dataset's checks
    paths       where the dataset is, and where each arm writes its own outputs
    compat      version shims for transformers / peft / accelerate

Everything else -- prompts, environments, training loops -- is per-baseline and stays in
that baseline's folder, because that is the part each method is entitled to change.
"""
__all__ = ['compat', 'data_csa', 'detectors', 'paths', 'verifier']
