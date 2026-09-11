# SOTOPIA-Ω on CSA

[SOTOPIA-Ω](https://aclanthology.org/2025.acl-long.1203/) (Zhang et al., ACL 2025 Main;
[arXiv:2502.15538](https://arxiv.org/abs/2502.15538), [code](https://github.com/WYRipple/SOTOPIA-Omega))
adapted to the CSA hidden-profile benchmark.

**Data synthesis + SFT. No RL anywhere.** That makes it the fourth distinct category in
this project, and the one that speaks directly to the EPO result — where the SFT
warm-start did all the work and 700 episodes of RL degraded it from there.

## The principle: keep the mechanism, swap the protocol

Omega's thesis is not "negotiate." It is *let the expert talk, detect that it has stalled,
switch it into a structured mode, and fine-tune on what results.* The four-stage
negotiation protocol is Omega's instantiation of that intervention because SOTOPIA is a
negotiation benchmark. CSA is an elicitation benchmark, so the intervention is an
elicitation protocol.

| Omega (negotiation) | Here (elicitation) |
|---|---|
| own utility over the matters | which decision fields are still unsupported |
| guess the opponent's utility | whose **role** makes them likely to hold it |
| draft a proposal | draft the targeted question |
| confirm the proposal | put the settlement on the record |

Forcing CSA into a negotiation frame would be adapting the dataset to the method.

## Four substitutions

| Omega | Here | Why |
|---|---|---|
| stall via `get_step_score()` LLM, `≤ 7.5` | `step ≥ stall_after` **and** decisive pooling flat for `stall_patience` turns | computable, zero calls, nothing to tune |
| 4-stage negotiation | 4-stage elicitation | CSA has aligned goals; no opponent to model |
| Qwen2.5-72B / GPT-4 expert | Qwen2.5-7B (or an API expert) | 72B does not fit; see below |
| LLM-scored quality filter | `csa_core.verifier.score()` | deterministic, already built |

The scaffold's three reasoning stages are **never labels**. Only the utterance the chair
actually spoke enters the transcript and the corpus, so the student learns to produce it
without the scaffold. That is where the distillation happens.

## Three corpora, and why you want more than one

Omega's headline — *a 7B trained on the corpus beats the expert that produced it* —
confounds **strategy injection** with **the teacher being larger than the student**.

| corpus | expert | isolates |
|---|---|---|
| **A** | plain self-play (the other arms already have this) | baseline |
| **B** | Qwen2.5-7B, strategy-injected | **strategy injection alone** |
| **C** | frontier model via API, strategy-injected | injection **+** teacher strength |

B vs A is Omega's mechanism. C vs B is the distillation gradient. C vs A is the headline
and confounds both — Omega's own confound, made visible here instead of hidden.

Student is Qwen2.5-7B throughout, so the comparison to PPDPP / EPO / Sotopia-RL holds.

> **C has a data advantage no other arm has.** A win for C partly means "the frontier
> model is better than Qwen2.5-7B," which is not a finding. Run C *alongside* B, never
> instead of it.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python selftest.py
```

The split, detectors and verifier come from `csa_core/`; the stall detector, the
four-stage scaffold and the expert/student split are this arm's own. Scenarios download
from
[`anusasaha/Coevolving_Social_Agents`](https://huggingface.co/datasets/anusasaha/Coevolving_Social_Agents)
into `data/raw/` on first use. `selftest.py` verifies the rebuilt split is exactly
complete and scenario-disjoint, and that the verifier reproduces published scores on 297
episodes.

## Run the probe first

The whole method rests on one assumption: **a 7B with the scaffold produces better turns
than the same 7B without.** If that is false, corpus B is no better than plain self-play
and the SFT is pointless.

```bash
python generate_omega.py --probe 10
```

Forces every scenario through both modes and prints the chair turns side by side, then
the disclosure delta. Minutes. **If slow mode does not improve disclosure, stop** — the
script says so explicitly — and try `--expert api` before generating anything.

## Stage 1 — corpus

```bash
python generate_omega.py --split train --k 6 --keep 2          # corpus B
```

```bash
export OPENAI_API_KEY=...
python generate_omega.py --split train --expert api --expert_model <id>   # corpus C
```

`k` rollouts per scenario, ranked **within** the scenario, top `keep` kept — so hard
scenarios still contribute rather than the filter selecting only easy ones. Resumes if
interrupted. Warns if **nothing** stalled (the intervention never fired, so this is plain
self-play) or if **everything** stalled (slow mode is always on, so the adaptive part is
untested).

## Stage 2 — SFT

```bash
python train_sft_om.py --episodes data/corpus-B-train.jsonl --epochs 3
```

Loss on completion tokens only, prompt masked to `-100`, boundary asserted.

`--mode_filter {all,slow,fast}` is the design decision Omega leaves open. Every episode
contains turns from before the stall and after it, and **the pre-stall turns are the ones
that caused the deadlock** — cloning the whole trajectory teaches the failure alongside
the recovery. `all` matches Omega; `slow` isolates the intervention. Run both.

## Stage 3 — evaluate

```bash
python evaluate_om.py --adapter "" --tag base            # untrained, the floor
python evaluate_om.py --adapter ckpt/sft --tag omega-B
```

The trained chair runs with the adapter on; **advisors run on the base weights**, so the
environment is unchanged and frozen by construction. Greedy decoding.

Default `--eval_mode fast` runs **without** the scaffold — the student is supposed to have
absorbed it, and leaving it on would measure the scaffold rather than the training.
`--eval_mode adaptive` quantifies that gap and must be reported as a separate row.

## The adversarial variant

```bash
python generate_omega.py --opponent withhold
python evaluate_om.py --adapter ckpt/sft --opponent withhold --tag omega-adv
```

One randomly chosen advisor becomes evasive — vague, deferring, never volunteering exact
figures, **and never lying** (a licence to lie would break the benchmark).

Three things this needs, all handled:

- **The ceiling drops.** Every advisor in this corpus holds a decisive fact, so silencing
  any of them makes some checks unreachable. Measured: `dca` ceiling **mean 0.660, range
  0.167–0.889, never 1.0**. `ceiling` and `dca_norm` are computed per episode — compare
  `dca_norm`, and only against runs with the same `--opponent`.
- **The opponent is exempt from the leak gate.** It is playing a role; its leaks land in
  `opponent_leaks` and never invalidate the episode. Without this the arm would poison its
  own reward.
- **The evasion margin is thin.** Full disclosure and careful evasion sit close together
  around the 0.35 threshold. `selftest.py` prints the gap; hand-label ~50 flagged
  disclosures before trusting any adversarial number.

Run the no-opponent version first — it is the control that makes the adversarial numbers
interpretable.

## Cost

Corpus B: 99 scenarios × k=6 ≈ 600 episodes, ~14 calls each, plus 3 scaffold calls per
stalled chair turn. All local. Corpus C: same shape against the API — price it before
launching, and run the probe first.
