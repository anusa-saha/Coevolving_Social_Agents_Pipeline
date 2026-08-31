# Sotopia-RL on CSA

[Sotopia-RL](https://arxiv.org/abs/2508.03905) (Yu, Qi et al.) applied to the CSA
hidden-profile benchmark: episode-level outcomes refined into **utterance-level,
multi-dimensional rewards**, distilled into an online reward model, then optimised with
single-turn GRPO.

The one change that matters most: on SOTOPIA both halves of the reward are GPT-4o
guesses. On CSA both are **computable**. `decisive_facts` states which checks each private
fact controls, and the transcript says which turn drew it out — so attribution is a
lookup, not a judgement. Zero API calls, byte-for-byte reproducible.

See `sotopia-rl-vs-vanilla.pdf` for every deviation from the paper and why.

---

## Self-contained by construction

This package imports **no code** from `ppdpp_csa/` or `epo/`. It reads the raw scenario
JSON and rebuilds the 99/9/42 split itself, and it has its own verifier, detectors and
prompts.

Re-deriving rather than reusing would be pointless if it silently diverged, so both are
checked against the published artefacts *as data*:

| Check | Result |
|---|---|
| split reproduces the published 99/9/42, order included | **exact** |
| verifier reproduces published `cbar`/`pbar`/`disclosure` on 297 episodes | **exact, all 297** |
| `is_eliciting` vs 650 annotated act labels | acc 0.817, P 0.845, R 0.803 |

Both comparisons skip cleanly if the other baselines aren't present. Nothing in the
training path reads them.

## Layout

```
sotopia_rl/
  paths.py             locate raw/, define data/logs/ckpt
  data_csa.py          load scenarios, rebuild the split from scratch
  verifier_sr.py       executable checks, canonicalisation, provenance resolution
  detectors_sr.py      disclosure / leak / addressing / is_eliciting
  prompts_sr.py        chair, advisor, settlement prompts (view-filtered)
  attribution.py       r_t = G . A(a_t, tau), computed          <- the core
  env_sr.py            the meeting; snapshot/restore for candidate lookahead
  models_sr.py         SharedBackbone + PolicyView / RewardView (one model, three roles)
  compat.py            version shims + `python compat.py` report
  collect_episodes.py  STAGE 1    self-play episodes, verifier-filtered
  make_rm_data.py      STAGE 1b   attributed labels + fitted normaliser
  train_sft.py         STAGE 2.1  behaviour cloning
  train_rm.py          STAGE 2.2  reward model (MSE)
  train_grpo.py        STAGE 2.3  single-turn GRPO
  evaluate_sr.py       held-out evaluation, records match the other arms
  selftest.py          no-GPU checks
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python compat.py && python selftest.py
```

### One model in memory, three roles

The frozen agents, the chair policy and the reward model are all the same checkpoint.
Three bf16 copies would be **~45 GB of weights**; `models_sr.SharedBackbone` loads it
**once** and switches LoRA adapters instead — **~15 GB plus ~300 MB of adapters**.

| Role | How it is obtained |
|---|---|
| frozen agents | base weights, adapters disabled |
| chair policy | `policy` adapter active |
| reward model | `rm` adapter + a linear value head |

Beyond the memory saving this is more honest: the frozen agent is *provably* the
untouched base model rather than a second copy that merely started out equal. It also
makes the GRPO KL reference free — `ref_logprob` is the same forward pass with adapters
off, no second model resident.

There is therefore **one device flag**, `--device`, on every script. `transformers>=4.37`
is the only hard floor; everything above it is absorbed by `compat.py`.

### Fitting a smaller card

```bash
# ~15 GB weights + ~4 GB activations -- fits a 24 GB card
python train_grpo.py --device cuda:0 --grad_checkpointing

# tighter still: no reward model resident at all, and no train_rm stage
python train_grpo.py --device cuda:0 --grad_checkpointing --reward_source lookahead
```

`--grad_checkpointing` cuts training activations roughly six-fold for about 30% slower
steps — the best memory trade at these sequence lengths. `--reward_source lookahead`
drops the `rm` adapter and head entirely; see the GRPO section for what it costs.

If it still does not fit, in order: `--max_len 1024`, then `--grpo_group 4`, then
quantise the backbone. **Quantising is the one to avoid if you can** — with a shared
backbone it also quantises the frozen agent, which changes the environment relative to
the other baselines and breaks cross-arm comparability.

### The scenarios

The model arrives from the Hub on first use, and so do the scenarios — a fresh clone needs
no manual data copying. Default source is
[`anusasaha/Coevolving_Social_Agents`](https://huggingface.co/datasets/anusasaha/Coevolving_Social_Agents),
files under `data/` in that repo.

`paths.find_raw()` resolves in this order:

1. `CSA_RAW_DIR` if set
2. `data/raw/` (the download cache), then a walk up the tree for a sibling `raw/`
3. **download** into `data/raw/`

Nothing needs exporting; it just works. Overrides exist if you need them:

```bash
export CSA_HF_REPO=<org>/<dataset>      # different source
export CSA_RAW_DIR=/path/to/raw         # skip the Hub, use a local copy
```

The download is `paths.download_raw()`; `data_csa.load_raw()` reads whatever it returns.
Verified byte-identical (sha256) to the copy the other two baselines used, and
`selftest.py` confirms the rebuilt split is still exactly 99/9/42.

> **Only three of the eleven domains are downloaded.** The Hub repo also ships
> `bargaining`, `education`, `entertainment`, `family_friends_informal`, `finance`,
> `legal`, `manufacturing` and `workplace_interpersonal`. The published split was built
> from `healthcare + defense + software_technology` alone, and adding any of the others
> reshuffles every bucket of the stratified split — this arm would then be evaluated on
> different scenarios than PPDPP and EPO. `paths.ALL_HUB_DOMAINS` lists them; widening
> `paths.DOMAINS` is a benchmark change that invalidates existing results and needs all
> three arms re-run.

---

## Stage 1 — collect episodes

The chair speaks **unprompted**. No planner, no act, no strategy: whatever it does is
what gets cloned and what the attributor later scores.

```bash
python collect_episodes.py --split train --k 6 --keep 2
python collect_episodes.py --split valid --k 4 --keep 2
```

`k` rollouts per scenario, ranked **within** the scenario, top `keep` retained — so hard
scenarios still contribute instead of the filter quietly selecting easy ones. Resumes if
interrupted. Prints a projection of the reward labels it will produce, and warns if all
`k` rollouts of a scenario scored identically (the filter has nothing to work with).

## Stage 1b — attribute

```bash
python make_rm_data.py --episodes data/episodes-train.jsonl
```

One row per chair turn: `prompt_messages`, `completion`, and the scalar `label`. Fits the
dataset-level normaliser and writes it to `data/normaliser.json` so every downstream
stage shares one scale.

**Three dimensions**, the CSA analogue of the paper's REL / KNO / GOAL:

| | |
|---|---|
| `pool` | did private information surface — the analogue of *knowledge seeking* |
| `use` | did it land in passing checks — the analogue of *goal completion* |
| `cover` | did the chair draw out the people holding it |

`cover` is **not** "advisors whose surname was mentioned". Measured that way it saturates
on the opening turn where the chair greets everyone, rewarding politeness. Here an
advisor counts only once it has been addressed **and has then disclosed something**.

Measured on 297 real episodes: 30.8% of chair turns carry a non-zero label, max 0.99,
std 0.14.

## Stage 2.1 — behaviour cloning

```bash
python train_sft.py --epochs 3 --min_dca 0.34
```

Loss on completion tokens only; the prompt is masked to `-100` and the boundary is
asserted, not assumed. `--min_dca` filters which episodes are worth cloning — the
demonstrations here are Qwen self-play, not GPT-4o, so filtering harder is usually right.

## Stage 2.2 — reward model

```bash
python train_rm.py --epochs 4
```

Scalar regressor over `(state, action)`, MSE against the attributed labels. Held out by
**scenario**, not by turn. Selected on **within-episode pair-ranking accuracy** rather
than MSE alone, because GRPO standardises inside each group — what matters is whether the
model orders candidates at the same state correctly.

## Stage 2.3 — GRPO

Check the plumbing first; this prints candidates and their scores without training:

```bash
python train_grpo.py --dry_run 3 --reward_source lookahead
```

Then:

```bash
python train_grpo.py --adapter ckpt/sft --rm ckpt/rm --groups 175 --group 8
```

Three ways to score a candidate:

| `--reward_source` | How | Cost per candidate |
|---|---|---|
| `rm` | the trained reward model — faithful to the paper | 1 forward |
| `lookahead` | commit it, let one advisor answer, read the detector, roll back | 1 generation |
| `hybrid` | lookahead, falling back to the RM when it reads zero | mixed |

Settling turns are **always scored exactly** — a settlement can be parsed and run through
the verifier with no rollout and no model — so the highest-stakes turn in every episode
gets ground truth regardless of the mode.

`lookahead` needs no reward model at all. That inverts the paper's design for a stated
reason: the RM exists to make an expensive judge cheap, and CSA's judge is already free.
Running both is what lets you *measure* the distillation error.

**Watch the collapsed-group fraction.** GRPO gets no gradient from a group whose
candidates all score alike; it's logged per group and summarised at the end, with a
warning past 50%.

## Evaluation

```bash
python evaluate_sr.py --adapter "" --tag base           # untrained chair = the floor
python evaluate_sr.py --adapter ckpt/grpo/grpo-rm-seed1/final --tag grpo
```

Greedy decoding. Records land in `logs/Record-<tag>-test.txt` in the same shape the other
arms write, so all three can be scored by one script and **paired per scenario** — far
more powerful than comparing unpaired means at n=42.

## Reading the result

Report in this order:

1. `disclosure_rate` and `any_reveal` — did private information surface at all
2. `elicited_frac` — was it drawn out or volunteered
3. `dca` **conditioned on** `disclosure_rate > 0` — given it surfaced, was it used

`dca` alone flatters a policy: on this benchmark it reads around 0.21 while disclosure
reads 0.09, meaning most passing checks pass from shared context without the private fact
ever appearing. Report it as secondary.

Guardrails that must not regress: `schema_valid`, `leaks == 0`, and `n_calls`. The last
matters — GRPO's group sampling makes this arm far more expensive per episode than the
planner baselines, so compare at matched budget, not just matched episode count.

## Honest caveat

This arm **fine-tunes the dialogue agent**; the planner baselines hold it frozen and
train a planner on top. The three are not interchangeable rows in one table. Either
report two axes — "planner over frozen agent" versus "fine-tuned agent" — or state
plainly that the agent differs. `n_calls` and the frozen-advisor setup stay comparable;
goal-completion numbers do not.
