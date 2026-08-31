# EPO on CSA

EPO ([Liu et al., ACL 2025](https://arxiv.org/abs/2502.12486)) applied to the CSA
hidden-profile benchmark, as a drop-in replacement for the PPDPP planner.

The comparison is the point: the environment dynamics, the scenario splits, the
disclosure detector and the verifier are held identical to `ppdpp_csa/`, so any
difference in the numbers is attributable to the planner and nothing else. This package
reads from `ppdpp_csa/` and never modifies it.

See `epo-vs-vanilla.pdf` for what had to change relative to the paper, and why.

---

## Where this folder lives

It can live anywhere. `config.py` locates `ppdpp_csa/` by walking up the directory tree
looking for `verifier.py`, `prompt.py` and `data/`, so it handles the folder sitting
beside `ppdpp_csa/`, beside `ppdpp/`, or a couple of levels away. If it is somewhere the
search does not reach:

```bash
export CSA_PPDPP_DIR=/path/to/baselines/ppdpp/ppdpp_csa
```

A wrong or missing path fails immediately with the list of places it looked, rather than
surfacing as `ModuleNotFoundError: No module named 'prompt'` three imports later.

## What is reused, and what is parallel

| Reused from `ppdpp_csa/` (must be identical) | Reimplemented here (this is the port) |
|---|---|
| `data/csa-{train,valid,test}.txt` — 99/9/42 splits | turn loop and reward (`env_epo.py`) |
| `verifier.py` — `score`, `floor_score` | prompts (`prompt_epo.py`) |
| `prompt.py` — personas, view filtering, schema | process reward (`prm.py`) |
| the 0.35 disclosure threshold | policy and optimiser (`strategist.py`) |

`detectors.py` holds a verbatim copy of the lexical disclosure detector so that stage 1
runs without a GPU stack. `run_epo.py` calls `assert_identical_to_ppdpp()` at startup and
fails loudly if that copy has drifted.

## Layout

```
epo/
  config.py             paths, dataset loader, every default in one place
  compat.py             version shims + `python compat.py` environment report
  detectors.py          disclosure detector (verbatim copy + drift check)
  prompt_epo.py         strategist prompt, chair injection, sigma cleaning
  prm.py                VerifierPRM (deterministic) and JudgePRM (EPO-faithful)
  env_epo.py            the meeting environment
  strategist.py         LLM_s: LoRA causal LM, EPO REINFORCE objective
  make_strategies.py    STAGE 1  manufacture SFT targets
  sft_epo.py            STAGE 2  SFT warm-start
  run_epo.py            STAGE 3  RL + evaluation
  make_epo_changes_pdf.py         builds epo-vs-vanilla.pdf
  data/ logs/ ckpt/     outputs
```

## Setup

Runs on the GPU box, not a laptop. Two devices are assumed by default: the frozen
dialogue agent on `cuda:0`, the trainable strategist on `cuda:1`.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Then check what the box actually has. This prints versions, GPU inventory, and anything
that blocks the run — it is the first thing to paste if something fails:

```bash
python compat.py
```

**Only one hard version floor: `transformers >= 4.37`**, where Qwen2 architecture support
landed. Below that Qwen2.5 will not load and no shim helps. Everything above it is
absorbed by `compat.py`, including the `from_pretrained(dtype=)` rename in 4.56 (it was
`torch_dtype=` before), missing `accelerate` (falls back from `device_map` to `.to()`),
`enable_thinking=` on non-Qwen3 tokenizers, the moved scheduler import, and old `peft`
without a `disable_adapter()` context manager — in that last case the KL term is skipped
with a printed warning rather than silently computing zero.

Single GPU instead? Put both on `cuda:0` and expect it to be tight with two 7B models:

```bash
python run_epo.py --agent_device cuda:0 --strategist_device cuda:0
```

The cleaner single-card option is to serve the dialogue agent separately and keep only
the strategist in the training process.

---

## Stage 1 — manufacture the SFT targets

`../ppdpp_csa/data_sft/` already holds 949 labelled chair turns (650 train / 164 valid /
135 test). Each gives the prompt and the **act tag**, but nothing after the colon: the
annotator was told *"Answer with exactly one word"*, so no rationale was ever recorded.
EPO's target is the full `tau: sigma` line, so `sigma` has to be produced.

The chair's real next utterance is the ground truth of what it did. A small model only
compresses it into an instruction, and **`tau` is supplied rather than predicted**, so
this pass cannot introduce label noise — the annotated tags survive untouched.

```bash
export OPENROUTER_API_KEY=...        # never put the key in a file
python make_strategies.py --split train
python make_strategies.py --split valid
```

Free-tier rate limits bite; the script resumes from what it already wrote, so just
re-run it. Add `--sleep 1.5` to pace calls, or `--model` to pick a different one.

No API at all — deterministic templates, also useful as a control:

```bash
python make_strategies.py --split train --fallback_only
```

If the verbalised `sigma` does not beat `--fallback_only` at stage 2, the verbaliser
added nothing and you should say so.

**Output:** `data/strategies-{split}.jsonl`, one row per labelled chair turn, carrying
`prefix` (the dialogue so far), `tau`, `sigma`, and the assembled `target`.

## Stage 2 — SFT warm-start

```bash
python sft_epo.py --epochs 3 --class_balance sqrt_inverse
```

Loss is on the completion tokens only; the prompt is masked to `-100`. The prompt is
truncated **from the left** so the most recent dialogue always survives.

`--class_balance` is not optional in practice. The label distribution is
`ask 304 / decide 190 / share 114 / followup 42` — the same 7:1 ratio that drove the
PPDPP planner's `followup` F1 to 0.000 and forced `logit_adjust.py` into existence.
Generative decoding gives no post-hoc logit adjustment, so it has to be handled here.

**Output:** `ckpt/sft/` (LoRA adapters, ~150–300 MB, not a 1.05 GB full checkpoint).

## Stage 3 — RL

Check the plumbing first. No gradients, prints the reward vector per episode:

```bash
python run_epo.py --dry_run 3
```

Then the real run:

```bash
python run_epo.py --episodes 700 --adapter ckpt/sft --eval_every 175
```

**Output:** `logs/Record-<run>-<tag>.txt` in exactly the format
`../ppdpp_csa/compute_all_metrics.py` consumes, plus `logs/<run>-history.jsonl`
(per-group training trace) and `logs/<run>-summary.json`.

### Knobs that matter

| Flag | Default | Note |
|---|---|---|
| `--prm` | `verifier` | `judge` reproduces EPO's GPT-4o PRM and costs one call per episode |
| `--prm_mode` | `binary` | EPO-faithful; `graded` weights by flip count and is continuous in `dca` |
| `--advantage` | `group` | `maxabs` restores the paper's rule |
| `--group_k` | 4 | rollouts per scenario, for the group baseline |
| `--w_outcome` | 1.0 | terminal outcome added at `t=T`; `0` = process rewards alone |
| `--kl_beta` | 0.01 | EPO reports none; `0` reproduces the paper |
| `--tag_weight` | 2.0 | upweights the act-tag tokens; `1.0` disables |
| `--adapter ""` | — | pure-RL ablation, no warm-start |

### Ablations worth running

```bash
python run_epo.py --advantage maxabs                  # vanilla EPO advantage
python run_epo.py --prm judge                         # vanilla EPO process reward
python run_epo.py --adapter "" --episodes 700         # pure RL, no warm-start
python run_epo.py --tag_weight 1.0                    # no tag upweighting
```

Running `--prm judge` and `--prm verifier` over the same trajectories gives
`prm.agreement(...)` — Cohen's κ between a judged and a computed process reward. No
existing EPO environment can produce that number, because none has executable ground
truth to check the judge against.

---

## What to look at

The PPDPP baseline on this benchmark **did not learn**: over 1,000 episodes, disclosure
went 0.077 → 0.089, `cbar` 0.245 → 0.245, `pbar` flat, SR 0.000 throughout, reward pinned
at −0.094. `'disclosure': 0.0` appears on 2,718 of 2,982 logged steps.

So the primary result is not a number, it is **a non-flat curve**. Read in this order:

1. `disclosure_rate` and `any_reveal` across eval points — did private information
   surface at all
2. elicited vs volunteered (`reveal_elicited`) — was it drawn out or offered
3. `dca` conditioned on `disclosure_rate > 0` — given it surfaced, was it used

`dca` alone flatters the policy: the baseline scored 0.209 with disclosure 0.089, meaning
most passing checks pass from shared context without the private fact ever appearing.
Report it as secondary.

Guardrails that must not regress: `schema_valid ≥ 0.976`, `leaks == 0`, `n_calls ≤ ~15`.
The last one matters — a planner that wins by making more calls has not won, which is why
`n_calls`, `calls_by_role` and `prompt_chars` are recorded per episode.

`tag_misses` is an EPO-only diagnostic: if it climbs, the policy is drifting away from
emitting a parseable act tag, and three sites in the environment depend on that tag.

## Comparing against PPDPP

Records are schema-compatible, so:

```bash
cd ../ppdpp_csa
python compute_all_metrics.py --records ../epo/logs/Record-<run>-final.txt
```

Pair per scenario against `tmp/csa/eval_result/Record-epoch-6-*.txt` — same 42 scenarios,
both planners. A paired test on 42 matched scenarios is far more powerful than comparing
two unpaired means, and the effect here is expected to be small.
