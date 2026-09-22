# train_1100_big

```
python run.py          # trains, then evaluates - resumes automatically after a crash
```

`train_450v2_big` on the **1100D** dataset. The method is untouched: same Qwen3 backbones,
same LoRA ranks, same prompts, same reward, same LAST CALL, same OPD, same TTPO, same
`GROUP_SIZE=4`, same estimator (GRPO), same learning rates, same seed, same alternating
phase schedule. Three things differ, none of them the algorithm:

```
data    450D  360/90          ->  1100D  720/380 with 2 domains held out entirely
GPU     1 x 80 GiB, uncapped  ->  1 card capped at 60 GiB (gpu.py)
eval    eval.py               ->  inf.py (seen / unseen / total + per-domain)
```

| | |
|---|---|
| router | `Qwen/Qwen3-4B` (`opd.OPD_CFG.MODEL`), ~8 GiB bf16 |
| agent | `Qwen/Qwen3-8B` (`rl.RL_CFG.MODEL`), ~16.4 GiB bf16 |

## The split, and why it is not 450D's

450D was a plain domain-stratified 80/20: every test domain was also a training domain, so
the test set could only ever measure held-out *episodes*. 1100D follows the 550D
methodology instead - **two domains are held out entirely** - so the test set measures
held-out *domains* as well:

| file | n | what |
|---|---|---|
| `data/1100_train.json` | 720 | 9 domains x 80 |
| `data/1100_test.json` | 380 | 180 `eval_group="seen"` + 200 `eval_group="unseen"` |

* **seen** (180) - those same 9 domains' held-out 20%, 20 each. Unseen episodes, trained domains.
* **unseen** (200) - `friends_family_informal` and `informal_commerce_bargaining`, 100 each.
  Never in training, in any phase. A gain that holds here is not domain memorisation.

`inf.py` reports the two groups separately, then TOTAL, then the gap between them.

**Train and test do not overlap.** `data/prepare_data.py` rebuilds both files from
`ICLR/1100D/` and hard-fails unless all three of these are disjoint: `uid`, a SHA-256 hash
of the scenario body, and `(domain, scenario_id)`. It also stamps each scenario with its
index in the pooled 1100-set as `uid`, so a uid identifies the same scenario everywhere -
`scenario_id` alone does not, because it restarts per domain. Re-run it any time:

```
python data/prepare_data.py
```

## Cost

720 x G=4 = **2,880 episodes**, four phases of 180 scenarios, 22 optimiser steps per phase
(44 router + 44 agent). That is 2x `train_450v2_big`. `ROUTER_WARMUP` is 10, not 5, purely
to hold the same ~22% warmup *fraction* over the larger step count - the rule `main.py`
already states for any change to the step schedule.

## GPU budget - 60 GiB

Everything hardware lives in `gpu.py`, which `main.py` and `inf.py` both read, so training
and eval agree on the card and the cap.

```
router + agent weights bf16              ~24.4 GiB
two LoRAs + AdamW moments                 ~1.0 GiB
------------------------------------------------
resident before any episode              ~25.4 GiB
left for KV cache / update activations   ~34.6 GiB
```

Both models are Qwen3 with the same KV geometry (36 layers, 8 KV heads, head_dim 128):
0.141 MiB per token, so a full 4096-token episode costs ~0.56 GiB of KV per model. That
sets the two batch knobs:

| knob | value | why |
|---|---|---|
| `main.CONFIG.SCENARIOS_PER_STEP` | 8 | unchanged from `train_450v2_big`; the real lockstep batch is this x `GROUP_SIZE` |
| `main.CONFIG.ROLLOUT_BATCH` | 32 | = 8 x 4, the ceiling that product allows - a larger number would be dead headroom |
| `gpu.EVAL_BATCH` | 32 | eval takes no backward, so the whole ~34.6 GiB is KV cache |
| `gpu.GRAD_CHECKPOINT` | False | spend VRAM on activations, update faster - identical gradients |

These are arithmetic, not measurements. **Watch `peak_gib` in
`results_train/train_steps.csv` for the first few steps.** If it is not leaving a
comfortable margin, drop `ROLLOUT_BATCH`; if `nvidia-smi` shows headroom, `EVAL_BATCH` is
the first knob to raise. Neither is a cliff: `env.run_batch` halves the rollout batch and
retries on OOM, and `main._update_with_retry` frees the cache and retries an update, so a
too-large value costs time rather than the run.

## Evaluation

`run.py` runs `main.py`, then `inf.py` (this folder has no `eval.py` - `inf.py` replaces
it, and is the only evaluator that knows about `eval_group`). Both arms are **blind**
(student profile, no privileged context), asserted at startup:

```
arm         router LoRA   agent LoRA
vanilla         off           off      the untrained base system
coevolved       ON            ON       ckpt/latest
```

`results_inf/inf.log` plus CSVs: full metric block per arm on SEEN / UNSEEN / TOTAL, the
headline table, the paired bootstrap (coevolved - vanilla, 95% CI, p), the generalisation
gap (unseen - seen) and whether training widened it, the per-domain overview and detail
tables, and the insight-leak guardrail.

Read the per-domain deltas as descriptive: n=20 per seen domain and n=100 per unseen
domain, so only a large, consistent effect is detectable at the domain level. The seen /
unseen / total blocks are where the result is.

## Crash recovery

`main.py` commits a full checkpoint (both LoRA adapters, both AdamW states, the LR
schedule, and every RNG - python/torch/cuda) after **every optimiser step**, atomically
(`resume.py`: write to `ckpt/latest.tmp`, fsync, rename - never a half-written directory).
Every CSV/JSONL output is truncated back to its checkpointed byte offset on restart, so
resumed rows are never duplicated. `inf.py` commits per arm (`results_inf/<arm>_done.json`,
keyed to a fingerprint of the checkpoint's adapter weights) so a stale result from a
different checkpoint is discarded rather than reused.

`run.py` wraps both stages: if a stage's process dies (OOM, a CUDA error, a kill), it is
restarted in a fresh process and picks up from the last commit. It only gives up on a fatal
config/data error, on the same crash repeating with no checkpoint progress, or on Ctrl+C -
otherwise re-running `python run.py` at any point continues exactly where it left off.

---

## The method (unchanged from train_450v2_big)

GRPO, applied to **both** learners: one advantage per rollout, shared by the router's
`next_agent` spans and the agent's text spans.

```
A_i = (r_i - mean_j r_j) / (std_j r_j + ADV_EPS)
```

`rl.RL_CFG.ESTIMATOR = "rloo"` switches back to `A_i = r_i - mean_{j != i} r_j`, which is
the cheapest way to confirm the two trees differ only where intended.

**`ADV_EPS = 1e-4` is load-bearing** - the canonical DeepSeekMath denominator floor. Too
small and near-degenerate groups saturate at ±`ADV_CLIP`; too large and the division stops
mattering, which quietly turns GRPO back into RLOO-with-a-rescale. Report the value you
used. Fully degenerate groups return all zeros rather than `0/eps`.

**GRPO advantages are ~5x larger on average than RLOO's**, so the effective learning rate
is ~5x higher. The learning rates are deliberately left identical, because that is what a
like-for-like comparison of published defaults means - but it is a confound and must be
stated when reporting. If GRPO underperforms, run one LR-matched control (`ROUTER_LR
4e-6`, `AGENT_LR 2e-6`) before concluding.

Three diagnostic columns in `results_train/train_steps.csv`, none of which touch the loss:

| column | what it tells you |
|---|---|
| `group_std` | mean reward spread inside a group. GRPO **divides** by this. |
| `low_std_frac` | share of non-degenerate groups with `std < 0.05`, where the advantage is manufactured from noise. Expect ~0.30. |
| `adv_saturated` | share of advantages pinned at ±`ADV_CLIP`. **0 under RLOO.** |

The failure signature: **`low_std_frac` high AND `adv_saturated` high** means the std
division is driving training rather than the reward.

## Files

```
main.py            training. CONFIG at the top is every knob.
inf.py             evaluation: vanilla vs coevolved, seen / unseen / total + per domain.
run.py             main.py -> inf.py, with crash restarts.
gpu.py             the ONLY file to edit per machine: card, 60 GiB cap, eval batch.
env.py             the environment: episodes, LAST CALL, the verifier, the metrics.
opd.py / rl.py     router (OPD) and agent (RL) models and their configs.
ttpo.py            token-level selection for the asymmetric split.
resume.py          crash-safe atomic checkpointing, shared by main.py and run.py.
prompts.py         student (blind) and teacher (privileged) profiles.
data/              1100_train.json, 1100_test.json, prepare_data.py (rebuild + overlap check)
```
