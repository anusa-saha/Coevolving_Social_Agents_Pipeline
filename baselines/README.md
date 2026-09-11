# CSA Baselines

Six social-agent methods ported to **CSA** ([Coevolving Social
Agents](https://huggingface.co/datasets/anusasaha/Coevolving_Social_Agents)), a
hidden-profile benchmark of multi-party meetings. In each meeting one agent chairs, three
to five advisors each hold a private fact, and the chair has to pool enough of those facts
to reach a settlement that passes the scenario's executable checks.

The point of the repo is a *fair* comparison. All six arms use the same dialogue backend
(Qwen2.5-7B-Instruct), the same scenario split, the same disclosure detector, the same
verifier and the same annotator — so a difference between two rows is a difference between
two methods and not between two measuring instruments.

### Benchmark configuration

**All 11 domains × 50 scenarios = 550**, split 363 / 33 / 154 (train / valid / test),
scenario-disjoint and stratified on (domain, num_agents).

The Hub ships 100 per domain; 50 is the configured subset, taken lowest-`scenario_id`
first. Head rather than sample, so raising the cap only ever *adds* scenarios instead of
reshuffling the ones already in use — and the three original domains are bit-identical to
the scenarios the earlier runs saw.

```bash
export CSA_SCENARIOS_PER_DOMAIN=100   # the full 1100
export CSA_DOMAINS=published          # the original 3 domains x 50 -> exactly 99/9/42
```

> **This configuration replaces the earlier 3-domain / 150-scenario one, and invalidates
> every result produced under it.** The split is rebuilt from scratch, so no arm is
> evaluated on the scenarios its archived records used. Every arm needs re-running before
> its numbers mean anything.

Two data notes, both handled at load time and neither written back to disk:

* `family_friends_informal` and `manufacturing` each have one scenario whose
  `settlement_schema` nests a level deeper under `settlement`. Normalised rather than
  dropped — the contents are identical.
* `family_friends_informal_scenarios.json` carries an internal `domain` field spelled
  `friends_family_informal` (transposed). The filename stem wins, so domain names, uids
  and filenames all agree.

Two scenarios ship content checks that can never pass — `healthcare::scenario_47` C2 and
`family_friends_informal::scenario_34` C6 both read a decision field absent from their own
schema. They cap the achievable score on those two scenarios; `python ppdpp/export_csa.py`
prints them.

---

## Layout

```
csa_core/          the shared contract: split, detectors, verifier, paths, compat
data/raw/          the three domain JSON files (fetched from the Hub on first use)

ppdpp/             PPDPP        RoBERTa act classifier + REINFORCE
epo/               EPO          LLM strategist emitting NL strategies, turn-level REINFORCE
sotopia_rl/        Sotopia-RL   utterance-level attributed rewards, reward model, GRPO
sotopia_tom/       Sotopia-ToM  four prompting strategies, no training
sotopia_omega/     SOTOPIA-Ω    stall detection -> slow thinking -> SFT corpus
dat/               DAT          frozen LM + 2-token continuous prefix, self-clone then TD3+BC

analysis/          cross-arm metrics and the rerun runbook
```

### `csa_core/` is the part that must not fork

The split, the disclosure detector and the verifier are the *instrument*. If two arms
score with different copies of them, their numbers are not comparable. There is exactly
one copy, and every arm imports it:

| module | what it fixes |
|---|---|
| `data_csa` | the split, re-derived from a fixed procedure rather than read from a file |
| `detectors` | word-overlap disclosure rules, threshold frozen at **0.35** |
| `verifier` | deterministic scoring of a settlement against the dataset's checks |
| `paths` | the domain list and per-domain cap; the annotator model; where each arm writes its own outputs |
| `compat` | version shims for transformers / peft / accelerate |

Prompts, environments and training loops stay per-arm, because that is the part each
method is entitled to change.

One duplicate survives on purpose: `ppdpp/env.py` still carries its own inline copy of the
overlap rule, because it is vendored upstream code the CSA port modified in place.
`csa_core.detectors.assert_matches_ppdpp()` watches it, and EPO's selftest calls it.

---

## Install

```bash
pip install -e .
```

That puts `csa_core` on the path, which is what the arms import. Each arm also ships its
own `requirements.txt` for the extras it alone needs. Report generation and API-backed
experts are optional:

```bash
pip install -e ".[reports]"     # reportlab + matplotlib, for the PDF builders
pip install -e ".[api]"         # openai, for EPO stage 1 and Omega corpus C
```

Running the scripts straight out of a clone works without installing — each arm's
`paths.py` puts the repo root on `sys.path` as a fallback — but `pip install -e .` is the
documented route.

The dataset downloads itself on first use via `csa_core.paths.download_raw()`. To point at
a local copy instead:

```bash
export CSA_RAW_DIR=/path/to/raw
```

The annotator that labels chair turns and writes EPO's strategy targets is also one
setting, for the same reason the verifier is — both PPDPP and EPO are supervised from it,
and letting them drift apart makes the arms incomparable:

```bash
export CSA_ANNOTATOR_MODEL=google/gemma-4-31b-it   # default: gemma-4-26b-a4b-it:free
```

---

## Check it works

Every arm ships a selftest that runs on CPU in about a minute and needs no model. They
assert the things that would otherwise fail silently: that the split is complete and
scenario-disjoint, that the verifier reproduces published scores, that no prompt leaks a
private fact into the wrong agent's view.

```bash
cd sotopia_rl && python selftest.py
```

| arm | checks |
|---|---|
| `epo` | 8 |
| `sotopia_rl` | 9 |
| `sotopia_tom` | 11 |
| `sotopia_omega` | 9 |
| `dat` | 23 |

"Verifier matches published scores" needs the archived records; without them it skips and
says so (see **Results**). "Matches the published split" only runs under
`CSA_DOMAINS=published`, since the published split describes the old 3-domain dataset —
and under that setting all four arms still reproduce 99/9/42 exactly.

---

## Results are not in this repo

Evaluation records, checkpoints and generated report PDFs are outputs. They are large
(~4.5 GB, mostly model weights) and they are results, so they live outside the tree.
Point the tooling at wherever they were archived:

```bash
export CSA_ARTIFACTS_DIR=/path/to/csa-artifacts
```

With that set, the selftests validate against the published run and the metrics scripts
score every arm that has records. Without it, everything still runs — the checks that need
records skip cleanly, and an arm you re-run is picked up from its own `logs/`.

---

## Cross-arm metrics

```bash
cd analysis && python compute_extended_metrics.py
```

Discovers every arm that has records and scores them all. Pure stdlib — no torch, no CUDA,
no judge model. Seconds, not hours.

| section | what it reports |
|---|---|
| Headline | dca and disclosure with bootstrap 95% CIs, cost-normalised by calls and tokens |
| Paired | sign test, Cliff's delta, bootstrap CI on the paired difference, Holm correction, N×N win matrix |
| E | gold recovered from content checks; exact-match accuracy; PE@10/20/30, MAE, RMSE, MAPE, R², correlations |
| F | distinct-1/2/3, utterance length, role-adherence violations, private-view leakage |
| G | act distribution and entropy, bigrams, return mean and variance |
| H | breakdown by domain and table size, per-scenario spread, every verifier subscore with its own CI, difficulty tertiles |

`make_rerun_runbook.py` builds a PDF that reads the current state off disk and says, per
arm, whether it needs re-running and what the command is.

---

## Running the baselines

Every arm follows the same shape: **manufacture training material → train → evaluate →
report**. What differs is how many of those stages exist. Sotopia-ToM has none of the
first three; SOTOPIA-Ω has all four.

Before anything, confirm the environment:

```bash
python csa_core/compat.py
```

That prints which of torch / transformers / peft / accelerate / nltk / openai are present
and whether the versions clear the floors. `transformers>=4.37` is the only hard floor —
Qwen2 support landed there and no shim can work around its absence.

### Hardware

Every arm runs Qwen2.5-7B-Instruct in bfloat16: **~15 GB of weights** plus activations.
The reference setup is 2×A5000. One 24 GB card is enough for every arm if you pass
`--grad_checkpointing` where it is offered (~6× less activation memory, ~30% slower).

Two arms want a second card by default:

| arm | why |
|---|---|
| EPO | `strategist_device = 'cuda:1'` — the strategist is a second 7B beside the dialogue agent. Set `--strategist_device cuda:0` to co-reside. |
| Sotopia-RL | policy, reward model and reference share **one backbone with three adapters**, so it fits one card; `--agent_device` splits it if you have two. |

DAT wants neither, but its stage 1 is the only place in the repo where a backward runs
**through** the frozen 7B (to reach a two-token prefix), so activation memory rather than
parameter memory is the limit there. Gradient checkpointing is on by default.

---

### 1. PPDPP

RoBERTa-large act classifier + REINFORCE. Two stages: supervised warm start, then RL.

```bash
cd ppdpp
```

**Stage 1 — planner warm start.** Needs the annotated chair turns (archived; see
*Results*). Skip if `sft/csa/roberta/best_checkpoint` already exists.

```bash
python sft.py --data_name csa --model_name roberta --model_name_or_path roberta-large --output_dir sft --do_train --do_eval  --num_train_epochs 10 --max_seq_length 512
```

**Stage 2 — RL.** `--csa_reward` is the flag that matters:

```bash
# PPDPP's own dense per-turn critic reward. This is the paper's configuration
# and the default -- and it has never actually been run on CSA.
python run.py --data_name csa --system qwen --user qwen --critic qwen --csa_reward critic --seed 1 --epochs 6 --max_turn 8 --do_train --do_eval
```

```bash
# the sparse shared episode reward the other four arms use
python run.py --data_name csa --system qwen --user qwen --critic qwen --csa_reward verifier --seed 1 --epochs 6 --do_train --do_eval
```

Records land in `tmp/csa/eval_result/Record-epoch-*-<reward>-seed<n>.txt`. The reward name
is in the filename, which is how you tell the two arms apart later.

**Watch:** the run prints `zero-gradient updates: X%` beside every learning curve. Under a
sparse constant reward that fraction is high — a constant raw reward carries no
cross-episode signal, and the flat curve is measuring the reward config rather than the
algorithm. Under `critic` it should drop sharply. If it does not, that is a result about
PPDPP on this task, not a bug to chase.

~6 GPU-hours for the full 6 epochs.

---

### 2. EPO

An LLM strategist emits a natural-language strategy each chair turn; the frozen dialogue
agent renders it. Three stages.

```bash
cd epo
```

**Stage 1 — manufacture the SFT targets.** An annotator model writes one strategy per
labelled chair turn. Needs an API key:

```bash
export OPENROUTER_API_KEY=...          # never put the key in a file

python make_strategies.py --split train --model google/gemma-3-27b-it
python make_strategies.py --split valid --model google/gemma-3-27b-it
```

No key, or no budget? `--fallback_only` derives the targets from the act labels
deterministically instead. Quality drops; the pipeline still runs:

```bash
python make_strategies.py --split train --fallback_only
```

Produces `data/strategies-{train,valid}.jsonl`.

**Stage 2 — SFT warm start.** LoRA (r=16), not the full fine-tune the paper uses — 99
scenarios would memorise:

```bash
python sft_epo.py --epochs 3 --lr 1e-5 --accum 8 --class_balance sqrt_inverse
```

Produces `ckpt/sft`. **This checkpoint is the one to beat.** In the archived run it was
the best checkpoint, and 700 episodes of RL degraded it from there.

**Stage 3 — RL.**

```bash
python run_epo.py --episodes 700 --seed 1 --prm verifier --prm_mode binary --advantage group --eval_every 175 --eval_split test --agent_device cuda:0 --strategist_device cuda:1
```

| flag | default | what it does |
|---|---|---|
| `--prm` | `verifier` | `verifier` is deterministic; `judge` calls an API model per turn |
| `--prm_mode` | `binary` | `binary` is EPO-faithful; `graded` uses the continuous score |
| `--advantage` | `group` | `group` baselines within a scenario's k rollouts; `maxabs` is the paper's max-abs rule |
| `--group_k` | 4 | rollouts per scenario for the baseline |
| `--kl_beta` | 0.01 | EPO reports none; this is insurance against collapse |

Single card: `--strategist_device cuda:0`. Sanity check first with `--dry_run 5`.

**Watch:** unique-strategy rate and act entropy, both printed each eval. In the archived
run they went 96.5% → 37.0% and 1.110 → 0.000 bits by ep700, with 89 tag misses and 20
empty strategy strings — the policy collapsed onto one act. `--kl_beta` is the knob;
raising it is the first thing to try.

~5 GPU-hours for 700 episodes.

---

### 3. Sotopia-RL

Utterance-level attributed rewards, a learned reward model, then GRPO. Four stages, and a
**gate** between stages 3 and 4 that is worth respecting.

```bash
cd sotopia_rl
```

**Stage 1 — collect episodes.** k=6 rollouts per scenario, keep the best 2 ranked *within*
the scenario so hard scenarios still contribute:

```bash
python collect_episodes.py --split train --k 6 --keep 2 --restart
```

`--restart` resumes an interrupted collection instead of starting over. Produces
`data/episodes-train.jsonl`.

**Stage 2 — attribute.** Turn the episode return into per-utterance rewards
`r_t = G · A(a_t, τ)` over three dimensions (pool / use / cover):

```bash
python make_rm_data.py --episodes data/episodes-train.jsonl
```

Produces `data/rm-train.jsonl` and `data/normaliser.json`.

**Stage 3 — behaviour cloning, then the reward model.**

```bash
python train_sft.py --epochs 3 --lr 1e-4 --accum 8 --grad_checkpointing

python train_rm.py --epochs 8 --lr 5e-6 --holdout 0.15 --grad_checkpointing
```

> **Gate — read `ckpt/rm/rm_meta.json` before spending GPU time on GRPO.**
> ```bash
> python -c "import json;m=json.load(open('ckpt/rm/rm_meta.json'));print(m['best_pair_rank'], m['best_epoch'])"
> ```
> `best_pair_rank` is pairwise ranking accuracy against a **0.500 chance floor**. The
> archived run reached **0.547 with `best_epoch=0`** — the first epoch was the best, which
> is what a model that never learned looks like. GRPO against a reward model at chance
> optimises its error: in that run the exact-scored signal fell 0.340 → 0.149 across
> quartiles while invalid schemas climbed 0 → 13, even as the RM-scored signal rose.
>
> Below ~0.60, skip stage 4's `--reward_source rm` and use `lookahead` instead: it commits
> the candidate, lets one advisor answer, and reads the disclosure detector — deterministic,
> API-free, and it drops the reward-model adapter and the whole `train_rm` stage. A reward
> model that will not rank is a finding worth reporting, not an obstacle to push past.

**Stage 4 — GRPO.**

```bash
python train_grpo.py --adapter ckpt/sft --rm ckpt/rm --reward_source rm --groups 175 --group 8 --kl_beta 0.02 --seed 1 --grad_checkpointing
```

```bash
# the honest fallback when the RM never ranks
python train_grpo.py --adapter ckpt/sft --reward_source lookahead \
                     --groups 175 --group 8 --seed 1 --grad_checkpointing
```

175 groups × 8 candidates = 1400 chair generations.

**Stage 5 — evaluate.** This is the step the archived run never reached, which is why the
arm contributes no rows to any comparison table:

```bash
python evaluate_sr.py --adapter ""            --split test --tag base   # untrained floor
python evaluate_sr.py --adapter ckpt/sft      --split test --tag sft    # SFT only
python evaluate_sr.py --adapter ckpt/grpo/final --split test --tag grpo # trained
```

Run all three. Without the floor and the SFT mid-point, a GRPO number means nothing.

~5 GPU-hours end to end.

---

### 4. Sotopia-ToM

Prompting only — no training, no gradient step, no checkpoint. The cheapest arm per
GPU-hour in the repo, and the fastest way to get a full row of results.

```bash
cd sotopia_tom
python selftest.py                    # ~1 min, catches prompt regressions

python run_tom.py --strategies stripped basic cot tom_coach tom_belief \
                  --split test --compare
```

The five arms, in ascending order of scaffolding:

| arm | what the chair is given |
|---|---|
| `stripped` | the meeting, nothing else — the floor |
| `basic` | a role and the task |
| `cot` | emit `THINKING` before `TURN` |
| `tom_coach` | an analyst table: who plausibly holds what |
| `tom_belief` | an explicit belief-state JSON, maintained across turns |

`--compare` runs them against identical scenarios and prints the paired comparison.

**Watch:** `tom_coach` and `tom_belief` were identical prompts at one point, which made
their difference zero by construction. They now carry distinct headers and `selftest.py`
asserts it — but if the five arms come back suspiciously close, diff the rendered prompts
before concluding that theory-of-mind prompting does not help.

**Reading InfoMgmt.** The arm's headline is a geometric mean:

```
InfoMgmt = [DA · IA · (1 − CPV) · EFF]^(1/4)
```

Any single component at zero zeroes the whole score. If an arm reports 0.000, read the
four components before concluding it failed.

~3 GPU-hours for all five arms (210 episodes, ~18 calls each).

---

### 5. SOTOPIA-Ω

Detect that the dialogue has stalled, switch the expert into a slow four-stage mode,
fine-tune the student on what results. The only arm whose stages strictly depend on each
other — nothing here can be partially skipped.

```bash
cd sotopia_omega
python selftest.py
```

**Stage 0 — probe. Do not launch the full generation blind.**

```bash
python generate_omega.py --split train --probe 5 --expert local
```

`--probe` runs a handful of scenarios and prints the stall decisions. If stalls never
fire, the fast/slow switch never engages and the corpus is plain rollouts with extra
steps. Stall detection here is deterministic — `step ≥ stall_after` **and** decisive-fact
pooling flat for `stall_patience` chair turns — so if it never triggers, tune
`--stall_after` / `--stall_patience` rather than shipping a corpus that has no slow mode
in it.

**Stage 1 — corpus.**

```bash
python generate_omega.py --split train --expert local \
                         --k 6 --keep 2 --stall_after 1 --stall_patience 1 \
                         --seed 1 --restart
python generate_omega.py --split valid --expert local --seed 1
```

Three corpora are worth having, and they answer different questions:

| corpus | expert | isolates |
|---|---|---|
| **A** | plain self-play (the other arms already have this) | baseline |
| **B** | `--expert local` — Qwen2.5-7B, strategy-injected | **strategy injection alone** |
| **C** | `--expert api --expert_model <frontier>` | injection **+** teacher strength |

B vs A is Ω's actual mechanism. C vs B is the distillation gradient. C vs A is the paper's
headline and confounds both. **Run C alongside B, never instead of it** — a win for C
partly means "the frontier model is better than Qwen2.5-7B", which is not a finding.

**Stage 2 — SFT the student.** Only the utterance the chair actually spoke becomes a
label; the three reasoning stages are scaffolding and never enter the corpus. That is
where the distillation happens.

```bash
python train_sft_om.py --mode_filter all --grad_checkpointing
```

`--mode_filter slow` trains only on stalled turns; `--min_dca 0.3` keeps only episodes
above a score floor.

**Stage 3 — evaluate.**

```bash
python evaluate_om.py --adapter ""       --split test --tag base    # untrained floor
python evaluate_om.py --adapter ckpt/sft/student --split test --tag omega
python evaluate_om.py --adapter ckpt/sft/student --split test --eval_mode adaptive --tag omega-adaptive
```

`--eval_mode adaptive` lets the student pick fast vs slow at inference; run it as a second
arm, not instead of the fast one.

**The leakage control.** The single-opponent adaptation selects one agent and has the rest
negotiate against it — and that agent sees the others' disclosures, so a corpus built from
its trajectories can encode private facts the student should not have:

```bash
python evaluate_om.py --adapter ckpt/sft --split test --opponent withhold --tag omega-withhold
```

If the trained student loses most of its advantage when the opponent view is withheld, the
gain was leakage. Run this before reporting any Ω number.

~9 GPU-hours for corpus B + SFT + evaluation.

---

### 6. DAT

Freeze the model, freeze the prompt, and steer generation with **two continuous prefix
embeddings** predicted by a 2.5 M-parameter MLP. The only arm that changes nothing a human
could read.

```bash
cd dat
python selftest.py                       # 23 checks, ~1 min, no GPU

python selfclone.py --episodes 120       # stage 1: clone the unsteered chair
python collect_buffer.py --episodes 400  # stage 2a: offline buffer, N(0, 0.25) in R^64
python train_dat.py --steps 4000         # stage 2b: TD3+BC. Loads no model at all
python run_dat.py --arms unsteered selfclone dat --split test
```

Three conditions, matching the paper's Table 1, from **one** checkpoint -- `selfclone` is
`dat` with the RL head forced to zero, so the two differ by nothing else:

| arm | prefix | reads as |
|---|---|---|
| `unsteered` | none | the control |
| `selfclone` | `pi_phi(s) W` | stage 1 preserved behaviour, or it did not |
| `dat` | `(pi_phi(s) + pi_phi_rl(s)) W` | the trained arm |

**Watch:** `action_cos` near 1.0 in the steering table means the planner emits the same
vector at every turn -- a constant prefix, not a policy -- and the outcome metrics cannot
see it. `policy_shift.mean_action_norm` at zero after `train_dat.py` means the RL head
never left its initialisation, usually because the buffer holds almost no reward;
`collect_buffer.py` warns when under 5% of transitions carry any.

The reward is EPO's verifier PRM, and `selftest.py` asserts the two are identical on 400
random traces -- two RL arms supervised by different rewards would not be comparable.

The buffer is the under-resourced number: 400 episodes against the paper's 10,000. Raise
it first. ~4 GPU-hours for all three stages plus evaluation.

---

### After any run

```bash
cd analysis && python compute_extended_metrics.py
```

It discovers whatever has records — the arm you just re-ran is picked up from its own
`logs/` automatically, and arms you did not touch keep their existing numbers. No GPU, a
few seconds.

`python make_rerun_runbook.py` builds a PDF that reads the current state off disk and
says, per arm, whether it needs re-running and why. **Read it before spending GPU time**:
of the six arms, PPDPP has never been run in its own configuration, Sotopia-RL's reward
model is at chance and its evaluation was never run at all, and Sotopia-ToM, Ω and DAT
have never been executed. (The runbook itself still enumerates five arms; DAT is scored by
`compute_extended_metrics.py` but is not yet a row in that PDF.)

---

## Sources

| arm | paper |
|---|---|
| PPDPP | [arXiv:2311.00262](https://arxiv.org/abs/2311.00262) |
| EPO | [arXiv:2502.12486](https://arxiv.org/abs/2502.12486) |
| Sotopia-RL | [arXiv:2508.03905](https://arxiv.org/abs/2508.03905) |
| Sotopia-ToM | [arXiv:2605.02307](https://arxiv.org/abs/2605.02307) |
| SOTOPIA-Ω | [ACL 2025](https://aclanthology.org/2025.acl-long.1203/) · [arXiv:2502.15538](https://arxiv.org/abs/2502.15538) |
| DAT | *Dialogue Action Tokens*, ICLR 2025 submission 8922 (double-blind at the time of writing) |
