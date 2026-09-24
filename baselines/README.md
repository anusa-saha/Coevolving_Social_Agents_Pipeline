# CSA Baselines

Five social-agent methods, plus a method-free floor, run on **CSA**
([Coevolving Social Agents](https://huggingface.co/datasets/anusasaha/Coevolving_Social_Agents)):
a hidden-profile benchmark where one agent chairs a meeting, three to five advisors each
hold a private fact, and the chair must draw out enough of those facts to reach a
settlement that passes the scenario's executable checks.

Every arm uses the same model (`Qwen/Qwen3.5-9B`, thinking off), the same scenario split,
the same disclosure detector and the same verifier, so a difference between two rows is a
difference between two methods.

| arm | folder | what it is | trains? |
|---|---|---|---|
| Round Table | `roundtable/` | agents talk, the chair settles: the floor | no |
| Sotopia-ToM | `sotopia_tom/` | five prompting strategies for the chair | no |
| PPDPP | `ppdpp/` | RoBERTa act planner + REINFORCE | yes |
| EPO | `epo/` | LLM strategist + turn-level REINFORCE | yes |
| Sotopia-RL | `sotopia_rl/` | attributed rewards, reward model, GRPO | yes |
| SOTOPIA-Ω | `sotopia_omega/` | stall detection, slow-mode corpus, SFT | yes |

**Contents:** [1 Requirements](#1-requirements) · [2 Setup](#2-setup-once) ·
[3 Check the setup](#3-check-the-setup) · [4 Run everything](#4-run-everything-recommended) ·
[5 Run one arm by hand](#5-run-one-arm-by-hand) · [6 Results](#6-results) ·
[7 Troubleshooting](#7-troubleshooting) · [8 Reference](#8-reference)

---

## 1. Requirements

| | |
|---|---|
| GPUs | 4 × 24 GB (A5000) is what the scheduler is sized for. One card holds one 9B model; EPO's RL stage needs two cards. |
| OS | Linux (the parallel runner uses process groups) |
| Python | 3.10 or newer (transformers 5.x and peft require it) |
| Disk | ~20 GB for the model in the Hugging Face cache, plus a few GB of logs and adapters |
| Archive | `csa-artifacts/` next to this folder (or `CSA_ARTIFACTS_DIR`): PPDPP and EPO are supervised from its annotated chair turns |
| API keys | none required. `OPENROUTER_API_KEY` is optional (EPO strategy writing) |

---

## 2. Setup (once)

Run from this folder (`baselines/`).

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch with CUDA first — pick the wheel for your driver on pytorch.org; for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then the repo and every arm's extras:

```bash
pip install -e .
pip install -r ppdpp/requirements.txt -r epo/requirements.txt -r sotopia_rl/requirements.txt \
            -r sotopia_tom/requirements.txt -r sotopia_omega/requirements.txt -r roundtable/requirements.txt
```

Fast kernels for Qwen3.5's linear-attention layers (without them generation still works, but
much more slowly):

```bash
pip install flash-linear-attention causal-conv1d
```

Sentence-splitting data used by every arm:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

If the archive is not next to this folder:

```bash
export CSA_ARTIFACTS_DIR=/path/to/csa-artifacts
```

Optional — lets EPO write its strategy targets with the annotator model instead of templates:

```bash
export OPENROUTER_API_KEY=...        # never put a key in a file
```

The dataset needs no download. `data/splits/` holds the prescribed split (1100 scenario rows, 945 distinct) and is used whenever it is present; `first_50.json` (11 domains × 50) is the older, derived-split configuration and is used when `data/splits/` is absent or `CSA_SPLITS_DIR=""` turns it off.

---

## 3. Check the setup

Library versions, and whether anything blocks a run:

```bash
python csa_core/compat.py
```

Every arm's selftest (CPU, about a minute each, no model needed). Exit code 0 = all passed;
2 = logic passed but the GPU stack is missing; 1 = a real failure:

```bash
for a in epo sotopia_rl sotopia_tom sotopia_omega roundtable; do (cd $a && python selftest.py); done
```

Every import in the repo resolves (the only expected failures are the five in `eval.py`,
which belongs to the co-evolved system and is not run here):

```bash
python analysis/check_imports.py
```

Versions, kernels, GPUs, data, and the model downloads:

```bash
python parallel/preflight.py --prefetch
```

On one GPU, the real model: generation speed, a LoRA training step at 2,560 tokens, and that
the training loss is computed correctly. **Fails if a training step will not fit on the card.**

```bash
CUDA_VISIBLE_DEVICES=0 python parallel/preflight.py --probe
```

`parallel/run_all.py` runs both preflight steps for you as its first jobs.

---

## 4. Run everything (recommended)

`parallel/run_all.py` runs every stage of every arm across the GPUs: one job per card, EPO's RL
stage on two, the longest chains of work first, idle cards filled with short jobs. Finished jobs
are remembered, so re-running the same command resumes.

See the plan and an estimated timeline (about 63 h on 4 GPUs; the estimates are rough):

```bash
python parallel/run_all.py --dry_run
```

Start it (use `tmux` or `nohup` — it runs for days):

```bash
nohup python parallel/run_all.py > run_all.out 2>&1 &
```

Watch it:

```bash
python parallel/run_all.py --status
tail -f runs/run_all.log
tail -f runs/logs/sr.collect_train.log
```

Other useful forms:

```bash
python parallel/run_all.py --list                 # every job, its GPUs, estimate and dependencies
python parallel/run_all.py --only sr om           # some arms (their prerequisites come along)
python parallel/run_all.py --gpus 0,1             # use only these cards
python parallel/run_all.py --reset sr.grpo        # forget that a job finished, so it runs again
python parallel/run_all.py --force_gates          # start Ω's corpus even if its probe said STOP
```

Stop with `Ctrl-C` (or `kill` the runner): running jobs are stopped and will re-run next time.

Arm names for `--only`: `setup ppdpp epo sr tom om rt analysis`.

**Choices the runner makes for you:**

- EPO's strategies use the annotator API only if `OPENROUTER_API_KEY` is set; otherwise templates (`--fallback_only`).
- Sotopia-RL's GRPO uses the reward model only if its pair-ranking accuracy is ≥ 0.60; otherwise `lookahead`.
- SOTOPIA-Ω's corpus does not start if its probe printed `STOP`.
- Jobs run offline after the first job downloads the model (`HF_HUB_OFFLINE=1`).

State and logs: `runs/state/<job>.done`, `runs/logs/<job>.log`, `runs/status.json`.

---

## 5. Run one arm by hand

Every command runs from the arm's folder. `CUDA_VISIBLE_DEVICES` picks the card; inside the
job it is always `cuda:0` (and `cuda:1` for EPO's RL stage). Training commands keep
`--grad_checkpointing` — without it a 9B training step does not fit on 24 GB.

### 5.1 Round Table — 1 GPU, ~1 h

```bash
cd roundtable
CUDA_VISIBLE_DEVICES=0 python run_rt.py --backend local --decide chair --split test
```

### 5.2 Sotopia-ToM — 1 GPU per strategy, ~2–2.5 h each

```bash
cd sotopia_tom
CUDA_VISIBLE_DEVICES=0 python run_tom.py --strategies stripped   --split test --out logs/summary-stripped.json
CUDA_VISIBLE_DEVICES=1 python run_tom.py --strategies basic      --split test --out logs/summary-basic.json
CUDA_VISIBLE_DEVICES=2 python run_tom.py --strategies cot        --split test --out logs/summary-cot.json
CUDA_VISIBLE_DEVICES=3 python run_tom.py --strategies tom_coach  --split test --out logs/summary-tom_coach.json
CUDA_VISIBLE_DEVICES=0 python run_tom.py --strategies tom_belief --split test --out logs/summary-tom_belief.json
```

When all five have finished, compare them (CPU):

```bash
python run_tom.py --compare --split test
```

### 5.3 PPDPP — planner SFT, then two RL runs (1 GPU each)

```bash
cd ppdpp
```

1. Write the scenario split in PPDPP's format, and copy the planner SFT data keeping only
   train-split scenarios (both CPU):

   ```bash
   python export_csa.py --out_dir ./data
   python filter_sft_split.py
   ```

2. Train the planner (~0.5 h):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python sft.py --data_name csa --model_name roberta --model_name_or_path /scratch/rohank__iitp/roberta-large --data_dir data_sft --output_dir sft --do_train --do_eval --overwrite_output_dir --num_train_epochs 10 --max_seq_length 512 --gpu 0 
   ```

3. RL, one run per reward, in parallel (~34 h and ~44 h; each evaluates on the test split after every step):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python run.py --data_name csa --system qwen --user qwen --critic qwen --csa_reward verifier --seed 1 --max_steps 6 --max_turn 8 --do_train --do_eval --qwen_device_map cuda:0 
   CUDA_VISIBLE_DEVICES=0 python run.py --data_name csa --system qwen --user qwen --critic qwen --csa_reward critic --seed 1 --max_steps 6 --max_turn 8 --do_train --do_eval --qwen_device_map cuda:0 
   ```

### 5.4 EPO — strategy targets, SFT (1 GPU), RL (2 GPUs)

Needs `ppdpp/data_sft/` from step 5.3.1.

```bash
cd epo
```

1. Strategy targets (CPU). Drop `--fallback_only` to use the annotator API (`OPENROUTER_API_KEY`):

   ```bash
   python make_strategies.py --split train --fallback_only
   python make_strategies.py --split valid --fallback_only
   ```

2. SFT warm start (~1 h):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python sft_epo.py --epochs 3 --lr 1e-5 --accum 8 --class_balance sqrt_inverse --strategist_device cuda:0 --grad_checkpointing
   ```

3. A 3-episode plumbing check, then RL (~22 h on two cards):

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python run_epo.py --dry_run 3 --agent_device cuda:0 --strategist_device cuda:1 --grad_checkpointing
   CUDA_VISIBLE_DEVICES=0,1 python run_epo.py --episodes 700 --seed 1 --prm verifier --prm_mode binary --advantage group --eval_every 175 --eval_split test --agent_device cuda:0 --strategist_device cuda:1 --grad_checkpointing
   ```

### 5.5 Sotopia-RL — collect, reward model, GRPO, evaluate

```bash
cd sotopia_rl
```

1. Self-play episodes (~27 h and ~2.5 h; re-running the same command resumes):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python collect_episodes.py --split train --k 6 --keep 2
   CUDA_VISIBLE_DEVICES=1 python collect_episodes.py --split valid --k 6 --keep 2
   ```

2. Reward labels (CPU), then behaviour cloning and the reward model in parallel (~3 h and ~6 h):

   ```bash
   python make_rm_data.py --episodes data/episodes-train.jsonl
   CUDA_VISIBLE_DEVICES=0 python train_sft.py --epochs 3 --lr 1e-4 --accum 8 --grad_checkpointing
   CUDA_VISIBLE_DEVICES=1 python train_rm.py --epochs 8 --lr 5e-6 --holdout 0.15 --grad_checkpointing
   ```

3. **Gate.** Read the reward model's pair-ranking accuracy (chance is 0.50):

   ```bash
   python -c "import json; m=json.load(open('ckpt/rm/rm_meta.json')); print(m['best_pair_rank'], m['best_epoch'])"
   ```

4. GRPO (~17 h). If the number above is **≥ 0.60**:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_grpo.py --adapter ckpt/sft --rm ckpt/rm --reward_source rm --groups 175 --group 8 --kl_beta 0.02 --seed 1 --grad_checkpointing
   ```

   Otherwise:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_grpo.py --adapter ckpt/sft/policy --reward_source lookahead \
       --groups 175 --group 8 --kl_beta 0.02 --seed 1 --grad_checkpointing
   ```

5. Evaluate the untrained floor, the SFT model and the GRPO model (~2 h each). Use
   `grpo-rm-seed1` or `grpo-lookahead-seed1` to match step 4:

   ```bash
   CUDA_VISIBLE_DEVICES=1 python evaluate_sr.py --adapter "" --split test --tag base
   CUDA_VISIBLE_DEVICES=1 python evaluate_sr.py --adapter ckpt/sft/policy --split test --tag sft
   CUDA_VISIBLE_DEVICES=0 python evaluate_sr.py --adapter ckpt/grpo/grpo-rm-seed1/final/policy --split test --tag grpo
   ```

### 5.6 SOTOPIA-Ω — probe, corpus, SFT, evaluate

```bash
cd sotopia_omega
```

1. Probe (~0.3 h). **If it prints `STOP`, do not build the corpus** — slow mode is not helping:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python generate_omega.py --split train --probe 5 --expert local
   ```

2. Corpus (~36 h and ~3.5 h; re-running resumes):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python generate_omega.py --split train --expert local --k 6 --keep 2 --stall_after 1 --stall_patience 1 --seed 1
   CUDA_VISIBLE_DEVICES=0 python generate_omega.py --split valid --expert local --seed 1
   ```

3. Train the student (~3 h):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_sft_om.py --mode_filter all --grad_checkpointing
   ```

4. Evaluate: the floor, the student, the student with the slow scaffold on, and the leakage
   control (~2–3 h each). Report `omega-withhold` next to any Ω number:

   ```bash
   CUDA_VISIBLE_DEVICES=1 python evaluate_om.py --adapter "" --split test --tag base
   CUDA_VISIBLE_DEVICES=1 python evaluate_om.py --adapter ckpt/sft/student --split test --tag omega
   CUDA_VISIBLE_DEVICES=1 python evaluate_om.py --adapter ckpt/sft/student --split test --eval_mode adaptive --tag omega-adaptive
   CUDA_VISIBLE_DEVICES=1 python evaluate_om.py --adapter ckpt/sft/student --split test --opponent withhold --tag omega-withhold
   ```

---

## 6. Results

### Compare every arm

```bash
cd analysis
python compute_extended_metrics.py --out results.json
```

CPU only, seconds. It finds every arm's records and prints the headline table, each arm's
summary, and the paired bootstrap against the baseline, followed by dca/disclosure and the
detailed sections.

> **Old results are picked up too.** If `csa-artifacts/` sits next to this folder (or
> `CSA_ARTIFACTS_DIR` is set), the archived PPDPP and EPO records from the earlier
> 3-domain, Qwen2.5 runs are loaded alongside the new ones. Move or rename the archive while
> building tables from new runs.

### Headline metrics

The same set `eval.py` reports for the co-evolved system, defined once in `csa_core/headline.py`:

| block | metrics |
|---|---|
| Checks | content, provenance and all checks passed; `checks_frac`; `success` (every check passes) |
| Behaviour | facts revealed; decisive facts revealed; settled rate (`settled_by`: chair or extractor); turns to settle; chair turns used |
| Addressing | of the advisors the chair named, the share holding a decisive fact, and vice versa (stands in for eval.py's routing) |
| Guardrail | `leaks`: a speaker stating a private fact it was never shown |
| Bottleneck | unsettled / settled with some decisive facts missing / settled with all of them |
| Comparison | paired table and paired bootstrap (2,000 resamples, 95% CI) against the baseline |

### Where each arm writes

Every arm writes logs under its own folder:

```
<arm>/logs/eval/<run>/       conversations.log  episodes.jsonl  scenarios.csv  turns.csv  summary.log  summary.json
<arm>/logs/rollouts/<run>/   conversations.log  rollouts.jsonl  rollouts.csv              summary.log  summary.json
```

`conversations.log` is the readable transcript, with the headline metrics above each episode.

| arm | evaluation records | training rollouts | checkpoints |
|---|---|---|---|
| Round Table | `logs/Record-rt-*.txt` | — | — |
| Sotopia-ToM | `logs/Record-tom-<strategy>-test.txt` | — | — |
| PPDPP | `tmp/csa/eval_result/Record-epoch-*.txt` | every REINFORCE episode | `sft/`, `tmp/csa/RL-agent/` |
| EPO | `logs/Record-epo-*-ep*.txt` | every group rollout, with advantages | `ckpt/sft`, `ckpt/rl/` |
| Sotopia-RL | `logs/Record-<tag>-test.txt` | all k self-play rollouts; every GRPO candidate | `ckpt/sft`, `ckpt/rm`, `ckpt/grpo/` |
| SOTOPIA-Ω | `logs/Record-<tag>-test.txt` | all k corpus rollouts; the probe | `ckpt/sft` |

---

## 7. Troubleshooting

| symptom | what to do |
|---|---|
| `setup.probe` fails with "does not fit" | another process is on the card (`nvidia-smi`); otherwise keep `--grad_checkpointing` on and lower `--max_len` in the SFT scripts |
| generation is very slow | `python parallel/preflight.py --prefetch` shows whether `flash-linear-attention` / `causal-conv1d` import; the probe prints tokens per second |
| a job failed | read `runs/logs/<job>.log`, fix, run the same `run_all.py` command again — only unfinished jobs run |
| `ppdpp.sft_data` fails: no archived data_sft | set `CSA_ARTIFACTS_DIR` to the archive (PPDPP and EPO both need it) |
| Ω corpus is `gated` | the probe printed STOP; read `runs/logs/om.probe.log`. `--force_gates` overrides |
| a selftest exits 2 | logic is fine; torch/transformers/peft are missing or too old (`python csa_core/compat.py`) |
| out of disk | PPDPP keeps only its newest RL checkpoint; LoRA adapters are small; the model cache is ~20 GB |

---

## 8. Reference

### Benchmark configuration

- **945 distinct scenarios**, read from `data/splits/` rather than derived: **664 / 56** train / valid (carved out of `train.json` by the same seeded bucket procedure, stratified on domain and table size) and **380 test** = **180 `test_seen`** + **200 `test_unseen`**.
- `test_seen` covers the nine domains training also covers. `test_unseen` is `family_friends_informal` + `informal_commerce_bargaining`, absent from training entirely — it is the clean generalisation measurement.
- **155 of the 380 test scenarios also appear in `train.json`, byte-identical.** That overlap ships with the split files, so the loader keeps it and warns once per process instead of failing. Seen-domain test numbers are optimistic because of it; quote `test_unseen` when the question is generalisation.
- `--split` accepts `test_seen` and `test_unseen` everywhere it accepts `test`. `CSA_TEST_FILE=test_unseen.json` makes them the default `test`.
- Results from the earlier 550-scenario (`first_50.json`) and 3-domain / 150-scenario configurations are **not comparable** and must be re-run. Archived records are matched to scenarios by uid, and uid numbering differs between datasets, so a stale record can match a *different* scenario — the selftests detect this and skip rather than compare.
- PPDPP's and EPO's planner SFT data keeps only train-split scenarios (`ppdpp/filter_sft_split.py`): it was annotated under the old split, and 11 of its 75 training scenarios are test scenarios now. It covers the 3 original domains only.

### Environment variables

| variable | effect |
|---|---|
| `CSA_ARTIFACTS_DIR` | the archive: planner SFT data and published records for the selftests |
| `OPENROUTER_API_KEY` | EPO strategy targets from the annotator model (optional) |
| `OPENAI_API_KEY` | only for `run_rt.py --backend api` and Ω `--expert api` (optional, not comparable with the Qwen arms) |
| `CSA_ANNOTATOR_MODEL` | annotator model id (default `google/gemma-4-26b-a4b-it:free`) |
| `CSA_SCENARIOS_FILE` | a different combined scenario file (`''` = use `data/raw/`) |
| `CSA_RAW_DIR`, `CSA_SCENARIOS_PER_DOMAIN` | per-domain files, e.g. `100` scenarios per domain (needs the raw files) |
| `CSA_DOMAINS=published` | the original 3 domains (99 / 9 / 42) |
| `CSA_SPLITS_DIR` | the directory holding `train.json` and the three test files; `""` falls back to the derived split |
| `CSA_TEST_FILE` | which file `test` resolves to (default `test_all.json`; `test_unseen.json` to evaluate generalisation by default) |

### Layout

```
csa_core/        shared instrument: split, detectors, verifier, headline metrics, logs, version shims
data/splits/     the prescribed split: train.json, test_all.json, test_seen.json, test_unseen.json
first_50.json    the older scenario set, used only when data/splits/ is absent
parallel/        run_all.py (scheduler), jobs.py (every stage), preflight.py (checks + GPU probe)
roundtable/      Round Table (standalone: vendors csa_core as _*.py; python vendor.py refreshes it)
sotopia_tom/     Sotopia-ToM
ppdpp/           PPDPP
epo/             EPO
sotopia_rl/      Sotopia-RL
sotopia_omega/   SOTOPIA-Ω
analysis/        cross-arm metrics, import checker
docs/            DESIGN_NOTES.md: the reasoning behind each choice and what the archived runs showed
eval.py          the co-evolved system's evaluation (reference for the headline metrics; not run here)
```

Each arm's own `README.md` explains its method and flags in depth.

### Sources

| arm | paper |
|---|---|
| PPDPP | [arXiv:2311.00262](https://arxiv.org/abs/2311.00262) |
| EPO | [arXiv:2502.12486](https://arxiv.org/abs/2502.12486) |
| Sotopia-RL | [arXiv:2508.03905](https://arxiv.org/abs/2508.03905) |
| Sotopia-ToM | [arXiv:2605.02307](https://arxiv.org/abs/2605.02307) |
| SOTOPIA-Ω | [ACL 2025](https://aclanthology.org/2025.acl-long.1203/) · [arXiv:2502.15538](https://arxiv.org/abs/2502.15538) |
