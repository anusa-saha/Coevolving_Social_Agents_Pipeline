# PPDPP on CSA

[PPDPP](https://arxiv.org/abs/2311.00262) (Plug-and-Play Dialogue Policy Planner) ported to
the CSA hidden-profile benchmark. A RoBERTa-large classifier picks one of four dialogue
acts for the chair; a frozen Qwen2.5-7B renders the act into an utterance; REINFORCE tunes
the classifier against an episode reward.

This is the oldest arm in the repo and the only one that is vendored upstream code
modified in place rather than written against `csa_core`. That shows: the module layout is
the original's, and `env.py` still carries its own inline copy of the disclosure detector.
`csa_core.detectors.assert_matches_ppdpp()` exists to catch that copy drifting.

## The four acts

`ask`, `followup`, `share`, `decide` — defined in `prompt.py` alongside the prompt layer.
EPO imports that same prompt layer so the dialogue agent is byte-identical across the two
arms; that import is the one cross-arm dependency in the repo and it is deliberate.

## Runtime

| file | role |
|---|---|
| `run.py` | the RL loop and evaluation entry point |
| `env.py` | the CSA meeting: turn order, view filtering, disclosure tracking |
| `agent.py` | the RoBERTa planner, REINFORCE update, zero-gradient detection |
| `prompt.py` | act taxonomy and prompt rendering for the dialogue agent |
| scoring | `csa_core.verifier` — shared with every other arm |
| `data_reader.py`, `utils.py` | dataset plumbing the upstream loop expects |
| `sft.py` | supervised warm start for the planner |

```bash
python run.py --csa_reward critic --seed 1 --epochs 6
```

`--csa_reward` selects the reward:

* `critic` — PPDPP's own dense per-turn signal. **This is the paper's configuration and
  the default.**
* `verifier` — the sparse shared episode reward the other arms use.

Every stored record in the archive used `verifier`, so PPDPP's native configuration has
never actually been run on CSA. The flat learning curve currently attributed to the method
measures the reward config, not the algorithm. `run.py` prints
`zero-gradient updates: X%` beside each curve; under a sparse constant reward that
fraction is high, and it should drop sharply under `critic`.

## Building the training material

These ran once, in this order, to manufacture the planner's supervised data. They are kept
because the corpus is not reproducible without them, not because they are part of a normal
run.

| step | file |
|---|---|
| 1 | `export_csa.py` — CSA into the line-per-dict format `utils.load_dataset` expects |
| 2 | `generate_conversations.py` — roll out and score meeting transcripts |
| 3 | `annotate_intents.py` — label chair turns with one of the four acts (API annotator) |
| 4 | `audit_annotations.py` — re-audit those labels offline |
| 5 | `build_sft_splits.py` — 80/10/10 into the format `sft.py` consumes |
| 6 | `make_sft_data.py` — filtered behaviour cloning |

`annotate_intents.py` reads `OPENROUTER_API_KEY` from the environment. Never put a key in
a file here.

## Diagnostics

Written to answer specific questions during the port; each is standalone.

| file | question |
|---|---|
| `eval_planner.py` | how does the planner do on held-out annotated turns? |
| `full_metrics.py` | everything a reviewer might ask about the classifier, with intervals |
| `analyse_label_structure.py` | where in a meeting does each act occur? |
| `diagnose_followup.py` | why does the planner never predict `followup`? |
| `logit_adjust.py` | can the rare classes be recovered post-hoc, without retraining? |
| `finalise_planner.py` | the final Stage-1 result, reported once on test |
| `compute_all_metrics.py` | the metric catalogue from RL episode records |
| `smoke_csa.py` | end-to-end check with a scripted LLM in place of a real one |
| `make_report.py` | builds the architecture + results PDF |

## Data

`data/csa-{train,valid,test}.txt` is the scenario split (363/33/154) in the line-per-dict
format the upstream loader wants. It is **generated, not authored** — `export_csa.py`
serialises the split that `csa_core.data_csa` owns, and recomputing it here would be how
PPDPP silently ends up on a different benchmark from the other four arms.

Regenerate it after any change to the domain list or the per-domain cap:

```bash
python export_csa.py --out_dir ./data
```

Note the filename collision with the archived `data_sft/csa-*.txt`, which is turn-level SFT
rows (225/54/45), not scenarios.
