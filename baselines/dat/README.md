# DAT — Dialogue Action Tokens, on CSA

Freeze the language model. Train a two-layer MLP that reads the chair's own hidden state
and emits **two continuous prefix embeddings** in front of its prompt. Fit that planner by
cloning the unsteered model, then move it with offline RL against the verifier.

Paper: *Dialogue Action Tokens: Steering Language Models in Goal-Directed Dialogue with a
Multi-Turn Planner* (ICLR 2025 submission 8922).

The arm exists to answer one question the other five cannot: **how much of the hidden-
profile task is reachable without changing a single language-model parameter, a single
word of the prompt, or a single discrete act?** PPDPP picks acts, EPO writes strategies,
Sotopia-ToM rewrites the prompt, Ω rewrites the training corpus. DAT changes nothing that
is readable — it adds 2 × 3584 floats in front of the chair's prompt and optimises them in
a 64-dimensional space.

---

## The mapping

| paper | CSA |
|---|---|
| steered agent Q | the **chair** (`case['decision_maker']`) |
| partner agent P | the advisors — same frozen Qwen2.5-7B-Instruct |
| one round | one chair turn plus the advisor replies before its next slot |
| judge model reward | `csa_core.verifier`, via `decisive_facts` — **no judge is ever called** |
| Sotopia's 7-dimension GPT-4 score | `dca`, disclosure, and the settlement gate |

The reward is the one place CSA is *better* off than the paper. DAT's limitations section
names cheap stable reward signals as the real constraint; CSA ships executable checks and
states which facts flip which of them, so per-turn credit is a lookup. The rule is
byte-identical to EPO's verifier PRM and `selftest.py` asserts it on 400 random traces —
two RL arms supervised by different rewards would not be comparable.

---

## Three arms, one checkpoint

| arm | prefix | what it is |
|---|---|---|
| `unsteered` | none | the control. The paper's `Llama-2-7B-chat` row |
| `selfclone` | `pi_phi(s) W` | the paper's `w/ self-clone` row. Should land **on top of** the control |
| `dat` | `(pi_phi(s) + pi_phi_rl(s)) W` | the trained arm |

`selfclone` and `dat` load the **same** checkpoint, with the RL head forced to zero for the
former, so the two provably differ by nothing but `pi_phi_rl`. Read `selfclone` vs
`unsteered` first: stage 1 is supposed to *preserve* behaviour, so a gap there is a bug in
the clone, and it makes the `dat` row uninterpretable.

---

## Run it

```bash
cd dat
python selftest.py                       # ~1 min, no GPU, 23 checks

python selfclone.py --episodes 120       # stage 1: collect, then fit pi_phi and W
python collect_buffer.py --episodes 400  # stage 2a: the offline buffer, with N(0,0.25)
python train_dat.py --steps 4000         # stage 2b: TD3+BC. No LM is loaded here
python run_dat.py --arms unsteered selfclone dat --split test
```

`run_dat.py --compare` re-scores whatever is already on disk and runs nothing.

Every stage writes to `dat/logs/`; checkpoints land in `dat/ckpt/` (`selfclone.pt` after
stage 1, `dat.pt` after stage 2). The planner is ~2.5 M parameters — about 10 MB, against
EPO's 150–300 MB of LoRA adapters.

### Where each stage's cost goes

| stage | cost | why |
|---|---|---|
| `selfclone.py` collect | ~120 episodes of generation | ordinary rollouts, no gradient |
| `selfclone.py` train | 1 backward **through the frozen 7B** per utterance | the gradient has to reach a prefix at position 0, so activation memory is the binding constraint — gradient checkpointing is on by default |
| `collect_buffer.py` | ~400 episodes of generation | the entire interaction budget of the RL stage |
| `train_dat.py` | minutes on CPU | it reads `buffer.npz` and never loads a model |

On a 24 GB card everything fits. On less, `--accum` and `config.selfclone_max_ctx` are the
levers, in that order; truncating the context clips the front of the chair's persona, which
is exactly what the state is supposed to encode, so it is the last resort.

---

## Departures from the paper, and why

| paper | here | reason |
|---|---|---|
| Llama-2-7B-chat steered, Llama-3-8B partner | Qwen2.5-7B-Instruct for **every** role | the other five arms use it; changing the backend would make the DAT row incomparable |
| 10,000 episodes (~30k transitions) in the buffer | 400 episodes (~1,200) | GPU budget. Figure 4 shows ASR still climbing at 80k steps, so **this is the most under-resourced number in the arm** and the first to raise |
| TD3+BC for 1 epoch | 4,000 gradient steps | one pass over a 25× smaller buffer is ~5 updates, which trains nothing |
| reward from GPT-4 / a fine-tuned judge | `csa_core.verifier` | deterministic, free, and checkable — see above |
| two-party dialogue | 3–5 agents, one chair | CSA is a meeting; only the chair is steered, and the advisors are environment |
| goal defined by the Sotopia rubric | hidden-profile pooling | the goal is to elicit facts the chair cannot see; steering has to buy elicitation, not eloquence |

Implemented but not on by default: **Appendix A** (`--pca_up`), which initialises `W` from
the embedding matrix's principal components and removes the need for stage 1 entirely. It
is the thing to try when the clone loss will not come down.

Appendix B is on by default: the residual planner (`a = pi_phi(s) + pi_phi_rl(ŝ)`, with
`pi_phi` frozen and the RL head zero-initialised) and a weighted MSE in the critic. The
paper does not say what the weighting is, so `td3_reward_weight` upweights transitions
that carry a non-zero reward — most CSA turns earn nothing, and an unweighted critic
reaches a low loss by predicting zero everywhere. Set it to `1.0` to disable.

---

## What to watch

**`selfclone` must not beat `unsteered` by much.** It is a clone. A large gain there means
the prefix is doing something the clone loss never asked for, and the honest reading is
that stage 1 is broken, not that it helped.

**`action_cos` near 1.0.** Printed by `run_dat.py`. It is the turn-to-turn cosine between
action vectors: at 1.0 the planner emits the same vector everywhere, which is a constant
prefix rather than a policy. The outcome metrics cannot see this and will look fine.

**`policy_shift.mean_action_norm` at zero** after `train_dat.py`. The RL head never left
its zero initialisation, so `dat` is `selfclone` with extra steps. Usually the buffer has
almost no reward in it — `collect_buffer.py` warns when under 5 % of transitions carry any.

**Language health.** `run_dat.py` prints distinct-1/2 and utterance length per arm. DAT's
opening argument is that RL over language degrades the language distribution and that a
frozen LM with a short prefix does not. That is a claim about the *text*: if the steered
chair's distinct-2 collapses against the control's, the arm has bought its reward in the
way the paper says should be impossible, and the finding is worth more than the reward.

**`act_history` is deliberately absent from the records.** DAT's action is a vector in
R^64; there is no act it chose. The detector's read of each chair turn is stored as
`derived_acts` instead, so it can never be mistaken for PPDPP's or EPO's planner decisions
in the cross-arm act tables.

---

## Metrics

Records use the shared schema, so both catalogue scripts run on them unchanged:

```bash
cd analysis && python compute_extended_metrics.py          # picks DAT up automatically
python ppdpp/compute_all_metrics.py --records dat/logs --glob 'Record-dat-*.txt'
```

`metrics_dat.py` adds what only this arm can report: action norms and their spread, the
turn-to-turn cosine, planner forwards per episode, the elicitation rate read off the
detector, and the paired steering effect against the control.

One cost note for fairness: steering costs **one extra forward pass** per chair turn (the
state extraction), and no extra generation. `n_calls` counts generation calls, so it stays
comparable across arms; the extra forwards are counted separately as `planner_forwards`.
