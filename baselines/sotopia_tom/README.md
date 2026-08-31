# Sotopia-ToM on CSA

[Sotopia-ToM](https://arxiv.org/abs/2605.02307) (Yashwanth YS, Wang, Zeng, Zhou, Onoue,
Varadarajan, Sap — CMU LTI / Fujitsu) evaluated on the CSA hidden-profile benchmark.

Unlike the other three baselines this one **trains nothing**. It is a prompting study:
the same frozen `Qwen2.5-7B-Instruct` plays every role, and the arms differ only in how
the chair's turn is prompted. That makes it the training-free reference the other arms
lack — the arm that answers *"does any of this training beat good prompting?"*

## Two things to know before reading any result

**This is a reimplementation from the paper, not their code.** The paper's "Code" link
points at the generic [sotopia](https://github.com/sotopia-lab/sotopia/) framework, and
its reproducibility statement says the Sotopia-ToM codebase *"will be made publicly
available"* — it was not, at the time of writing, and there is no `sotopia-tom` repo under
`sotopia-lab`. Every prompt template here is our reading of the paper's prose.

**The privacy half of the benchmark does not exist on CSA.** Sotopia-ToM agents have a
public *and* a private channel, and facts they must *not* share. CSA is a single public
meeting in which every private fact *should* be pooled. So `CPV` (critical privacy
violations) has no analogue and the composite is a three-way mean, not four-way. See
[metrics_tom.py](metrics_tom.py).

## The five arms

| `--strategies` | What | Source |
|---|---|---|
| `stripped` | no elicitation instruction at all | **ours** — the control |
| `basic` | pursue the decision, engage naturally, close when done | their vanilla |
| `cot` | CoT-Elicitation: goal progress → gaps → who holds it → missing-fact check → act | their CoT-Privacy, repointed |
| `tom_coach` | a separate LLM call reads the room, injected into the chair prompt | theirs |
| `tom_belief` | a persistent belief state, updated incrementally each turn | theirs |

**Why `stripped` exists and is not optional.** CSA's own chair prompt already says *"The
others hold information you do not have; it is your job to draw it out."* That is most of
what CoT and ToM scaffolding are meant to induce. Without a control that removes it, the
four real arms would very likely land on top of each other and the study would measure
nothing. Run `stripped` and `basic` first — if they don't separate, stop and report the
saturation rather than building out the rest.

**CoT-Privacy is renamed CoT-Elicitation.** Its leakage-check step is repointed at *"which
decisive fact is still missing"*, because on CSA there is nothing to withhold and a
leakage check would be a no-op consuming the strategy's reasoning budget.

## Metrics

Three of the paper's four dimensions have exact CSA analogues that need **no LLM judge**:

| Paper | Here | From |
|---|---|---|
| DA — disclosure alignment | decisive facts actually pooled | `disclosure_rate` |
| IA — inquiry alignment | disclosures that followed an eliciting turn | `reveal_elicited` |
| EFF — efficiency | how early pooling happened vs the turn budget | `reveal_turn` / `max_turn` |
| CPV — privacy violations | **absent** | — |

```
InfoMgmt3 = [ DA · IA · EFF ]^(1/3)
```

> `InfoMgmt3` is a **three-way** mean and reads higher than the paper's four-way
> `InfoMgmt` on identical behaviour. Use it to rank these five arms against each other.
> **Never compare it to the paper's table.**

`leaks` is deliberately *not* substituted for CPV — a leak here is an agent stating a fact
it was never shown (fabrication), not inappropriate disclosure. It's reported separately.

The composite is averaged **over episodes**, not computed from the averaged dimensions:
otherwise an episode that scored zero on one dimension gets rescued by the others, which
is exactly what a geometric mean exists to prevent.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python selftest.py
```

Self-contained — imports no code from the other baselines. Scenarios download from
[`anusasaha/Coevolving_Social_Agents`](https://huggingface.co/datasets/anusasaha/Coevolving_Social_Agents)
into `data/raw/` on first use; nothing needs exporting. `selftest.py` verifies the
rebuilt split is exactly 99/9/42 and matches the other arms, and that the verifier
reproduces published scores on 297 episodes.

No `peft` needed — nothing trains here.

## Running

The probe first. Two arms, 42 scenarios, about an hour:

```bash
python run_tom.py --strategies stripped basic --device cuda:0
```

If `stripped` and `basic` separate, run the rest:

```bash
python run_tom.py --strategies cot tom_coach tom_belief --device cuda:0
```

Score everything already on disk without re-running anything:

```bash
python run_tom.py --compare
```

Records land in `logs/Record-tom-<arm>-test.txt` in the same schema as the other
baselines, so `ppdpp_csa/compute_all_metrics.py` runs on them and every arm pairs per
scenario. The report prints a per-arm table plus paired sign tests against `stripped`.

## Cost

Five arms × 42 scenarios ≈ **210 episodes**. Baseline arms ~14 model calls per episode;
the ToM arms add one per chair turn (~4), so ~18. Roughly 3,400 calls total, one 7B model
resident, no training and no API. Hours, not days.

`n_calls` is recorded per episode for exactly the reason it is in the other arms: a
strategy that wins by making more calls has not won. The ToM arms are ~30% more expensive
per episode, and the comparison should be read at matched budget.

## Reading the result

In the paper, `IA` is **0.288 across every model** — agents share but almost never
deliberately ask. That is the gap ToM-Belief is meant to close, and it is the number to
watch here.

For reference, EPO's SFT checkpoint on the same 42 scenarios reaches `elicited_fraction`
0.919 with a chair prompt that explicitly instructs elicitation. If `stripped` lands far
below that and the interventions recover it, the scaffolds are doing real work. If
everything sits near 0.9, CSA's task instruction already does what the scaffolds do —
which is a finding about the two benchmarks' difficulty, and worth reporting as one.
