# Scenario-Generation Pipeline

Runs the Challenger -> Verifier -> Weak arm -> Strong arm -> accept/reject loop.

No API endpoints or servers to configure. gpt-5.4 is called through the normal
OpenAI client; DeepSeek-R1-Distill-Qwen-7B is loaded directly into your GPU's
memory in the same Python process, no server involved.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

That's the entire setup. The weak-arm model downloads from Hugging Face the
first time it's used and stays loaded in GPU memory for the rest of the run.

## Run

**Option A: run every stage automatically, end to end.**

```bash
python main.py --n 3                      # attempt 3 accepted scenarios
python main.py --n 1 --scenario-type staffing
```

**Option B: run each stage yourself, one command at a time.**

Each stage is its own subcommand in `cli.py`. Each one reads scenarios from a
file, runs its own revise-and-retry feedback loop against the Challenger, and
writes out two files: one for everything that passed, one for everything that
was rejected along the way. Feed one stage's "accepted" file into the next
stage's `--in`:

```bash
# 1. Generate candidate scenarios
python cli.py challenger --n 5 --out output/challenger_scenarios.json

# 2. Verify them (revises + retries on MALFORMED automatically)
python cli.py verifier \
  --in output/challenger_scenarios.json \
  --out-accepted output/verifier_accepted.json \
  --out-rejected output/verifier_rejected.json

# 3. Run the weak (lone) arm (revises + retries on LEAKED automatically)
python cli.py weak-arm \
  --in output/verifier_accepted.json \
  --out-accepted output/weak_arm_accepted.json \
  --out-rejected output/weak_arm_rejected.json

# 4. Run the strong arm (revises + retries on UNCOORDINATED automatically)
python cli.py strong-arm \
  --in output/weak_arm_accepted.json \
  --out-accepted output/strong_arm_accepted.json \
  --out-rejected output/strong_arm_rejected.json
```

`output/strong_arm_accepted.json` at the end of that chain is your final
dataset. Every intermediate attempt -- passed or rejected, at every stage, at
every round -- is also logged into `output/all_iterations.json`,
`output/accepted.json`, and `output/rejected.json`, and every rollout's full
transcript is saved under `output/transcripts/<scenario_id>/`.

Useful flags on every stage subcommand (`verifier` / `weak-arm` / `strong-arm`):
- `--max-rounds N` -- how many times that stage will revise-and-retry a single
  scenario before giving up on it (default: `config.MAX_REFINEMENT_ROUNDS`).
- A scenario that runs out of rounds without passing is **not** included in
  `--out-accepted` -- but every attempt it made is still in `--out-rejected`
  and in `output/all_iterations.json`, so nothing is silently lost.

You can feed the output of any stage into a later stage directly (both raw
scenario JSON and the wrapped `{"scenario": {...}, ...}` records from
`cli.py`'s own outputs are accepted as input).

## How each model is used

- **Challenger + Verifier -> gpt-5.4.** `challenger.py` and `verifier.py` both
  call `llm_clients.gpt_chat()`, which is just `OpenAI().chat.completions.create(...)`.
  Their system prompts live in `prompts/challenger_prompt.md` and
  `prompts/verifier_prompt.md`.

- **Weak / lone arm -> Qwen/Qwen3.5-9B, on your GPU.**
  `weak_arm_model.py` loads the model once with `transformers`
  (`AutoModelForCausalLM.from_pretrained(..., device_map="cuda")`) and generates
  with a plain `model.generate()` call, using these sampling parameters:
  `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
  repetition_penalty=1.0`. `weak_arm.py` calls this directly -- no HTTP request
  anywhere.

  **A note on `presence_penalty`:** `transformers`' `generate()` has no native
  `presence_penalty` parameter -- it only has `repetition_penalty`, which is a
  different mechanism (it *scales* a token's logit every time the token
  reoccurs, rather than applying one flat penalty the first time it appears).
  Since you asked for both, `weak_arm_model.py` implements `presence_penalty`
  itself as a small custom `LogitsProcessor` (`PresencePenaltyLogitsProcessor`)
  that subtracts a flat `1.5` from any token already seen in the sequence so
  far, applied alongside `repetition_penalty=1.0` (a no-op at that value) --
  matching OpenAI/vLLM's `presence_penalty` semantics exactly rather than
  approximating it.

  **A note on the model itself:** Qwen3.5-9B uses a newer hybrid architecture
  (Gated DeltaNet linear-attention layers mixed with regular Gated Attention
  layers). If `from_pretrained` fails with an unrecognized-architecture error,
  it means your installed `transformers` predates support for it -- upgrade
  with:
  ```bash
  pip install -U git+https://github.com/huggingface/transformers.git
  ```
  `trust_remote_code=True` is already passed as a safety net in case the model
  repo still ships custom modeling code when you pull it.

- **Strong arm -> gpt-5.4 for every agent.** `strong_arm.py` calls the same
  `gpt_chat()` function each turn, just with a different agent's prompt --
  one model wearing different hats.

- **Prompts are automated from the scenario JSON.** `prompt_builder.py`
  contains no scenario-specific text -- it only ever reads `agents`,
  `shared_context`, `private_facts`, `views`, `decision_maker`, and
  `settlement_schema` off whatever scenario dict it's given.

## What gets saved, and when

- `output/accepted.json` -- every scenario that cleared all three gates.
- `output/rejected.json` -- every rejected attempt, tagged `MALFORMED`,
  `LEAKED`, or `UNCOORDINATED`.
- `output/all_iterations.json` -- every single attempt at every stage
  (superset of the two above), across every round.
- `output/transcripts/<scenario_id>/round_<n>_<arm>_rollout_<i>.json` -- the
  full transcript/output of every weak-arm and strong-arm rollout, every round.

Every one of these is written immediately via `storage.append_json_array()`
right after that stage's result is known -- not batched up at the end of the
run -- so a crash mid-run doesn't lose anything already processed.

**Every logged entry (accepted, rejected, and every iteration) includes the
actual JSON that stage generated, not just a text summary:**
- **Verifier entries** include `raw_verdict` -- the verifier's full JSON
  response, including the per-sub-check booleans (`leakage_free`,
  `satisfiable`, `falsifiable`, etc.).
- **Weak-arm and strong-arm entries** include `rollouts` -- a list with each
  rollout's actual generated settlement, its per-check pass/fail results, and
  (for the strong arm) the full transcript and which facts were revealed. If a
  rollout's output failed to parse as JSON, the raw text and parse error are
  right there too.
- All entries also carry `evidence_data` -- the structured numbers behind the
  text `diagnosis`/`evidence` (per-check pass rates, which decisive facts were
  never revealed, etc. -- see `feedback.py`).

This means you can open `rejected.json` alone and see exactly what went wrong
at every round, without needing to cross-reference the separate transcript
files under `output/transcripts/`.

## The generalized feedback signal

Every stage's rejection now goes through one function:
`feedback.build_feedback(stage, scenario, result)`. Instead of a hand-written
sentence per stage, it builds the diagnosis, evidence, and fix instructions
from the actual rollout data:

- **Per-check pass rates** across every rollout (e.g. `C4: 0/4`, `C2: 3/4`) --
  so the Challenger sees exactly which check is the problem, not just an
  overall pass count.
- **Which decisive facts never got revealed, and by whom** -- cross-referenced
  against the scenario's own `decisive_facts` field, so a `strong_arm`
  rejection can say "the agent who never got their fact into the conversation
  is A3" instead of a generic "add a consultation norm."
- **A real example** of a failing settlement (and, for the strong arm, the
  last few turns of an actual failing transcript) -- concrete evidence, not
  just statistics.

`build_feedback` returns the same shape regardless of stage:
```python
{
    "stage": "...",
    "reject_tag": "...",
    "diagnosis": "...",        # one or two sentence summary
    "evidence": "...",         # the check tallies + never-revealed facts + example, as text
    "evidence_data": {...},    # the same evidence as raw structured data (for logging/analysis)
    "fix_instructions": "...",
}
```
Both `cli.py` and `pipeline.py` call this same function -- there's no more
duplicated, stage-specific diagnosis text scattered across the codebase.

## File map

```
prompts/
  challenger_prompt.md   # Challenger system prompt
  verifier_prompt.md     # Verifier system prompt
config.py                # model names, rollout counts, gate thresholds
llm_clients.py            # gpt-5.4 client + JSON/reasoning-block parsing helpers
weak_arm_model.py          # loads DeepSeek-R1-Distill-Qwen-7B on the GPU, plain generate()
prompt_builder.py            # automated per-agent prompt construction
grader.py                      # safe eval() of content_checks / provenance_checks
feedback.py                      # generalized, data-driven feedback for revise_scenario()
weak_arm.py                     # lone-agent rollouts (calls weak_arm_model.generate)
strong_arm.py                    # gpt-5.4 multi-agent turn loop
verifier.py                       # Gate 1
challenger.py                      # scenario generation + revision-on-feedback
storage.py                          # append-one-at-a-time JSON + transcript persistence
pipeline.py                          # the outer loop tying every stage together (used by main.py)
main.py                                # CLI entry point: run the whole pipeline end to end
cli.py                                   # CLI entry point: run any single stage on its own
output/                                # created at runtime
```

## Things to know before running

- **GPU memory:** the 9B model needs roughly 18-20 GB of VRAM in bf16. If you
  hit an out-of-memory error, either use a GPU with more headroom or load in
  8-bit (`load_in_8bit=True` via `bitsandbytes`) in `weak_arm_model.py`.
- **First run is slow:** the model (~19 GB of weights) downloads from Hugging
  Face the first time `weak_arm_model.generate()` is called, then stays cached
  locally and loaded in memory for the rest of the process.
- **Reasoning tokens:** Qwen3.5-9B emits a `<think>...</think>` block before
  its actual answer (same as the earlier DeepSeek model this replaced).
  `WEAK_ARM_MAX_NEW_TOKENS` in `config.py` is set to 4000 to leave room for
  both the reasoning and the JSON settlement -- raise it if rollouts are
  getting cut off before the model reaches its answer.
- **Transformers version:** Qwen3.5's hybrid architecture is new enough that
  it may not be recognized by an older `transformers` release. If you hit an
  unrecognized-architecture error on load, install from source (see above).
- **`max_refinement_rounds`** (default 10, in `config.py`) caps how many
  revise-and-retry cycles one scenario gets before the pipeline gives up and
  returns `"exhausted"` -- every round is still fully recorded either way.
