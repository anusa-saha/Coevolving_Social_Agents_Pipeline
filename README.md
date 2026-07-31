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

```bash
python main.py --n 3                      # attempt 3 accepted scenarios
python main.py --n 1 --scenario-type staffing
```

## How each model is used

- **Challenger + Verifier -> gpt-5.4.** `challenger.py` and `verifier.py` both
  call `llm_clients.gpt_chat()`, which is just `OpenAI().chat.completions.create(...)`.
  Their system prompts live in `prompts/challenger_prompt.md` and
  `prompts/verifier_prompt.md`.

- **Weak / lone arm -> DeepSeek-R1-Distill-Qwen-7B, on your GPU.**
  `weak_arm_model.py` loads the model once with `transformers`
  (`AutoModelForCausalLM.from_pretrained(..., device_map="cuda")`) and generates
  with a plain `model.generate()` call. `weak_arm.py` calls this directly --
  no HTTP request anywhere.

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
weak_arm.py                     # lone-agent rollouts (calls weak_arm_model.generate)
strong_arm.py                    # gpt-5.4 multi-agent turn loop
verifier.py                       # Gate 1
challenger.py                      # scenario generation + revision-on-feedback
storage.py                          # append-one-at-a-time JSON + transcript persistence
pipeline.py                          # the outer loop tying every stage together
main.py                                # CLI entry point
output/                                # created at runtime
```

## Things to know before running

- **GPU memory:** the 7B model needs roughly 15-16 GB of VRAM in fp16. If you
  hit an out-of-memory error, either use a GPU with more headroom or load in
  8-bit (`load_in_8bit=True` via `bitsandbytes`) in `weak_arm_model.py`.
- **First run is slow:** the model (~15 GB of weights) downloads from Hugging
  Face the first time `weak_arm_model.generate()` is called, then stays cached
  locally and loaded in memory for the rest of the process.
- **Reasoning tokens:** DeepSeek-R1-Distill emits a `<think>...</think>` block
  before its actual answer. `WEAK_ARM_MAX_NEW_TOKENS` in `config.py` is set to
  4000 to leave room for both the reasoning and the JSON settlement -- raise it
  if rollouts are getting cut off before the model reaches its answer.
- **`max_refinement_rounds`** (default 10, in `config.py`) caps how many
  revise-and-retry cycles one scenario gets before the pipeline gives up and
  returns `"exhausted"` -- every round is still fully recorded either way.
