# Scenario-Generation Pipeline

A runnable implementation of the Challenger → Verifier → Weak arm → Strong arm →
accept/reject loop, wired to your specific constraints.

## How each constraint is satisfied

1. **Challenger + Verifier prompts are `.md` files, gpt-5.4.**
   `prompts/challenger_prompt.md` and `prompts/verifier_prompt.md` are plain
   Markdown, loaded at runtime in `challenger.py` / `verifier.py`. Both are called
   through `CHALLENGER_MODEL` / `VERIFIER_MODEL` in `config.py`, both defaulting to
   `gpt-5.4`.

2. **Weak / lone arm uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.**
   `weak_arm.py` calls `weak_arm_client`, which points at `WEAK_ARM_BASE_URL` in
   `config.py`. Point that at wherever you're actually serving the model
   (a local vLLM/TGI server, or a hosted OpenAI-compatible provider) — the code
   doesn't care, as long as the endpoint speaks the OpenAI chat completions API.
   Because R1-distill models emit a `<think>...</think>` block before their
   answer, `llm_clients.extract_json()` strips that out before parsing.

3. **Strong arm — every agent uses gpt-5.4.**
   `strong_arm.py` calls the same `strong_arm_client`/`STRONG_ARM_MODEL` for
   every turn, regardless of which agent is "speaking" — it's one model wearing
   a different prompt each turn, per the source design.

4. **Prompts are automated from the JSON scenario entry.**
   `prompt_builder.py` contains zero scenario-specific text. `build_weak_arm_prompt`
   and `build_turn_prompt` only ever read `agents`, `shared_context`,
   `private_facts`, `views`, `decision_maker`, and `settlement_schema` off the
   scenario dict. Drop in a brand-new scenario and both prompt builders work
   immediately, with no code changes.

5. **All strong-arm conversations are saved.**
   Every rollout's full `transcript` (every `say`/`reveal`/`settle` event, plus
   raw model output per turn) is written to
   `output/transcripts/<scenario_id>/round_<n>_strong_arm_rollout_<i>.json` via
   `storage.save_transcript`, called right after each rollout finishes.

6. **All rejected and accepted entries, at every stage, are tracked.**
   - `output/all_iterations.json` — every single attempt, at every stage
     (verifier reject, weak-arm reject, strong-arm reject, or accept), with a
     timestamp and (where applicable) a `reject_tag` of `MALFORMED`, `LEAKED`,
     or `UNCOORDINATED`.
   - `output/rejected.json` — just the rejected attempts, same tagging.
   - `output/accepted.json` — just the accepted scenarios.
   - `output/transcripts/<scenario_id>/` — every weak-arm and strong-arm rollout
     from every round, so you can inspect exactly what led to each verdict.

7. **Entries are added to the JSON file one at a time.**
   `storage.append_json_array()` is called immediately after each stage
   resolves (not batched at the end of the whole run). Since a JSON array has
   to remain one valid file, "one at a time" means: read the array, append the
   new item, atomically rewrite the file (`os.replace`) — so a crash mid-run
   never loses or corrupts anything already recorded.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --n 3                      # attempt 3 accepted scenarios
python main.py --n 1 --scenario-type staffing
```

## File map

```
prompts/
  challenger_prompt.md   # Challenger system prompt (constraint #1)
  verifier_prompt.md     # Verifier system prompt (constraint #1)
config.py                # every model, endpoint, temperature, and gate threshold
llm_clients.py            # OpenAI-compatible client wrappers + JSON/reasoning-block extraction
prompt_builder.py         # automated per-agent prompt construction (constraint #4)
grader.py                 # safe eval() of content_checks / provenance_checks
weak_arm.py                # DeepSeek-R1-Distill-Qwen-7B lone-agent rollouts (constraint #2)
strong_arm.py              # gpt-5.4 multi-agent turn loop (constraint #3)
verifier.py                 # Gate 1
challenger.py                # scenario generation + revision-on-feedback
storage.py                    # append-one-at-a-time JSON + transcript persistence (constraints #6, #7)
pipeline.py                    # the outer loop tying every stage together
main.py                          # CLI entry point
output/                            # created at runtime: accepted.json, rejected.json,
                                    # all_iterations.json, transcripts/
```

## Notes / things to double check before production use

- `WEAK_ARM_BASE_URL` defaults to `http://localhost:8000/v1` — you need an actual
  server (e.g. `vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) running
  there, or point it at a hosted provider that serves this model.
- `gpt-5.4` is used as a literal model string throughout — swap `CHALLENGER_MODEL`
  / `VERIFIER_MODEL` / `STRONG_ARM_MODEL` in `.env` if the real model identifier
  differs from what your API expects.
- `max_refinement_rounds` (default 10) caps how many revise-and-retry cycles a
  single scenario gets before the pipeline gives up and returns `"exhausted"` —
  every one of those rounds is still fully recorded in `all_iterations.json`.
- The turn loop currently allows `settle` for a decision-maker once every
  non-decision-maker agent has spoken at least once — this generically encodes
  the "everyone is consulted before final sign-off" norm without hardcoding any
  scenario-specific logic.
