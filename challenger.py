"""
Challenger: writes and revises scenario objects using gpt-5.4.
Prompt lives in prompts/challenger_prompt.md.
"""
import json
from pathlib import Path

import config
from llm_clients import gpt_chat, extract_json


def _load_prompt(name: str) -> str:
    return Path(config.PROMPTS_DIR, name).read_text()


CHALLENGER_SYSTEM_PROMPT = _load_prompt("challenger_prompt.md")


def generate_scenario(scenario_type: str = None) -> dict:
    user_msg = "Generate 1 new scenario."
    if scenario_type:
        user_msg += f" Prefer scenario_type: {scenario_type}."
    messages = [
        {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=1.0,
        max_tokens=4000,
        json_mode=True,
    )
    return extract_json(raw)


def build_feedback_message(reject_tag: str, diagnosis: str, evidence: str, fix_instructions: str) -> str:
    return (
        f"Your previous scenario was REJECTED. Tag: {reject_tag}\n\n"
        f"What went wrong: {diagnosis}\n\n"
        f"Evidence: {evidence}\n\n"
        f"What to fix: {fix_instructions}\n\n"
        f"Rules for your revision:\n"
        f"- Keep everything that already works. Change only what caused the rejection.\n"
        f"- Return the complete scenario object again, in the same JSON format as before.\n"
        f"- Return JSON only -- no explanation."
    )


def revise_scenario(previous_scenario: dict, reject_tag: str, diagnosis: str,
                     evidence: str, fix_instructions: str) -> dict:
    feedback = build_feedback_message(reject_tag, diagnosis, evidence, fix_instructions)
    messages = [
        {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is your previous scenario:\n{json.dumps(previous_scenario, indent=2)}"},
        {"role": "user", "content": feedback},
    ]
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=4000,
        json_mode=True,
    )
    return extract_json(raw)
