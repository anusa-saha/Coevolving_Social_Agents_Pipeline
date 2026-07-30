"""
Challenger: writes and revises scenario objects.
Prompt lives in prompts/challenger_prompt.md, per constraint #1.
"""
import json
from pathlib import Path

from config import CONFIG
from llm_clients import challenger_client, chat_completion, extract_json


def _load_prompt(name: str) -> str:
    return Path(CONFIG.prompts_dir, name).read_text()


CHALLENGER_SYSTEM_PROMPT = _load_prompt("challenger_prompt.md")


def generate_scenario(scenario_type: str = None) -> dict:
    user_msg = "NUM_SCENARIOS = 1. Generate one NEW scenario."
    if scenario_type:
        user_msg += f" Prefer scenario_type: {scenario_type}."
    messages = [
        {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = chat_completion(
        challenger_client,
        model=CONFIG.challenger_model,
        messages=messages,
        temperature=1.0,
        max_tokens=4000,
        json_mode=True,
    )
    return extract_json(raw)


def build_feedback_message(reject_tag: str, diagnosis: str, evidence: str, fix_instructions: str) -> str:
    """Mirrors the feedback prompt template documented in prompts/challenger_prompt.md."""
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
    raw = chat_completion(
        challenger_client,
        model=CONFIG.challenger_model,
        messages=messages,
        temperature=1.0,
        max_tokens=4000,
        json_mode=True,
    )
    return extract_json(raw)
