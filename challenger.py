import json
from pathlib import Path

import config
import domain_loader
from llm_clients import gpt_chat, extract_json
from storage import next_scenario_id


def _load_prompt(name: str) -> str:
    return Path(config.PROMPTS_DIR, name).read_text()


CHALLENGER_SYSTEM_PROMPT = _load_prompt("challenger_prompt.md")


def _domain_messages(domain_key) -> list:
    if not domain_key:
        return []
    return [{"role": "system", "content": domain_loader.build_domain_block(domain_key)}]


def generate_scenario(scenario_type: str = None) -> dict:
    domain_key = domain_loader.resolve_domain(scenario_type)

    user_msg = "Generate 1 new scenario."
    if scenario_type and not domain_key:
        user_msg += f" Prefer scenario_type: {scenario_type}."
    elif domain_key:
        user_msg += f" Set it in the {domain_loader.display_name_for(domain_key)} domain."

    messages = (
        [{"role": "system", "content": CHALLENGER_SYSTEM_PROMPT}]
        + _domain_messages(domain_key)
        + [{"role": "user", "content": user_msg}]
    )
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=1.0,
        max_tokens=4000,
        json_mode=True,
    )
    scenario = extract_json(raw)
    scenario["scenario_id"] = next_scenario_id()
    if domain_key:
        scenario["domain"] = domain_key
    return scenario


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
    domain_key = previous_scenario.get("domain")
    feedback = build_feedback_message(reject_tag, diagnosis, evidence, fix_instructions)

    messages = (
        [{"role": "system", "content": CHALLENGER_SYSTEM_PROMPT}]
        + _domain_messages(domain_key)
        + [
            {"role": "user", "content": f"Here is your previous scenario:\n{json.dumps(previous_scenario, indent=2)}"},
            {"role": "user", "content": feedback},
        ]
    )
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=1.0,
        max_tokens=4000,
        json_mode=True,
    )
    scenario = extract_json(raw)
    scenario["scenario_id"] = previous_scenario["scenario_id"]
    if domain_key:
        scenario["domain"] = domain_key
    return scenario
