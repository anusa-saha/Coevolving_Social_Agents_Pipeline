import json
from pathlib import Path

import config
from llm_clients import gpt_chat, extract_json
from grader import validate_scenario_checks


def _load_prompt(name: str) -> str:
    return Path(config.PROMPTS_DIR, name).read_text()


VERIFIER_SYSTEM_PROMPT = _load_prompt("verifier_prompt.md")


def run_verifier(scenario: dict) -> dict:
    try:
        validate_scenario_checks(scenario)
    except ValueError as e:
        return {
            "verdict": "REJECT",
            "tag": "MALFORMED",
            "diagnosis": f"A check is not a valid Python boolean expression: {e}",
            "evidence": str(e),
            "fix_instructions": (
                "Rewrite the offending check so it only uses decisions, credited_facts, commitments, "
                "justification_fact_ids, revealed, with every string literal quoted."
            ),
        }

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]
    raw = gpt_chat(
        model=config.VERIFIER_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        json_mode=True,
    )
    return extract_json(raw)
