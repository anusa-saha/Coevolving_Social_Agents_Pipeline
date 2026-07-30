"""
Verifier: one cheap LLM call (gpt-5.4) that checks a candidate scenario before any
rollouts are run. Prompt lives in prompts/verifier_prompt.md, per constraint #1.
"""
import json
from pathlib import Path

from config import CONFIG
from llm_clients import verifier_client, chat_completion, extract_json
from grader import validate_scenario_checks


def _load_prompt(name: str) -> str:
    return Path(CONFIG.prompts_dir, name).read_text()


VERIFIER_SYSTEM_PROMPT = _load_prompt("verifier_prompt.md")


def run_verifier(scenario: dict) -> dict:
    # Fast, local, free check first: are the check strings even valid Python?
    try:
        validate_scenario_checks(scenario)
    except ValueError as e:
        return {
            "verdict": "REJECT",
            "tag": "MALFORMED",
            "checks": {"checks_are_valid_python": False},
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
    raw = chat_completion(
        verifier_client,
        model=CONFIG.verifier_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        json_mode=True,
    )
    return extract_json(raw)
