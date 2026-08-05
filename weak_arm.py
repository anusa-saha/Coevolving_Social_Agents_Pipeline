import random

import config
import weak_arm_model
from llm_clients import extract_json
from prompt_builder import build_weak_arm_prompt
from grader import grade_content


def run_weak_arm_rollout(scenario: dict) -> dict:
    temperature = round(random.uniform(*config.WEAK_ARM_TEMPERATURE_RANGE), 3)
    messages = build_weak_arm_prompt(scenario)
    raw = weak_arm_model.generate(messages, temperature=temperature)

    try:
        settlement = extract_json(raw)
        content_results = grade_content(scenario, settlement)
        passed = bool(content_results) and all(content_results.values())
    except Exception as e:
        settlement, content_results, passed = None, {}, False
        raw = f"{raw}\n\n[PARSE_ERROR] {e}"

    return {
        "arm": "weak",
        "temperature": temperature,
        "raw_output": raw,
        "settlement": settlement,
        "content_results": content_results,
        "passed": passed,
    }


def run_weak_arm(scenario: dict) -> dict:
    rollouts = [run_weak_arm_rollout(scenario) for _ in range(config.WEAK_ARM_ROLLOUTS)]
    pass_count = sum(r["passed"] for r in rollouts)
    gate_passed = pass_count <= config.WEAK_ARM_MAX_PASS
    return {"rollouts": rollouts, "pass_count": pass_count, "gate_passed": gate_passed}
