"""
Weak arm: the lone decision-maker, one API call, no conversation.

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (config.weak_arm_*), per constraint #2.
Graded by content checks only -- there is no transcript, so provenance can't apply.
"""
from config import CONFIG
from llm_clients import weak_arm_client, chat_completion, extract_json
from prompt_builder import build_weak_arm_prompt
from grader import grade_content


def run_weak_arm_rollout(scenario: dict) -> dict:
    messages = build_weak_arm_prompt(scenario)
    raw = chat_completion(
        weak_arm_client,
        model=CONFIG.weak_arm_model,
        messages=messages,
        temperature=CONFIG.weak_arm_temperature,
        max_tokens=CONFIG.weak_arm_max_tokens,
    )
    try:
        settlement = extract_json(raw)
        content_results = grade_content(scenario, settlement)
        passed = bool(content_results) and all(content_results.values())
    except Exception as e:
        settlement, content_results, passed = None, {}, False
        raw = f"{raw}\n\n[PARSE_ERROR] {e}"

    return {
        "arm": "weak",
        "raw_output": raw,
        "settlement": settlement,
        "content_results": content_results,
        "passed": passed,
    }


def run_weak_arm(scenario: dict) -> dict:
    rollouts = [run_weak_arm_rollout(scenario) for _ in range(CONFIG.weak_arm_rollouts)]
    pass_count = sum(r["passed"] for r in rollouts)
    gate_passed = pass_count <= CONFIG.weak_arm_max_pass
    return {"rollouts": rollouts, "pass_count": pass_count, "gate_passed": gate_passed}
