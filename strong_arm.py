"""
Strong arm: the full multi-agent group conversation. Every agent, including the
decision-maker, is played by the same model -- gpt-5.4 -- called through the
normal OpenAI client with no custom endpoint. One model, called repeatedly,
wearing a different agent's prompt each turn.
"""
import config
from llm_clients import gpt_chat, extract_json
from prompt_builder import build_turn_prompt, _dm_list
from grader import grade_content, grade_provenance, build_revealed_set


def _settle_allowed_now(scenario: dict, speaker: str, transcript: list, dm_list: list) -> bool:
    """Settle is offered to a decision-maker once every non-decision-maker agent has spoken."""
    if speaker not in dm_list:
        return False
    non_dm_agents = {a["agent_id"] for a in scenario["agents"] if a["agent_id"] not in dm_list}
    spoken = {e["speaker"] for e in transcript if e["speaker"] not in dm_list}
    return non_dm_agents.issubset(spoken)


def run_strong_arm_rollout(scenario: dict) -> dict:
    transcript = []
    turn_order = scenario["interaction_config"]["turn_order"]
    turn_cap = scenario["interaction_config"]["turn_cap"]
    dm_list = _dm_list(scenario)
    settlement = None
    settled = False

    for t in range(turn_cap):
        speaker = turn_order[t % len(turn_order)]
        settle_allowed = _settle_allowed_now(scenario, speaker, transcript, dm_list)

        messages = build_turn_prompt(scenario, speaker, transcript, settle_allowed)
        raw = gpt_chat(
            model=config.STRONG_ARM_MODEL,
            messages=messages,
            temperature=config.STRONG_ARM_TEMPERATURE,
            max_tokens=1500,
        )

        try:
            action = extract_json(raw)
        except Exception as e:
            action = {"type": "say", "text": f"[unparsable model output: {e}]"}

        action["speaker"] = speaker
        action["turn"] = t
        action["raw_output"] = raw
        transcript.append(action)

        if action.get("type") == "settle":
            settlement = action.get("settlement")
            settled = True
            break

    if not settled:
        return {
            "arm": "strong",
            "transcript": transcript,
            "settlement": None,
            "settled": False,
            "content_results": {},
            "provenance_results": {},
            "revealed": build_revealed_set(transcript),
            "passed": False,  # turn cap hit with no settlement -> automatic fail
        }

    content_results = grade_content(scenario, settlement)
    provenance_results = grade_provenance(scenario, settlement, transcript)
    passed = (
        bool(content_results) and all(content_results.values()) and
        bool(provenance_results) and all(provenance_results.values())
    ) if scenario.get("provenance_checks") else (bool(content_results) and all(content_results.values()))

    return {
        "arm": "strong",
        "transcript": transcript,
        "settlement": settlement,
        "settled": True,
        "content_results": content_results,
        "provenance_results": provenance_results,
        "revealed": build_revealed_set(transcript),
        "passed": passed,
    }


def run_strong_arm(scenario: dict) -> dict:
    rollouts = [run_strong_arm_rollout(scenario) for _ in range(config.STRONG_ARM_ROLLOUTS)]
    pass_count = sum(r["passed"] for r in rollouts)
    gate_passed = pass_count >= config.STRONG_ARM_MIN_PASS
    return {"rollouts": rollouts, "pass_count": pass_count, "gate_passed": gate_passed}
