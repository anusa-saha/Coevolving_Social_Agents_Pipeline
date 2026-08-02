"""
Runs the stage cascade for ONE scenario: Verifier -> Weak arm -> Strong arm,
in that fixed order, up to whichever stage you asked for.

The key rule: on ANY failure at ANY stage in that prefix, the scenario goes
back to the Challenger for revision, and the cascade restarts from the
Verifier -- never resuming partway through. So:
  - target_stage="verifier"   -> just Verifier, revise-and-retry on failure.
  - target_stage="weak_arm"   -> Verifier -> Weak arm. A weak-arm failure goes
                                  back to Challenger, then Verifier again,
                                  then Weak arm again. A verifier failure also
                                  goes back to Challenger then Verifier again.
  - target_stage="strong_arm" -> Verifier -> Weak arm -> Strong arm. A
                                  strong-arm failure goes all the way back to
                                  Challenger -> Verifier -> Weak arm -> Strong
                                  arm again. Same for a weak-arm or verifier
                                  failure anywhere in the chain.

This is the single place this retry logic lives -- cli.py and pipeline.py
both just call run_cascade().
"""
import config
from challenger import revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from storage import record_iteration, record_rejected, record_accepted, save_transcript
from feedback import build_feedback

STAGE_ORDER = ["verifier", "weak_arm", "strong_arm"]

_ROLLOUT_FIELDS = {
    "weak_arm": ("passed", "settlement", "content_results", "raw_output"),
    "strong_arm": ("passed", "settled", "settlement", "revealed", "content_results",
                   "provenance_results", "transcript"),
}


def _run_stage(stage_name: str, scenario: dict):
    if stage_name == "verifier":
        result = run_verifier(scenario)
        passed = result.get("verdict") == "PASS"
    elif stage_name == "weak_arm":
        result = run_weak_arm(scenario)
        passed = result["gate_passed"]
    elif stage_name == "strong_arm":
        result = run_strong_arm(scenario)
        passed = result["gate_passed"]
    else:
        raise ValueError(f"Unknown stage: {stage_name!r}")
    return result, passed


def _rollout_summaries(stage_name: str, result: dict) -> list:
    fields = _ROLLOUT_FIELDS.get(stage_name)
    if not fields:
        return []
    return [{f: r.get(f) for f in fields} for r in result.get("rollouts", [])]


def _log_entry(scenario_id: str, round_num: int, stage_name: str, result: dict,
                passed: bool, fb: dict, scenario: dict) -> dict:
    entry = {
        "scenario_id": scenario_id,
        "round": round_num,
        "stage": stage_name,
        "passed": passed,
        "diagnosis": fb["diagnosis"],
        "evidence": fb["evidence"],
        "evidence_data": fb["evidence_data"],
        "scenario": scenario,
    }
    if stage_name == "verifier":
        entry["raw_verdict"] = result
    else:
        entry["pass_count"] = result.get("pass_count")
        entry["rollouts"] = _rollout_summaries(stage_name, result)

    record_iteration(entry)
    if passed:
        record_accepted(entry)
    else:
        entry["reject_tag"] = fb["reject_tag"]
        record_rejected(entry)
    return entry


def run_cascade(scenario: dict, target_stage: str, max_rounds: int = None) -> dict:
    """
    Returns:
        {
            "status": "accepted" | "exhausted",
            "scenario": <final scenario dict>,
            "rounds_taken": int,
            "history": [every logged entry, every stage, every round -- in order],
        }
    """
    if target_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown target_stage: {target_stage!r}")
    if max_rounds is None:
        max_rounds = config.MAX_REFINEMENT_ROUNDS

    stages_to_run = STAGE_ORDER[:STAGE_ORDER.index(target_stage) + 1]
    scenario_id = scenario.setdefault("scenario_id", scenario.get("scenario_id"))
    current = scenario
    history = []

    for round_num in range(1, max_rounds + 1):
        for stage_name in stages_to_run:
            result, passed = _run_stage(stage_name, current)

            if stage_name in ("weak_arm", "strong_arm"):
                for i, rollout in enumerate(result["rollouts"]):
                    save_transcript(scenario_id, round_num, stage_name, i, rollout)

            fb = build_feedback(stage_name, current, result)
            entry = _log_entry(scenario_id, round_num, stage_name, result, passed, fb, current)
            history.append(entry)

            if not passed:
                print(f"scenario_id={scenario_id} round {round_num}: {stage_name} "
                      f"REJECTED ({fb['reject_tag']})")
                if round_num == max_rounds:
                    return {"status": "exhausted", "scenario": current,
                            "rounds_taken": round_num, "history": history}
                current = revise_scenario(
                    current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
                )
                current["scenario_id"] = scenario_id
                break  # restart the stage cascade from the Verifier, next round
        else:
            # Every stage in stages_to_run passed without a break -> target_stage cleared.
            print(f"scenario_id={scenario_id}: PASSED {target_stage} after {round_num} round(s)")
            return {"status": "accepted", "scenario": current,
                    "rounds_taken": round_num, "history": history}

    return {"status": "exhausted", "scenario": current, "rounds_taken": max_rounds, "history": history}
