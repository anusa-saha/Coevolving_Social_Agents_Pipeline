"""
The outer loop: Challenger -> Verifier -> Weak arm -> Strong arm -> accept, or
reject + diagnosis + feedback back to the Challenger. Cheap gates run before
expensive ones. Feedback for every stage is built by the single, generalized
feedback.build_feedback() function -- see feedback.py.
"""
import uuid

import config
from challenger import generate_scenario, revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from storage import record_iteration, record_rejected, record_accepted, save_transcript
from feedback import build_feedback


def _save_rollouts(scenario_id: str, round_num: int, arm: str, result: dict):
    for i, rollout in enumerate(result["rollouts"]):
        save_transcript(scenario_id, round_num, arm, i, rollout)


def _rollout_summaries(stage_name: str, stage_result: dict) -> list:
    """Extracts exactly what each rollout generated, so it's visible in the saved entry."""
    fields_by_stage = {
        "weak_arm": ("passed", "settlement", "content_results", "raw_output"),
        "strong_arm": ("passed", "settled", "settlement", "revealed", "content_results",
                       "provenance_results", "transcript"),
    }
    fields = fields_by_stage.get(stage_name)
    if not fields:
        return []
    return [{f: r.get(f) for f in fields} for r in stage_result.get("rollouts", [])]


def _try_stage(scenario_id, round_num, stage_name, stage_result, passed, scenario):
    """
    Builds the generalized feedback, logs the attempt (iteration + accepted/
    rejected) -- including the actual JSON that stage generated, not just a
    text summary -- and returns the feedback dict so the caller can decide
    whether to revise and retry.
    """
    fb = build_feedback(stage_name, scenario, stage_result)
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
        entry["raw_verdict"] = stage_result
    else:
        entry["pass_count"] = stage_result.get("pass_count")
        entry["rollouts"] = _rollout_summaries(stage_name, stage_result)

    record_iteration(entry)
    if passed:
        record_accepted(entry)
    else:
        entry["reject_tag"] = fb["reject_tag"]
        record_rejected(entry)
    return fb


def run_pipeline(scenario_type: str = None) -> dict:
    scenario = generate_scenario(scenario_type)
    scenario.setdefault("scenario_id", f"seed_{uuid.uuid4().hex[:8]}")
    scenario_id = scenario["scenario_id"]

    for round_num in range(1, config.MAX_REFINEMENT_ROUNDS + 1):

        # ---------------- Gate 1: Verifier (cheapest) ----------------
        verdict = run_verifier(scenario)
        passed = verdict.get("verdict") == "PASS"
        fb = _try_stage(scenario_id, round_num, "verifier", verdict, passed, scenario)

        if not passed:
            scenario = revise_scenario(
                scenario, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Gate 2: Weak arm ----------------
        weak_result = run_weak_arm(scenario)
        _save_rollouts(scenario_id, round_num, "weak_arm", weak_result)
        fb = _try_stage(scenario_id, round_num, "weak_arm", weak_result, weak_result["gate_passed"], scenario)

        if not weak_result["gate_passed"]:
            scenario = revise_scenario(
                scenario, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Gate 3: Strong arm (most expensive) ----------------
        strong_result = run_strong_arm(scenario)
        _save_rollouts(scenario_id, round_num, "strong_arm", strong_result)
        fb = _try_stage(scenario_id, round_num, "strong_arm", strong_result, strong_result["gate_passed"], scenario)

        if not strong_result["gate_passed"]:
            scenario = revise_scenario(
                scenario, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Accepted ----------------
        return {"status": "accepted", "scenario": scenario, "rounds_taken": round_num}

    return {"status": "exhausted", "scenario": scenario, "rounds_taken": config.MAX_REFINEMENT_ROUNDS}
