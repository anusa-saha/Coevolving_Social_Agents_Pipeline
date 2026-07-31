"""
The outer loop: Challenger -> Verifier -> Weak arm -> Strong arm -> accept, or
reject + diagnosis + feedback back to the Challenger. Cheap gates run before
expensive ones, exactly as in the source pipeline.
"""
import uuid

import config
from challenger import generate_scenario, revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from storage import record_iteration, record_rejected, record_accepted, save_transcript


def _save_rollouts(scenario_id: str, round_num: int, arm: str, result: dict):
    for i, rollout in enumerate(result["rollouts"]):
        save_transcript(scenario_id, round_num, arm, i, rollout)


def run_pipeline(scenario_type: str = None) -> dict:
    scenario = generate_scenario(scenario_type)
    scenario.setdefault("scenario_id", f"seed_{uuid.uuid4().hex[:8]}")
    scenario_id = scenario["scenario_id"]

    for round_num in range(1, config.MAX_REFINEMENT_ROUNDS + 1):

        # ---------------- Gate 1: Verifier (cheapest) ----------------
        verdict = run_verifier(scenario)
        if verdict.get("verdict") != "PASS":
            tag = verdict.get("tag", "MALFORMED")
            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "verifier",
                "reject_tag": tag,
                "diagnosis": verdict.get("diagnosis"),
                "evidence": verdict.get("evidence"),
                "scenario": scenario,
            }
            record_iteration(entry)
            record_rejected(entry)

            scenario = revise_scenario(
                scenario, tag,
                verdict.get("diagnosis", ""),
                verdict.get("evidence", ""),
                verdict.get("fix_instructions", ""),
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Gate 2: Weak arm ----------------
        weak_result = run_weak_arm(scenario)
        _save_rollouts(scenario_id, round_num, "weak_arm", weak_result)

        if not weak_result["gate_passed"]:
            diagnosis = (
                f"The lone decision-maker passed {weak_result['pass_count']}/"
                f"{config.WEAK_ARM_ROLLOUTS} runs (at most {config.WEAK_ARM_MAX_PASS} allowed). "
                f"The hidden information was not actually necessary to reach a passing decision."
            )
            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "weak_arm",
                "reject_tag": "LEAKED",
                "diagnosis": diagnosis,
                "pass_count": weak_result["pass_count"],
                "scenario": scenario,
            }
            record_iteration(entry)
            record_rejected(entry)

            scenario = revise_scenario(
                scenario, "LEAKED", diagnosis,
                evidence=f"{weak_result['pass_count']} of {config.WEAK_ARM_ROLLOUTS} lone rollouts "
                         f"passed all content checks.",
                fix_instructions=(
                    "Tighten checks back to exactly what the private facts imply. Remove any threshold "
                    "loose enough for generic fairness, seniority, or stereotype-based reasoning to "
                    "satisfy it without the hidden facts."
                ),
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Gate 3: Strong arm (most expensive) ----------------
        strong_result = run_strong_arm(scenario)
        _save_rollouts(scenario_id, round_num, "strong_arm", strong_result)

        if not strong_result["gate_passed"]:
            diagnosis = (
                f"The group only passed {strong_result['pass_count']}/{config.STRONG_ARM_ROLLOUTS} runs "
                f"(at least {config.STRONG_ARM_MIN_PASS} needed). The information gap is real, but the "
                f"group could not reliably surface and use the hidden facts."
            )
            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "strong_arm",
                "reject_tag": "UNCOORDINATED",
                "diagnosis": diagnosis,
                "pass_count": strong_result["pass_count"],
                "scenario": scenario,
            }
            record_iteration(entry)
            record_rejected(entry)

            scenario = revise_scenario(
                scenario, "UNCOORDINATED", diagnosis,
                evidence=f"{strong_result['pass_count']} of {config.STRONG_ARM_ROLLOUTS} group rollouts passed.",
                fix_instructions=(
                    "Add or strengthen a shared consultation norm in shared_context so every fact-holder "
                    "gets a natural, unmistakable opening to speak before the decision is finalized. "
                    "Fix the shared environment only -- never change an agent's personality to make them "
                    "more talkative or assertive."
                ),
            )
            scenario["scenario_id"] = scenario_id
            continue

        # ---------------- Accepted ----------------
        entry = {
            "scenario_id": scenario_id,
            "round": round_num,
            "stage": "accepted",
            "weak_arm_pass_count": weak_result["pass_count"],
            "strong_arm_pass_count": strong_result["pass_count"],
            "scenario": scenario,
        }
        record_iteration(entry)
        record_accepted(entry)
        return {"status": "accepted", "scenario": scenario, "rounds_taken": round_num}

    return {"status": "exhausted", "scenario": scenario, "rounds_taken": config.MAX_REFINEMENT_ROUNDS}
