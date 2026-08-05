import config
from challenger import revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from feedback import build_feedback
from storage import next_scenario_id, save_stage_result, save_scenario_outcome

STAGE_ORDER = ["verifier", "weak_arm", "strong_arm"]


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


def run_cascade(scenario: dict, target_stage: str, max_rounds: int = None) -> dict:
    if target_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown target_stage: {target_stage!r}")
    if max_rounds is None:
        max_rounds = config.MAX_REFINEMENT_ROUNDS

    stages_to_run = STAGE_ORDER[:STAGE_ORDER.index(target_stage) + 1]

    scenario_id = scenario.get("scenario_id")
    if not scenario_id:
        scenario_id = next_scenario_id()
        scenario["scenario_id"] = scenario_id

    current = scenario
    history = []
    stage_failure_counts = {stage: 0 for stage in STAGE_ORDER}
    round_num = 0

    while True:
        round_num += 1
        for stage_name in stages_to_run:
            result, passed = _run_stage(stage_name, current)
            fb = build_feedback(stage_name, current, result)
            save_stage_result(scenario_id, round_num, stage_name, current, result, fb, passed)

            history.append({
                "round": round_num,
                "stage": stage_name,
                "passed": passed,
                "reject_tag": fb.get("reject_tag"),
                "scenario": current,
            })

            print(f"scenario_id={scenario_id} round={round_num} stage={stage_name} passed={passed}")

            if not passed:
                stage_failure_counts[stage_name] += 1
                if stage_failure_counts[stage_name] >= max_rounds:
                    save_scenario_outcome(
                        scenario_id, "exhausted", current, round_num, stage_name,
                        dict(stage_failure_counts),
                    )
                    return {
                        "status": "exhausted",
                        "scenario": current,
                        "rounds_taken": round_num,
                        "exhausted_stage": stage_name,
                        "stage_failure_counts": dict(stage_failure_counts),
                        "history": history,
                    }

                current = revise_scenario(
                    current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
                )
                current["scenario_id"] = scenario_id
                break
        else:
            save_scenario_outcome(scenario_id, "accepted", current, round_num, None, dict(stage_failure_counts))
            return {
                "status": "accepted",
                "scenario": current,
                "rounds_taken": round_num,
                "exhausted_stage": None,
                "stage_failure_counts": dict(stage_failure_counts),
                "history": history,
            }
