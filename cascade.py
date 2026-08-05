"""
Runs the stage cascade for ONE scenario: Verifier -> Weak arm -> Strong arm,
in that fixed order, up to whichever stage you asked for.

The restart rule is unchanged: on ANY failure at ANY stage in that prefix, the scenario goes
back to the Challenger for revision, and the cascade restarts from the top of the stage list --
never resuming partway through.

What's different: `max_rounds` is now an INDEPENDENT retry budget PER STAGE, not one shared
counter for the whole chain. Verifier, Weak arm, and Strong arm each get up to `max_rounds`
failed attempts of their own before the scenario is abandoned as "exhausted":
  - If Verifier itself fails 10 times in a row, that exhausts Verifier's own budget immediately
    -- Weak arm and Strong arm never even got a chance to run yet.
  - If Verifier keeps passing quickly but Weak arm fails 10 times (each failure sent back
    through the Challenger and a fresh Verifier pass), it's WEAK ARM's budget that runs out --
    Verifier's own failure count stays wherever it happens to be, since it's tracked separately.
  - Same independence for Strong arm.

So a single scenario can end up going through more than `max_rounds` total passes through the
stage list -- what's capped is how many times any ONE stage is allowed to fail, not the total
number of trips through the cascade.

This is the single place this retry logic lives -- cli.py and pipeline.py both just call
run_cascade().
"""
import config
from challenger import revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from storage import record_iteration, record_rejected, record_accepted, save_transcript, next_scenario_id
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
    max_rounds: the per-stage failure budget (default: config.MAX_REFINEMENT_ROUNDS). Each of
    Verifier / Weak arm / Strong arm is allowed up to this many FAILURES of its own before the
    scenario is abandoned -- independently of how the other stages are doing.

    Returns:
        {
            "status": "accepted" | "exhausted",
            "scenario": <final scenario dict>,
            "rounds_taken": int,                  # total passes through the stage list
            "exhausted_stage": str | None,         # which stage's budget ran out, if exhausted
            "stage_failure_counts": {stage: int},  # final failure tally per stage
            "history": [every logged entry, every stage, every round -- in order],
        }
    """
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

            if stage_name in ("weak_arm", "strong_arm"):
                for i, rollout in enumerate(result["rollouts"]):
                    save_transcript(scenario_id, round_num, stage_name, i, rollout)

            fb = build_feedback(stage_name, current, result)
            entry = _log_entry(scenario_id, round_num, stage_name, result, passed, fb, current)
            history.append(entry)

            if not passed:
                stage_failure_counts[stage_name] += 1
                print(
                    f"scenario_id={scenario_id} round {round_num}: {stage_name} "
                    f"REJECTED ({fb['reject_tag']}) "
                    f"[{stage_name} failure {stage_failure_counts[stage_name]}/{max_rounds}]"
                )

                if stage_failure_counts[stage_name] >= max_rounds:
                    print(
                        f"scenario_id={scenario_id}: EXHAUSTED -- {stage_name} hit its own "
                        f"{max_rounds}-failure limit after {round_num} total round(s)"
                    )
                    return {
                        "status": "exhausted", "scenario": current, "rounds_taken": round_num,
                        "exhausted_stage": stage_name,
                        "stage_failure_counts": dict(stage_failure_counts), "history": history,
                    }

                current = revise_scenario(
                    current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
                )
                current["scenario_id"] = scenario_id
                break  # restart the stage cascade from the top, next round
        else:
            # Every stage in stages_to_run passed without a break -> target_stage cleared.
            print(f"scenario_id={scenario_id}: PASSED {target_stage} after {round_num} round(s)")
            return {
                "status": "accepted", "scenario": current, "rounds_taken": round_num,
                "exhausted_stage": None,
                "stage_failure_counts": dict(stage_failure_counts), "history": history,
            }
