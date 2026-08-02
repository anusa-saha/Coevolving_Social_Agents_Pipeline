"""
Per-stage CLI. Each stage is its own subcommand, reads scenarios from a file,
runs that stage (with its own revise-and-retry feedback loop against the
Challenger), and writes out two files: everything that passed, and everything
that was rejected along the way -- so you can inspect each stage in isolation
and manually chain one stage's output into the next stage's input.

Every single attempt, at every round, is also appended one-at-a-time into the
shared output/all_iterations.json (and output/accepted.json / rejected.json),
via storage.py -- exactly like the full pipeline does.

Usage:
    python cli.py challenger --n 5 --out output/challenger_scenarios.json
    python cli.py verifier --in output/challenger_scenarios.json \
        --out-accepted output/verifier_accepted.json \
        --out-rejected output/verifier_rejected.json
    python cli.py weak-arm --in output/verifier_accepted.json \
        --out-accepted output/weak_arm_accepted.json \
        --out-rejected output/weak_arm_rejected.json
    python cli.py strong-arm --in output/weak_arm_accepted.json \
        --out-accepted output/strong_arm_accepted.json \
        --out-rejected output/strong_arm_rejected.json
"""
import argparse
import json
import uuid
from pathlib import Path

import config
from challenger import generate_scenario, revise_scenario
from verifier import run_verifier
from weak_arm import run_weak_arm
from strong_arm import run_strong_arm
from storage import record_iteration, record_rejected, record_accepted, save_transcript
from feedback import build_feedback


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_scenarios(path: str) -> list:
    """
    Reads a JSON array from disk. Each item can either be a raw scenario dict
    (e.g. straight from `challenger`) or a wrapped record with a "scenario" key
    (e.g. the accepted/rejected output of `verifier`, `weak-arm`, `strong-arm`)
    -- either shape is unwrapped into a plain list of scenario dicts.
    """
    with open(path) as f:
        data = json.load(f)
    scenarios = []
    for item in data:
        if isinstance(item, dict) and "scenario" in item:
            scenarios.append(item["scenario"])
        else:
            scenarios.append(item)
    return scenarios


def write_json(path: str, data: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_scenario_id(scenario: dict) -> str:
    return scenario.setdefault("scenario_id", f"seed_{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Stage: challenger
# ---------------------------------------------------------------------------

def cmd_challenger(args):
    results = []
    for i in range(args.n):
        scenario = generate_scenario(args.scenario_type)
        scenario_id = ensure_scenario_id(scenario)

        entry = {"scenario_id": scenario_id, "stage": "challenger", "scenario": scenario}
        record_iteration(entry)
        results.append(entry)
        print(f"[{i + 1}/{args.n}] generated scenario_id={scenario_id}")

    write_json(args.out, results)
    print(f"\nWrote {len(results)} scenario(s) to {args.out}")


# ---------------------------------------------------------------------------
# Stage: verifier
# ---------------------------------------------------------------------------

def cmd_verifier(args):
    scenarios = load_scenarios(args.input)
    accepted, rejected = [], []

    for scenario in scenarios:
        scenario_id = ensure_scenario_id(scenario)
        current = scenario

        for round_num in range(1, args.max_rounds + 1):
            verdict = run_verifier(current)
            passed = verdict.get("verdict") == "PASS"
            fb = build_feedback("verifier", current, verdict)

            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "verifier",
                "passed": passed,
                "diagnosis": fb["diagnosis"],
                "evidence": fb["evidence"],
                "evidence_data": fb["evidence_data"],
                "raw_verdict": verdict,
                "scenario": current,
            }
            record_iteration(entry)

            if passed:
                record_accepted(entry)
                accepted.append(entry)
                print(f"scenario_id={scenario_id}: ACCEPTED by verifier after {round_num} round(s)")
                break

            entry["reject_tag"] = fb["reject_tag"]
            record_rejected(entry)
            rejected.append(entry)
            print(f"scenario_id={scenario_id} round {round_num}: REJECTED ({fb['reject_tag']})")

            if round_num == args.max_rounds:
                print(f"scenario_id={scenario_id}: EXHAUSTED after {args.max_rounds} round(s), giving up")
                break

            current = revise_scenario(
                current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            current["scenario_id"] = scenario_id

    write_json(args.out_accepted, accepted)
    write_json(args.out_rejected, rejected)
    print(f"\nVerifier stage done: {len(accepted)} accepted, {len(rejected)} rejected attempt(s) logged.")
    print(f"  Accepted -> {args.out_accepted}")
    print(f"  Rejected -> {args.out_rejected}")


# ---------------------------------------------------------------------------
# Stage: weak arm
# ---------------------------------------------------------------------------

def cmd_weak_arm(args):
    scenarios = load_scenarios(args.input)
    accepted, rejected = [], []

    for scenario in scenarios:
        scenario_id = ensure_scenario_id(scenario)
        current = scenario

        for round_num in range(1, args.max_rounds + 1):
            result = run_weak_arm(current)
            for i, rollout in enumerate(result["rollouts"]):
                save_transcript(scenario_id, round_num, "weak_arm", i, rollout)

            fb = build_feedback("weak_arm", current, result)
            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "weak_arm",
                "passed": result["gate_passed"],
                "pass_count": result["pass_count"],
                "diagnosis": fb["diagnosis"],
                "evidence": fb["evidence"],
                "evidence_data": fb["evidence_data"],
                "rollouts": [
                    {
                        "passed": r.get("passed"),
                        "settlement": r.get("settlement"),
                        "content_results": r.get("content_results"),
                        "raw_output": r.get("raw_output"),
                    }
                    for r in result["rollouts"]
                ],
                "scenario": current,
            }
            record_iteration(entry)

            if result["gate_passed"]:
                record_accepted(entry)
                accepted.append(entry)
                print(
                    f"scenario_id={scenario_id}: PASSED weak-arm gate after {round_num} round(s) "
                    f"({result['pass_count']}/{config.WEAK_ARM_ROLLOUTS} lone rollouts passed)"
                )
                break

            entry["reject_tag"] = fb["reject_tag"]
            record_rejected(entry)
            rejected.append(entry)
            print(f"scenario_id={scenario_id} round {round_num}: REJECTED ({fb['reject_tag']}, "
                  f"{result['pass_count']}/{config.WEAK_ARM_ROLLOUTS} passed)")

            if round_num == args.max_rounds:
                print(f"scenario_id={scenario_id}: EXHAUSTED after {args.max_rounds} round(s), giving up")
                break

            current = revise_scenario(
                current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            current["scenario_id"] = scenario_id

    write_json(args.out_accepted, accepted)
    write_json(args.out_rejected, rejected)
    print(f"\nWeak-arm stage done: {len(accepted)} accepted, {len(rejected)} rejected attempt(s) logged.")
    print(f"  Accepted -> {args.out_accepted}")
    print(f"  Rejected -> {args.out_rejected}")


# ---------------------------------------------------------------------------
# Stage: strong arm
# ---------------------------------------------------------------------------

def cmd_strong_arm(args):
    scenarios = load_scenarios(args.input)
    accepted, rejected = [], []

    for scenario in scenarios:
        scenario_id = ensure_scenario_id(scenario)
        current = scenario

        for round_num in range(1, args.max_rounds + 1):
            result = run_strong_arm(current)
            for i, rollout in enumerate(result["rollouts"]):
                save_transcript(scenario_id, round_num, "strong_arm", i, rollout)

            fb = build_feedback("strong_arm", current, result)
            entry = {
                "scenario_id": scenario_id,
                "round": round_num,
                "stage": "strong_arm",
                "passed": result["gate_passed"],
                "pass_count": result["pass_count"],
                "diagnosis": fb["diagnosis"],
                "evidence": fb["evidence"],
                "evidence_data": fb["evidence_data"],
                "rollouts": [
                    {
                        "passed": r.get("passed"),
                        "settled": r.get("settled"),
                        "settlement": r.get("settlement"),
                        "revealed": r.get("revealed"),
                        "content_results": r.get("content_results"),
                        "provenance_results": r.get("provenance_results"),
                        "transcript": r.get("transcript"),
                    }
                    for r in result["rollouts"]
                ],
                "scenario": current,
            }
            record_iteration(entry)

            if result["gate_passed"]:
                record_accepted(entry)
                accepted.append(entry)
                print(
                    f"scenario_id={scenario_id}: ACCEPTED into dataset after {round_num} round(s) "
                    f"({result['pass_count']}/{config.STRONG_ARM_ROLLOUTS} group rollouts passed)"
                )
                break

            entry["reject_tag"] = fb["reject_tag"]
            record_rejected(entry)
            rejected.append(entry)
            print(f"scenario_id={scenario_id} round {round_num}: REJECTED ({fb['reject_tag']}, "
                  f"{result['pass_count']}/{config.STRONG_ARM_ROLLOUTS} passed)")

            if round_num == args.max_rounds:
                print(f"scenario_id={scenario_id}: EXHAUSTED after {args.max_rounds} round(s), giving up")
                break

            current = revise_scenario(
                current, fb["reject_tag"], fb["diagnosis"], fb["evidence"], fb["fix_instructions"],
            )
            current["scenario_id"] = scenario_id

    write_json(args.out_accepted, accepted)
    write_json(args.out_rejected, rejected)
    print(f"\nStrong-arm stage done: {len(accepted)} accepted, {len(rejected)} rejected attempt(s) logged.")
    print(f"  Accepted -> {args.out_accepted}")
    print(f"  Rejected -> {args.out_rejected}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run individual stages of the scenario pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("challenger", help="Generate new candidate scenarios.")
    p.add_argument("--n", type=int, default=1, help="How many scenarios to generate.")
    p.add_argument("--scenario-type", type=str, default=None, help="Optional preferred scenario_type hint.")
    p.add_argument("--out", type=str, default="output/challenger_scenarios.json")
    p.set_defaults(func=cmd_challenger)

    p = sub.add_parser("verifier", help="Run the verifier gate, with revise-and-retry feedback to the Challenger.")
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/verifier_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/verifier_rejected.json")
    p.set_defaults(func=cmd_verifier)

    p = sub.add_parser("weak-arm", help="Run the weak-arm gate, with revise-and-retry feedback to the Challenger.")
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/weak_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/weak_arm_rejected.json")
    p.set_defaults(func=cmd_weak_arm)

    p = sub.add_parser("strong-arm", help="Run the strong-arm gate, with revise-and-retry feedback to the Challenger.")
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/strong_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/strong_arm_rejected.json")
    p.set_defaults(func=cmd_strong_arm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
