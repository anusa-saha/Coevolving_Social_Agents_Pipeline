import argparse
import json
from pathlib import Path

import config
import domain_loader
from challenger import generate_scenario
from cascade import run_cascade
from storage import write_json_array, next_scenario_id


def load_scenarios(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    scenarios = []
    for item in data:
        if isinstance(item, dict) and "scenario" in item:
            scenarios.append(item["scenario"])
        else:
            scenarios.append(item)
    return scenarios


def ensure_scenario_id(scenario: dict) -> str:
    if not scenario.get("scenario_id"):
        scenario["scenario_id"] = next_scenario_id()
    return scenario["scenario_id"]


def cmd_domains(args):
    for key, display_name in domain_loader.available_domains():
        print(f"{key:32s} ({display_name})")


def cmd_challenger(args):
    results = []
    for i in range(args.n):
        scenario = generate_scenario(args.scenario_type)
        entry = {"scenario_id": scenario["scenario_id"], "stage": "challenger", "scenario": scenario}
        results.append(entry)
        print(f"[{i + 1}/{args.n}] generated scenario_id={scenario['scenario_id']}")

    write_json_array(args.out, results)
    print(f"Wrote {len(results)} scenario(s) to {args.out}")


def _run_gated_stage(args, target_stage: str):
    scenarios = load_scenarios(args.input)
    accepted, rejected = [], []

    for scenario in scenarios:
        ensure_scenario_id(scenario)
        result = run_cascade(scenario, target_stage=target_stage, max_rounds=args.max_rounds)

        for entry in result["history"]:
            if entry["passed"] and entry["stage"] == target_stage:
                accepted.append(entry)
            elif not entry["passed"]:
                rejected.append(entry)

        if result["status"] == "exhausted":
            print(f"scenario_id={scenario.get('scenario_id')} EXHAUSTED at "
                  f"{result['exhausted_stage']} after {result['rounds_taken']} round(s)")

    write_json_array(args.out_accepted, accepted)
    write_json_array(args.out_rejected, rejected)
    print(f"{len(accepted)} accepted, {len(rejected)} rejected attempts logged")
    print(f"Accepted -> {args.out_accepted}")
    print(f"Rejected -> {args.out_rejected}")


def cmd_verifier(args):
    _run_gated_stage(args, target_stage="verifier")


def cmd_weak_arm(args):
    _run_gated_stage(args, target_stage="weak_arm")


def cmd_strong_arm(args):
    _run_gated_stage(args, target_stage="strong_arm")


def main():
    parser = argparse.ArgumentParser(description="Run individual stages of the scenario pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("domains")
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("challenger")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--scenario-type", type=str, default=None)
    p.add_argument("--out", type=str, default="output/challenger_scenarios.json")
    p.set_defaults(func=cmd_challenger)

    p = sub.add_parser("verifier")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/verifier_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/verifier_rejected.json")
    p.set_defaults(func=cmd_verifier)

    p = sub.add_parser("weak-arm")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/weak_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/weak_arm_rejected.json")
    p.set_defaults(func=cmd_weak_arm)

    p = sub.add_parser("strong-arm")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/strong_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/strong_arm_rejected.json")
    p.set_defaults(func=cmd_strong_arm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
