"""
Per-stage CLI. Each stage is its own subcommand, reads scenarios from a file,
and writes out two files: everything that passed, and everything that was
rejected along the way -- so you can inspect each stage in isolation and
manually chain one stage's output into the next stage's input.

Retries always restart from the top of the chain. All of that logic lives in
cascade.py's run_cascade() -- this file just wires the CLI around it:
  - `verifier` runs: Verifier only. A rejection revises and retries Verifier.
  - `weak-arm` runs: Verifier -> Weak arm. A rejection at EITHER stage revises
    and restarts from the Verifier -- a weak-arm failure never skips straight
    back to weak-arm alone.
  - `strong-arm` runs: Verifier -> Weak arm -> Strong arm. A rejection at ANY
    of the three restarts the whole chain from the Verifier again.

So `python cli.py strong-arm` fully re-earns every earlier stage's pass on
every single retry, exactly like the full end-to-end pipeline does.

Every single attempt, at every round, at every stage, is also appended
one-at-a-time into the shared output/all_iterations.json (and
output/accepted.json / rejected.json), via storage.py.

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
from challenger import generate_scenario
from cascade import run_cascade
from storage import record_iteration
import domain_loader


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
# Stage: domains (list available domain keys -- no generation, no cascade)
# ---------------------------------------------------------------------------

def cmd_domains(args):
    print("Available --scenario-type domain values:\n")
    for key, display_name in domain_loader.available_domains():
        print(f"  {key:32s} ({display_name})")
    print(
        "\nPass any of the keys above (or its display name) as --scenario-type to inject that "
        "domain's context + few-shot examples into the Challenger prompt. Any other value is still "
        "accepted as a free-text hint, same as before."
    )


# ---------------------------------------------------------------------------
# Stage: challenger (no gate -- just generation, no cascade involved)
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
# Stages: verifier / weak-arm / strong-arm -- all thin wrappers over the
# shared cascade, which is what actually restarts from the top on failure.
# ---------------------------------------------------------------------------

def _run_gated_stage(args, target_stage: str, stage_label: str):
    scenarios = load_scenarios(args.input)
    accepted, rejected = [], []

    for scenario in scenarios:
        ensure_scenario_id(scenario)
        result = run_cascade(scenario, target_stage=target_stage, max_rounds=args.max_rounds)

        # Every attempt at every stage in the chain, across every round, is in
        # result["history"] -- separate the final pass from every failure along
        # the way so both are visible in the output files.
        for entry in result["history"]:
            if entry["passed"] and entry["stage"] == target_stage:
                accepted.append(entry)
            elif not entry["passed"]:
                rejected.append(entry)

        if result["status"] == "exhausted":
            print(f"scenario_id={scenario.get('scenario_id')}: EXHAUSTED after "
                  f"{result['rounds_taken']} round(s), giving up")

    write_json(args.out_accepted, accepted)
    write_json(args.out_rejected, rejected)
    print(f"\n{stage_label} stage done: {len(accepted)} accepted, "
          f"{len(rejected)} rejected attempt(s) logged across the whole chain.")
    print(f"  Accepted -> {args.out_accepted}")
    print(f"  Rejected -> {args.out_rejected}")


def cmd_verifier(args):
    _run_gated_stage(args, target_stage="verifier", stage_label="Verifier")


def cmd_weak_arm(args):
    _run_gated_stage(args, target_stage="weak_arm", stage_label="Weak-arm")


def cmd_strong_arm(args):
    _run_gated_stage(args, target_stage="strong_arm", stage_label="Strong-arm")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run individual stages of the scenario pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("domains", help="List available --scenario-type domain values.")
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("challenger", help="Generate new candidate scenarios.")
    p.add_argument("--n", type=int, default=1, help="How many scenarios to generate.")
    p.add_argument(
        "--scenario-type", type=str, default=None,
        help="A known domain key/name (see `python cli.py domains`) to inject that domain's "
             "context + few-shot examples, or any other free-text hint (used as before).",
    )
    p.add_argument("--out", type=str, default="output/challenger_scenarios.json")
    p.set_defaults(func=cmd_challenger)

    p = sub.add_parser(
        "verifier",
        help="Run the verifier gate. A rejection revises via the Challenger and retries the verifier.",
    )
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/verifier_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/verifier_rejected.json")
    p.set_defaults(func=cmd_verifier)

    p = sub.add_parser(
        "weak-arm",
        help="Run Verifier -> Weak arm. A rejection at either stage revises via the Challenger and "
             "restarts from the Verifier -- never resumes partway through.",
    )
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/weak_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/weak_arm_rejected.json")
    p.set_defaults(func=cmd_weak_arm)

    p = sub.add_parser(
        "strong-arm",
        help="Run Verifier -> Weak arm -> Strong arm. A rejection at any of the three revises via the "
             "Challenger and restarts the whole chain from the Verifier.",
    )
    p.add_argument("--in", "--input", dest="input", type=str, required=True, help="Input scenarios JSON file.")
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--out-accepted", type=str, default="output/strong_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default="output/strong_arm_rejected.json")
    p.set_defaults(func=cmd_strong_arm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
