import argparse
import json
import re
from pathlib import Path

import config
import domain_loader
from challenger import generate_scenario, VALID_AGENT_COUNTS
from cascade import run_cascade
from storage import write_json_array, next_scenario_id, set_scenario_counter, current_scenario_counter


def _apply_output_dir(args):
    """Point config.OUTPUT_DIR (used by storage.py for the scenarios/ tree, the scenario
    counter, and all the .jsonl logs) at whatever --output-dir was given, before any
    storage function runs. Assigning through the module object -- not a bound import --
    means every other module that did `import config` and reads config.OUTPUT_DIR sees
    the override too, with no changes needed anywhere else."""
    if getattr(args, "output_dir", None):
        config.OUTPUT_DIR = args.output_dir
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _default_out_path(args, filename: str) -> str:
    """Resolve a per-command output file's default path relative to --output-dir if the
    user didn't explicitly override that specific file with its own flag."""
    base = args.output_dir if getattr(args, "output_dir", None) else "output"
    return str(Path(base) / filename)


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


def _scenario_number(scenario_id):
    match = re.match(r"^scenario_(\d+)$", str(scenario_id or ""))
    return int(match.group(1)) if match else None


def filter_by_scenario_range(scenarios: list, start_from, end_at) -> list:
    if start_from is None and end_at is None:
        return scenarios

    kept, skipped_unnumbered = [], 0
    for scenario in scenarios:
        n = _scenario_number(scenario.get("scenario_id"))
        if n is None:
            skipped_unnumbered += 1
            continue
        if start_from is not None and n < start_from:
            continue
        if end_at is not None and n > end_at:
            continue
        kept.append(scenario)

    if skipped_unnumbered:
        print(f"Note: {skipped_unnumbered} scenario(s) had a non-standard scenario_id and were "
              f"skipped because --start-from/--end-at is active.")
    return kept


def ensure_scenario_id(scenario: dict) -> str:
    if not scenario.get("scenario_id"):
        scenario["scenario_id"] = next_scenario_id()
    return scenario["scenario_id"]


def cmd_domains(args):
    for key, display_name in domain_loader.available_domains():
        print(f"{key:32s} ({display_name})")


def cmd_reset_counter(args):
    _apply_output_dir(args)
    before = current_scenario_counter()
    if args.set is None:
        print(f"Current scenario counter: {before} (next scenario would be scenario_{before + 1})")
        return
    set_scenario_counter(args.set)
    print(f"Scenario counter changed: {before} -> {args.set}")
    print(f"Next generated scenario will be scenario_{args.set + 1}")


def cmd_challenger(args):
    _apply_output_dir(args)
    out_path = args.out if args.out is not None else _default_out_path(args, "challenger_scenarios.json")

    results = []
    for i in range(args.n):
        scenario = generate_scenario(args.scenario_type, args.num_agents)
        entry = {"scenario_id": scenario["scenario_id"], "stage": "challenger", "scenario": scenario}
        results.append(entry)
        print(f"[{i + 1}/{args.n}] generated scenario_id={scenario['scenario_id']}")

    write_json_array(out_path, results)
    print(f"Wrote {len(results)} scenario(s) to {out_path}")


def _run_gated_stage(args, target_stage: str):
    _apply_output_dir(args)
    out_accepted = args.out_accepted if args.out_accepted is not None else _default_out_path(args, f"{target_stage}_accepted.json")
    out_rejected = args.out_rejected if args.out_rejected is not None else _default_out_path(args, f"{target_stage}_rejected.json")

    scenarios = load_scenarios(args.input)
    total_loaded = len(scenarios)
    scenarios = filter_by_scenario_range(scenarios, args.start_from, args.end_at)

    if args.start_from is not None or args.end_at is not None:
        print(f"Processing {len(scenarios)} of {total_loaded} loaded scenario(s) "
              f"(start_from={args.start_from}, end_at={args.end_at})")

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

    write_json_array(out_accepted, accepted)
    write_json_array(out_rejected, rejected)
    print(f"{len(accepted)} accepted, {len(rejected)} rejected attempts logged")
    print(f"Accepted -> {out_accepted}")
    print(f"Rejected -> {out_rejected}")


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

    p = sub.add_parser("reset-counter")
    p.add_argument("--set", type=int, default=None,
                    help="Set the scenario counter to this value. Next scenario generated will be "
                         "scenario_<value+1>. Omit to just print the current value.")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory the scenario counter (and all other pipeline storage) lives in. "
                         "Overrides config.OUTPUT_DIR for this run. Defaults to config.py's built-in path.")
    p.set_defaults(func=cmd_reset_counter)

    p = sub.add_parser("challenger")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--scenario-type", type=str, default=None)
    p.add_argument("--num-agents", type=int, default=None, choices=VALID_AGENT_COUNTS)
    p.add_argument("--out", type=str, default=None,
                    help="Defaults to <output-dir>/challenger_scenarios.json")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory all pipeline storage (scenarios/ tree, counter, .jsonl logs, and "
                         "this command's default --out) lives in. Overrides config.OUTPUT_DIR.")
    p.set_defaults(func=cmd_challenger)

    p = sub.add_parser("verifier")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--start-from", type=int, default=None)
    p.add_argument("--end-at", type=int, default=None)
    p.add_argument("--out-accepted", type=str, default=None,
                    help="Defaults to <output-dir>/verifier_accepted.json")
    p.add_argument("--out-rejected", type=str, default=None,
                    help="Defaults to <output-dir>/verifier_rejected.json")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory all pipeline storage lives in. Overrides config.OUTPUT_DIR.")
    p.set_defaults(func=cmd_verifier)

    p = sub.add_parser("weak-arm")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--start-from", type=int, default=None)
    p.add_argument("--end-at", type=int, default=None)
    p.add_argument("--out-accepted", type=str, default=None,
                    help="Defaults to <output-dir>/weak_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default=None,
                    help="Defaults to <output-dir>/weak_arm_rejected.json")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory all pipeline storage lives in. Overrides config.OUTPUT_DIR.")
    p.set_defaults(func=cmd_weak_arm)

    p = sub.add_parser("strong-arm")
    p.add_argument("--in", "--input", dest="input", type=str, required=True)
    p.add_argument("--max-rounds", type=int, default=config.MAX_REFINEMENT_ROUNDS)
    p.add_argument("--start-from", type=int, default=None)
    p.add_argument("--end-at", type=int, default=None)
    p.add_argument("--out-accepted", type=str, default=None,
                    help="Defaults to <output-dir>/strong_arm_accepted.json")
    p.add_argument("--out-rejected", type=str, default=None,
                    help="Defaults to <output-dir>/strong_arm_rejected.json")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory all pipeline storage lives in. Overrides config.OUTPUT_DIR.")
    p.set_defaults(func=cmd_strong_arm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
