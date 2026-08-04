import argparse

import domain_loader
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the scenario-generation pipeline.")
    parser.add_argument("--n", type=int, default=1, help="Number of accepted scenarios to attempt.")
    parser.add_argument(
        "--scenario-type", type=str, default=None,
        help="A known domain key/name (see --list-domains) to inject that domain's context + "
             "few-shot examples, or any other free-text hint (used as before).",
    )
    parser.add_argument(
        "--list-domains", action="store_true",
        help="Print available --scenario-type domain values and exit.",
    )
    args = parser.parse_args()

    if args.list_domains:
        for key, display_name in domain_loader.available_domains():
            print(f"  {key:32s} ({display_name})")
        return

    for i in range(args.n):
        result = run_pipeline(args.scenario_type)
        print(
            f"[{i + 1}/{args.n}] status={result['status']} "
            f"rounds_taken={result['rounds_taken']} "
            f"scenario_id={result['scenario'].get('scenario_id')}"
        )


if __name__ == "__main__":
    main()
