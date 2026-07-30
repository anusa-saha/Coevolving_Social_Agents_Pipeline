import argparse

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the scenario-generation pipeline.")
    parser.add_argument("--n", type=int, default=1, help="Number of accepted scenarios to attempt.")
    parser.add_argument("--scenario-type", type=str, default=None, help="Optional preferred scenario_type hint.")
    args = parser.parse_args()

    for i in range(args.n):
        result = run_pipeline(args.scenario_type)
        print(
            f"[{i + 1}/{args.n}] status={result['status']} "
            f"rounds_taken={result['rounds_taken']} "
            f"scenario_id={result['scenario'].get('scenario_id')}"
        )


if __name__ == "__main__":
    main()
