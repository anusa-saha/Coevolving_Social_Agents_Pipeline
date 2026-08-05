"""
Full end-to-end pipeline: generate a scenario, then run it through the full
cascade (Verifier -> Weak arm -> Strong arm), restarting from the top on
any failure at any stage. Verifier/Weak arm/Strong arm each get their own
independent failure budget (config.MAX_REFINEMENT_ROUNDS) -- see cascade.py.
This is just the entry point that generates a fresh scenario first.
"""
import config
from challenger import generate_scenario
from cascade import run_cascade
from storage import next_scenario_id


def run_pipeline(scenario_type: str = None) -> dict:
    scenario = generate_scenario(scenario_type)
    if not scenario.get("scenario_id"):
        scenario["scenario_id"] = next_scenario_id()

    result = run_cascade(scenario, target_stage="strong_arm", max_rounds=config.MAX_REFINEMENT_ROUNDS)
    return {
        "status": result["status"],
        "scenario": result["scenario"],
        "rounds_taken": result["rounds_taken"],
        "exhausted_stage": result.get("exhausted_stage"),
    }
