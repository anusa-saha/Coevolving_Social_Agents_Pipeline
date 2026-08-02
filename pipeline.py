"""
Full end-to-end pipeline: generate a scenario, then run it through the full
cascade (Verifier -> Weak arm -> Strong arm), restarting from the Verifier on
any failure at any stage. All the actual retry logic lives in cascade.py --
this is just the entry point that generates a fresh scenario first.
"""
import uuid

import config
from challenger import generate_scenario
from cascade import run_cascade


def run_pipeline(scenario_type: str = None) -> dict:
    scenario = generate_scenario(scenario_type)
    scenario.setdefault("scenario_id", f"seed_{uuid.uuid4().hex[:8]}")

    result = run_cascade(scenario, target_stage="strong_arm", max_rounds=config.MAX_REFINEMENT_ROUNDS)
    return {
        "status": result["status"],
        "scenario": result["scenario"],
        "rounds_taken": result["rounds_taken"],
    }
