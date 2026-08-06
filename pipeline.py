import config
from challenger import generate_scenario
from cascade import run_cascade


def run_pipeline(scenario_type: str = None, num_agents: int = None) -> dict:
    scenario = generate_scenario(scenario_type, num_agents)
    result = run_cascade(scenario, target_stage="strong_arm", max_rounds=config.MAX_REFINEMENT_ROUNDS)
    return {
        "status": result["status"],
        "scenario": result["scenario"],
        "rounds_taken": result["rounds_taken"],
        "exhausted_stage": result.get("exhausted_stage"),
    }
