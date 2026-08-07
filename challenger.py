import json
from pathlib import Path

import config
import domain_loader
from llm_clients import gpt_chat, extract_json
from storage import next_scenario_id, record_scenario_summary, recent_scenario_summaries


def _load_prompt(name: str) -> str:
    return Path(config.PROMPTS_DIR, name).read_text()


CHALLENGER_SYSTEM_PROMPT = _load_prompt("challenger_prompt.md")


def _domain_messages(domain_key) -> list:
    if not domain_key:
        return []
    return [{"role": "system", "content": domain_loader.build_domain_block(domain_key)}]


def _diversity_message() -> list:
    recent = recent_scenario_summaries(config.CHALLENGER_DIVERSITY_WINDOW)
    if not recent:
        return []

    lines = [
        "## Do not repeat these recently generated scenarios",
        "",
        "Every scenario below was already generated. Your new scenario must be meaningfully "
        "DIFFERENT from every one of them: a different premise, a different decision-maker role, "
        "a different resource/stakes type (do not always split money, assign a slot, or pick a "
        "vendor), and a different kind of hidden lever or decisive-fact pattern (do not reuse the "
        "same 'a hidden deadline forces X' or 'a hidden defect reverses the obvious choice' shape "
        "repeatedly). Reusing a close variant of any of these counts as a rejection-worthy repeat.",
        "",
    ]
    for entry in recent:
        role = entry.get("decision_maker_role") or "?"
        desc = entry.get("description") or ""
        themes = "; ".join(t for t in entry.get("decisive_fact_themes", []) if t)
        lines.append(f"- [{entry.get('domain') or entry.get('scenario_type') or '?'}] "
                      f"decision-maker={role}: {desc}" + (f" (hidden levers: {themes})" if themes else ""))

    return [{"role": "system", "content": "\n".join(lines)}]


VALID_AGENT_COUNTS = (3, 4, 5)


def _agent_count_message(num_agents) -> list:
    if not num_agents:
        return []
    return [{
        "role": "system",
        "content": (
            f"This scenario MUST use EXACTLY {num_agents} agents -- one decision-maker and "
            f"{num_agents - 1} non-decision-makers, each owning at least one decisive fact. Do not "
            f"generate any other agent count."
        ),
    }]


def generate_scenario(scenario_type: str = None, num_agents: int = None) -> dict:
    if num_agents is not None and num_agents not in VALID_AGENT_COUNTS:
        raise ValueError(f"num_agents must be one of {VALID_AGENT_COUNTS}, got {num_agents!r}")

    domain_key = domain_loader.resolve_domain(scenario_type)

    user_msg = "Generate 1 new scenario."
    if scenario_type and not domain_key:
        user_msg += f" Prefer scenario_type: {scenario_type}."
    elif domain_key:
        user_msg += f" Set it in the {domain_loader.display_name_for(domain_key)} domain."
    if num_agents:
        user_msg += f" Use exactly {num_agents} agents."

    messages = (
        [{"role": "system", "content": CHALLENGER_SYSTEM_PROMPT}]
        + _domain_messages(domain_key)
        + _agent_count_message(num_agents)
        + _diversity_message()
        + [{"role": "user", "content": user_msg}]
    )
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=4000,
        json_mode=True,
    )
    scenario = extract_json(raw)
    scenario["scenario_id"] = next_scenario_id()
    if domain_key:
        scenario["domain"] = domain_key
    if num_agents:
        scenario["requested_num_agents"] = num_agents
    record_scenario_summary(scenario)
    return scenario


def build_feedback_message(reject_tag: str, diagnosis: str, evidence: str, fix_instructions: str) -> str:
    return (
        f"Your previous scenario was REJECTED. Tag: {reject_tag}\n\n"
        f"What went wrong: {diagnosis}\n\n"
        f"Evidence: {evidence}\n\n"
        f"What to fix: {fix_instructions}\n\n"
        f"Rules for your revision:\n"
        f"- Keep everything that already works. Change only what caused the rejection.\n"
        f"- Return the complete scenario object again, in the same JSON format as before.\n"
        f"- Return JSON only -- no explanation."
    )


def revise_scenario(previous_scenario: dict, reject_tag: str, diagnosis: str,
                     evidence: str, fix_instructions: str) -> dict:
    domain_key = previous_scenario.get("domain")
    num_agents = previous_scenario.get("requested_num_agents")
    feedback = build_feedback_message(reject_tag, diagnosis, evidence, fix_instructions)

    messages = (
        [{"role": "system", "content": CHALLENGER_SYSTEM_PROMPT}]
        + _domain_messages(domain_key)
        + _agent_count_message(num_agents)
        + [
            {"role": "user", "content": f"Here is your previous scenario:\n{json.dumps(previous_scenario, indent=2)}"},
            {"role": "user", "content": feedback},
        ]
    )
    raw = gpt_chat(
        model=config.CHALLENGER_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=7000,
        json_mode=True,
    )
    scenario = extract_json(raw)
    scenario["scenario_id"] = previous_scenario["scenario_id"]
    if domain_key:
        scenario["domain"] = domain_key
    if num_agents:
        scenario["requested_num_agents"] = num_agents
    return scenario
