"""
Builds every LLM prompt automatically from a scenario JSON object.

Nothing in this file is scenario-specific. It only ever reads:
  agents, shared_context, private_facts, views, decision_maker,
  settlement_schema, interaction_config, and the transcript-so-far.

This is what satisfies "prompts should be automated from the JSON entry itself":
add a new scenario and these functions build correct prompts for it with zero
code changes.
"""
import json


def _dm_list(scenario: dict) -> list:
    dm = scenario["decision_maker"]
    return dm if isinstance(dm, list) else [dm]


def _agent_name(scenario: dict, agent_id: str):
    for a in scenario["agents"]:
        if a["agent_id"] == agent_id:
            return a["name"], a["role"]
    raise KeyError(f"Unknown agent_id: {agent_id}")


def _fact_text(scenario: dict, fact_id: str) -> str:
    shared = scenario["shared_context"]
    private = scenario["private_facts"]
    if fact_id in shared:
        return shared[fact_id]
    if fact_id in private:
        return private[fact_id]["text"]
    raise KeyError(f"Fact id not found in scenario: {fact_id}")


def _facts_block(scenario: dict, fact_ids: list) -> str:
    return "\n".join(f"{fid}: {_fact_text(scenario, fid)}" for fid in fact_ids)


def build_weak_arm_prompt(scenario: dict) -> list:
    """
    One-shot prompt for the lone decision-maker (no conversation).
    Built entirely from scenario["views"][decision_maker] -- nothing else.
    """
    dm = scenario["decision_maker"]
    if isinstance(dm, list):
        dm = dm[0]  # weak arm only makes sense for a single decision-maker; use the first
    dm_name, dm_role = _agent_name(scenario, dm)
    facts_block = _facts_block(scenario, scenario["views"][dm])
    schema_block = json.dumps(scenario["settlement_schema"], indent=2)

    system = (
        f"You are {dm_name}, the {dm_role}. You must make a final decision right now, alone. "
        f"There is no discussion and nobody else is in the room. Here is everything you know:\n\n"
        f"{facts_block}\n\n"
        f"Decide now. Output ONLY a JSON object matching exactly this shape (fill in real values, "
        f"do not leave placeholders):\n{schema_block}\n\n"
        f"Return JSON only. No explanation, no markdown, no extra text."
    )
    return [{"role": "system", "content": system}]


def _render_history(scenario: dict, transcript: list) -> str:
    if not transcript:
        return "(the conversation has not started yet)"
    lines = []
    for event in transcript:
        speaker_name, _ = _agent_name(scenario, event["speaker"])
        if event["type"] == "say":
            lines.append(f"{speaker_name}: {event.get('text', '')}")
        elif event["type"] == "reveal":
            lines.append(f"{speaker_name} (revealing {event.get('fact_id')}): {event.get('text', '')}")
        elif event["type"] == "settle":
            lines.append(f"{speaker_name} SETTLED with: {json.dumps(event.get('settlement', {}))}")
    return "\n".join(lines)


def build_turn_prompt(scenario: dict, speaker_id: str, transcript: list, settle_allowed: bool) -> list:
    """
    Per-turn prompt for one agent in the strong arm. Built entirely from:
      - shared_context             (Block 1: identical every turn, every agent)
      - private_facts[speaker_id]  (Block 2: only this agent's secrets, never changes)
      - transcript so far          (Block 3: grows every turn)
    """
    name, role = _agent_name(scenario, speaker_id)
    shared_block = "\n".join(f"{k}: {v}" for k, v in scenario["shared_context"].items())

    own_private = {fid: f for fid, f in scenario["private_facts"].items() if f["owner"] == speaker_id}
    private_block = "\n".join(f"{fid}: {f['text']}" for fid, f in own_private.items()) or "(nothing private)"

    history_block = _render_history(scenario, transcript)
    is_decision_maker = speaker_id in _dm_list(scenario)

    actions = [
        'say -- {"type": "say", "text": "..."}',
        'reveal -- {"type": "reveal", "fact_id": "<a fact_id you own and have not yet revealed>", "text": "..."}',
    ]
    if is_decision_maker and settle_allowed:
        schema_block = json.dumps(scenario["settlement_schema"], indent=2)
        actions.append(
            f'settle -- {{"type": "settle", "settlement": <object matching this shape>}}:\n{schema_block}'
        )

    system = (
        f"You are {name}, the {role}, in a group conversation.\n\n"
        f"What everyone knows:\n{shared_block}\n\n"
        f"What only you privately know:\n{private_block}\n\n"
        f"Conversation so far:\n{history_block}\n\n"
        f"Rules:\n"
        f"- Disclosing a private fact only counts if you use the 'reveal' action naming its fact_id. "
        f"Mentioning its content inside a plain 'say' does not count as disclosure.\n"
        f"- Take exactly one action this turn.\n\n"
        f"Available actions:\n" + "\n".join(f"- {a}" for a in actions) + "\n\n"
        f"Respond with ONLY a single JSON object for your one action. No explanation, no markdown."
    )
    return [{"role": "system", "content": system}]
