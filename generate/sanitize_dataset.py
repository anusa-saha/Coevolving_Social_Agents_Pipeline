import json
import sys


def sanitize_scenario(scenario: dict) -> dict:
    agents = [
        {
            "agent_id": a.get("agent_id", ""),
            "name": a.get("name", ""),
            "role": a.get("role", ""),
        }
        for a in scenario.get("agents", [])
    ]

    dm = scenario.get("decision_maker")
    if isinstance(dm, list):
        dm = dm[0] if dm else ""
    elif dm is None:
        dm = ""

    ic = scenario.get("interaction_config", {}) or {}
    interaction_config = {
        "turn_cap": int(ic.get("turn_cap", 0) or 0),
    }

    decisive_facts = [
        {
            "fact_id": df.get("fact_id", ""),
            "owner": df.get("owner", ""),
            "flips": [str(x) for x in df.get("flips", [])],
            "why": df.get("why", ""),
        }
        for df in scenario.get("decisive_facts", [])
    ]

    return {
        "scenario_id": scenario.get("scenario_id", ""),
        "scenario_type": scenario.get("scenario_type", ""),
        "description": scenario.get("description", ""),
        "agents": agents,
        "shared_context": scenario.get("shared_context", {}) or {},
        "private_facts": scenario.get("private_facts", {}) or {},
        "views": scenario.get("views", {}) or {},
        "decision_maker": dm,
        "interaction_config": interaction_config,
        "settlement_schema": scenario.get("settlement_schema", {}) or {},
        "acceptance_conditions": [str(x) for x in scenario.get("acceptance_conditions", [])],
        "content_checks": scenario.get("content_checks", {}) or {},
        "provenance_checks": scenario.get("provenance_checks", {}) or {},
        "decisive_facts": decisive_facts,
        "domain": scenario.get("domain") or "",
        "num_agents": len(agents),
    }


def sanitize_file(in_path: str, out_path: str) -> None:
    with open(in_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    cleaned = [sanitize_scenario(s) for s in data]

    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    canonical_keys = set(cleaned[0].keys()) if cleaned else set()
    mismatches = 0
    for i, original in enumerate(data):
        extra = set(original.keys()) - canonical_keys
        if extra:
            mismatches += 1
            sid = original.get("scenario_id", f"index_{i}")
            print(f"  {sid}: stripped extra top-level key(s) {sorted(extra)}")
        for a in original.get("agents", []):
            extra_agent_keys = set(a.keys()) - {"agent_id", "name", "role"}
            if extra_agent_keys:
                sid = original.get("scenario_id", f"index_{i}")
                print(f"  {sid}: stripped extra agent key(s) {sorted(extra_agent_keys)}")
        ic = original.get("interaction_config", {}) or {}
        extra_ic_keys = set(ic.keys()) - {"turn_cap"}
        if extra_ic_keys:
            sid = original.get("scenario_id", f"index_{i}")
            print(f"  {sid}: stripped extra interaction_config key(s) {sorted(extra_ic_keys)}")

    print(f"\nSanitized {len(cleaned)} scenario(s). {mismatches} record(s) had extra top-level fields removed.")
    print(f"Every record now has exactly these keys: {sorted(canonical_keys)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sanitize_dataset.py <input.json> <output.json>")
        sys.exit(1)
    sanitize_file(sys.argv[1], sys.argv[2])
