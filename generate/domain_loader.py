import json
from pathlib import Path

import config

_domains_cache = None


def _load() -> dict:
    global _domains_cache
    if _domains_cache is None:
        path = Path(config.PROMPTS_DIR, "domains.json")
        with open(path) as f:
            _domains_cache = json.load(f)
    return _domains_cache


def available_domains() -> list:
    return [(key, entry.get("display_name", key)) for key, entry in _load().items()]


def resolve_domain(scenario_type_arg):
    if not scenario_type_arg:
        return None

    def _norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    target = _norm(scenario_type_arg)
    for key, entry in _load().items():
        if _norm(key) == target or _norm(entry.get("display_name", "")) == target:
            return key
    return None


def display_name_for(domain_key: str) -> str:
    return _load().get(domain_key, {}).get("display_name", domain_key)


def build_domain_block(domain_key: str) -> str:
    domains = _load()
    if domain_key not in domains:
        raise KeyError(f"Unknown domain key: {domain_key!r}. Known: {list(domains.keys())}")

    entry = domains[domain_key]
    display_name = entry.get("display_name", domain_key)
    context = entry.get("domain_context", "")
    examples = entry.get("few_shot_examples", [])

    lines = [
        f"## Domain: {display_name}",
        "",
        context,
        "",
        "### Domain-specific few-shot examples",
        "",
        "These are additional worked examples for this domain, on top of the general few-shot "
        "examples above. Match this domain's tone and typical decision types -- do not copy these "
        "examples verbatim, generate a new scenario.",
        "",
    ]
    for i, example in enumerate(examples, start=1):
        lines.append(f"#### Domain example {i}")
        lines.append("```json")
        lines.append(json.dumps(example, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
