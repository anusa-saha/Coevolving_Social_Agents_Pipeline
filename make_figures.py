"""
Generates all 10 figures directly from your real output directory structure:

    output/<domain>/scenarios/<scenario_id>/outcome.json          (scenario-level)
    output/<domain>/scenarios/<scenario_id>/round_<n>/<stage>/outcome.json    (passed: bool)
    output/<domain>/scenarios/<scenario_id>/round_<n>/<stage>/feedback.json   (reject_tag, ...)
    output/<domain>/scenarios/<scenario_id>/round_<n>/<stage>/result.json    (rollouts: [...])

No all_iterations.jsonl and no pre-combined scenarios file needed -- this walks the
directory tree directly. .ipynb_checkpoints subdirectories are skipped explicitly.

Key semantics, confirmed against real data before writing this (not assumed):
  - A stage's outcome.json "passed" is the TRUE gate-control signal -- it's what
    actually decided whether the cascade proceeded to the next stage. A stage's
    feedback.json can carry a non-null reject_tag with fine-grained critique even
    when outcome.json says passed: true (e.g. a per-check leakage note on a gate
    that nonetheless passed overall) -- that does NOT mean the round was rejected.
  - Per round, stages run in order (verifier, weak_arm, strong_arm) and stop at the
    first one whose outcome.json says passed: false -- that stage is the round's
    true rejection point, and that stage's feedback.json holds the reject_tag that
    actually drove the revision. This matches cascade.py's real control flow.
  - result.json's "rollouts" list (present only for weak_arm/strong_arm, not
    verifier, which doesn't run rollouts) has one entry per rollout with a
    "passed": bool field -- this gives a true rollout-level pass rate (e.g. 1/4,
    3/4), not just a binary gate pass/fail, confirmed directly from real data.

Usage:
    python make_figures.py <output_dir> <figures_out_dir>

Example:
    python make_figures.py output figures/
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, **kwargs):
        """Minimal fallback if tqdm isn't installed -- prints periodic progress
        instead of a live bar, but never crashes the script over a missing
        optional dependency. Install tqdm for a real progress bar: pip install tqdm"""
        total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        label = f"{desc}: " if desc else ""
        step = max((total or 1) // 20, 1)
        for i, item in enumerate(iterable):
            if total and (i % step == 0 or i == total - 1):
                pct = 100 * (i + 1) / total
                print(f"\r{label}{i + 1}/{total} ({pct:.0f}%)", end="", flush=True)
            yield item
        if total:
            print()


STAGE_ORDER = ["verifier", "weak_arm", "strong_arm"]
TAG_COLORS = {"MALFORMED": "#E63946", "LEAKED": "#F77F00", "UNCOORDINATED": "#118AB2"}

# A vibrant, high-contrast qualitative palette -- cycles if there are more bars than colors.
VIBRANT_COLORS = [
    "#E63946", "#F1C40F", "#2ECC71", "#3498DB", "#9B59B6",
    "#E67E22", "#1ABC9C", "#FF6B9D", "#00BCD4", "#8BC34A",
    "#FF5722", "#5C6BC0", "#D4AC0D", "#EC407A", "#26A69A",
]


def _vibrant(n: int) -> list:
    return [VIBRANT_COLORS[i % len(VIBRANT_COLORS)] for i in range(n)]


def _add_headroom(ax, values, two_line_labels=False):
    """Reserve enough space above the tallest bar for its text label before it can
    reach the title. Two-line labels (e.g. '100%\\n(n=418)') need roughly double the
    single-line margin -- this is what was actually colliding with chart titles on
    real data (confirmed from the reported screenshots): a value at or near the axis
    ceiling left zero room for its own label."""
    top = max(values) if values else 1
    margin = 0.30 if two_line_labels else 0.15
    ax.set_ylim(0, top * (1 + margin) if top > 0 else 1)


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def find_domain_dirs(output_dir: str) -> list:
    root = Path(output_dir)
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "scenarios").is_dir())


def find_scenario_dirs(domain_dir: Path) -> list:
    scenarios_dir = domain_dir / "scenarios"
    return sorted(
        d for d in scenarios_dir.iterdir()
        if d.is_dir() and d.name.startswith("scenario_") and ".ipynb_checkpoints" not in d.parts
    )


def find_round_dirs(scenario_dir: Path) -> list:
    rounds = [
        d for d in scenario_dir.iterdir()
        if d.is_dir() and d.name.startswith("round_") and ".ipynb_checkpoints" not in d.parts
    ]

    def round_num(d):
        m = re.match(r"round_(\d+)", d.name)
        return int(m.group(1)) if m else 0

    return sorted(rounds, key=round_num)


def load_all_scenarios(output_dir: str) -> list:
    records = []
    domain_dirs = find_domain_dirs(output_dir)
    for domain_dir in tqdm(domain_dirs, desc="Loading scenarios by domain"):
        for sdir in find_scenario_dirs(domain_dir):
            outcome = _read_json(sdir / "outcome.json")
            if not outcome:
                continue
            scenario = outcome.get("scenario", {})
            records.append({
                "scenario_id": outcome.get("scenario_id", sdir.name),
                "domain_folder": domain_dir.name,
                "domain": scenario.get("domain", domain_dir.name),
                "status": outcome.get("status"),
                "rounds_taken": outcome.get("rounds_taken"),
                "stage_failure_counts": outcome.get("stage_failure_counts", {}),
                "scenario_dir": sdir,
                "agents": scenario.get("agents", []),
                "scenario_type": scenario.get("scenario_type"),
                "content_checks": scenario.get("content_checks", {}),
                "provenance_checks": scenario.get("provenance_checks", {}),
            })
    return records


def load_round_events(scenario_record: dict) -> list:
    """Returns one event PER STAGE ACTUALLY ATTEMPTED in each round -- not just the
    stage that ended up deciding the round's fate. A fully-successful round runs
    verifier AND weak_arm AND strong_arm, all three passing; each needs its own
    event, or pass-rate/funnel graphs would only ever see the last stage examined
    and silently miss every earlier stage's genuine passes (confirmed as a real bug
    by checking the funnel output against real data before shipping this)."""
    events = []
    for round_dir in find_round_dirs(scenario_record["scenario_dir"]):
        round_num_match = re.match(r"round_(\d+)", round_dir.name)
        round_num = int(round_num_match.group(1)) if round_num_match else None

        for stage in STAGE_ORDER:
            stage_dir = round_dir / stage
            if not stage_dir.is_dir():
                break  # this stage never ran this round -- neither did anything after it

            outcome = _read_json(stage_dir / "outcome.json")
            if outcome is None:
                break
            passed = outcome.get("passed")

            feedback = _read_json(stage_dir / "feedback.json") or {}
            reject_tag = feedback.get("reject_tag") if not passed else None

            rollout_pass_rate = None
            if stage in ("weak_arm", "strong_arm"):
                result = _read_json(stage_dir / "result.json") or {}
                rollouts = result.get("rollouts", [])
                if rollouts:
                    n_passed = sum(1 for r in rollouts if r.get("passed"))
                    rollout_pass_rate = n_passed / len(rollouts)

            events.append({
                "round": round_num,
                "stage": stage,
                "passed": passed,
                "reject_tag": reject_tag,
                "rollout_pass_rate": rollout_pass_rate,
            })

            if not passed:
                break  # cascade.py stops the round here -- later stages didn't run

    return events


def load_all_round_events(scenario_records: list) -> list:
    """scenario_id is prefixed with domain here specifically because bare scenario_id
    values collide across domains -- every domain has its own scenario_1 through
    scenario_100 (confirmed against the real data: all 11 domains have a
    'scenario_18'). Without this, any downstream aggregation keyed on scenario_id
    (the funnel's per-stage sets, the gap chart's per-scenario dicts) silently
    collapses 11 different scenarios into one slot -- caught by checking the funnel
    output against a direct count of one domain's outcome.json statuses before
    shipping this."""
    all_events = []
    for rec in tqdm(scenario_records, desc="Loading round/stage events"):
        global_id = f"{rec['domain']}::{rec['scenario_id']}"
        for ev in load_round_events(rec):
            ev["scenario_id"] = global_id
            ev["domain"] = rec["domain"]
            all_events.append(ev)
    return all_events


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_agent_count_distribution(scenarios: list, outdir: Path):
    counts = Counter(len(s["agents"]) for s in scenarios)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(x) for x in xs], ys, color=_vibrant(len(xs)))
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Agent-count distribution", pad=20)
    _add_headroom(ax, ys)
    for i, y in enumerate(ys):
        ax.text(i, y, str(y), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "01_agent_count_distribution.png", dpi=150)
    plt.close(fig)


def plot_scenario_types_per_domain(scenarios: list, outdir: Path):
    types_by_domain = defaultdict(set)
    for s in scenarios:
        if s["scenario_type"]:
            types_by_domain[s["domain"]].add(s["scenario_type"])

    domains = sorted(types_by_domain, key=lambda d: -len(types_by_domain[d]))
    counts = [len(types_by_domain[d]) for d in domains]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(domains))))
    ax.barh(domains, counts, color=_vibrant(len(domains)))
    ax.set_xlabel("Distinct scenario_type count")
    ax.set_title("Distinct scenario types per domain", pad=20)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.15 if counts else 1)
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center")
    fig.tight_layout()
    fig.savefig(outdir / "02_scenario_types_per_domain.png", dpi=150)
    plt.close(fig)


def classify_check(expr: str) -> str:
    if "any(c[" in expr or "all(c[" in expr:
        return "commitment/structural"
    if re.search(r"decisions\['\w+'\]\s*==\s*(True|False)", expr):
        return "boolean"
    if re.search(r"decisions\['\w+'\]\s*==\s*'[^']*'", expr):
        return "exact string"
    if re.search(r"decisions\['\w+'\]\s*[<>]=", expr):
        return "numeric range"
    if re.search(r"decisions\['\w+'\]\s*==\s*[\d.]+", expr):
        return "exact number"
    if "justification_fact_ids" in expr or "credited_facts" in expr or "revealed" in expr:
        return "provenance-membership"
    return "other"


def plot_check_type_composition(scenarios: list, outdir: Path):
    counts = Counter()
    for s in scenarios:
        all_checks = {**s["content_checks"], **s["provenance_checks"]}
        for expr in all_checks.values():
            counts[classify_check(expr)] += 1

    labels = sorted(counts, key=lambda k: -counts[k])
    values = [counts[l] for l in labels]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=_vibrant(len(labels)))
    ax.set_ylabel("Count")
    ax.set_title(f"Check-type composition (n={total} checks)", pad=20)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _add_headroom(ax, values, two_line_labels=True)
    for b, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({pct:.0f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "03_check_type_composition.png", dpi=150)
    plt.close(fig)


def plot_rounds_to_acceptance(round_events: list, outdir: Path):
    by_round_tag = defaultdict(lambda: Counter())
    max_round = 1
    for e in round_events:
        if e["passed"] or not e["reject_tag"]:
            continue
        by_round_tag[e["round"]][e["reject_tag"]] += 1
        max_round = max(max_round, e["round"])

    rounds = list(range(1, max_round + 1))
    tags = sorted({t for c in by_round_tag.values() for t in c})

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(rounds))
    for tag in tags:
        vals = np.array([by_round_tag[r].get(tag, 0) for r in rounds])
        ax.bar(rounds, vals, bottom=bottom, label=tag, color=TAG_COLORS.get(tag, "#888888"))
        bottom += vals

    ax.set_xlabel("Round number")
    ax.set_ylabel("Number of rejections")
    ax.set_title("Rejections per round, split by rejection tag", pad=20)
    ax.set_xticks(rounds)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "04_rounds_to_acceptance_by_tag.png", dpi=150)
    plt.close(fig)


def plot_rejection_tag_frequency_per_domain(round_events: list, scenarios: list, outdir: Path):
    n_scenarios_by_domain = Counter(s["domain"] for s in scenarios)
    rejections = defaultdict(lambda: Counter())

    for e in round_events:
        if not e["passed"] and e["reject_tag"]:
            rejections[e["domain"]][e["reject_tag"]] += 1

    domains = sorted(n_scenarios_by_domain)
    tags = sorted({t for c in rejections.values() for t in c})

    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(domains)), 7))
    x = np.arange(len(domains))
    width = 0.8 / max(len(tags), 1)
    for i, tag in enumerate(tags):
        n = np.array([max(n_scenarios_by_domain[d], 1) for d in domains])
        avg = np.array([rejections[d].get(tag, 0) for d in domains]) / n
        ax.bar(x + i * width, avg, width=width, label=tag, color=TAG_COLORS.get(tag, "#888888"))

    ax.set_xticks(x + width * (len(tags) - 1) / 2)
    ax.set_xticklabels(domains, rotation=35, ha="right")
    ax.set_ylabel("Average rejections per scenario")
    ax.set_title("Rejection-tag frequency, average per domain", pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "05_rejection_tag_frequency_per_domain.png", dpi=150)
    plt.close(fig)


def _stage_pass_rate_per_domain(round_events: list, stage: str) -> dict:
    attempts = defaultdict(lambda: [0, 0])
    for e in round_events:
        if e["stage"] != stage:
            continue
        attempts[e["domain"]][1] += 1
        if e["passed"]:
            attempts[e["domain"]][0] += 1
    return {d: (p / t if t else 0.0, t) for d, (p, t) in attempts.items()}


def _plot_stage_pass_rate(rates: dict, stage_label: str, filename: str, outdir: Path, color=None):
    domains = sorted(rates, key=lambda d: -rates[d][0])
    values = [rates[d][0] * 100 for d in domains]
    ns = [rates[d][1] for d in domains]

    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(domains)), 7))
    bars = ax.bar(domains, values, color=_vibrant(len(domains)))
    ax.set_ylabel("Pass rate (%)")
    _add_headroom(ax, values or [100], two_line_labels=True)
    ax.set_title(f"{stage_label} pass rate per domain", pad=20)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for b, v, n in zip(bars, values, ns):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%\n(n={n})", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close(fig)


def plot_verifier_pass_rate(round_events, outdir):
    rates = _stage_pass_rate_per_domain(round_events, "verifier")
    _plot_stage_pass_rate(rates, "Verifier", "07_verifier_pass_rate_per_domain.png", outdir)


def plot_weak_arm_pass_rate(round_events, outdir):
    rates = _stage_pass_rate_per_domain(round_events, "weak_arm")
    _plot_stage_pass_rate(rates, "Weak-arm", "08_weak_arm_pass_rate_per_domain.png", outdir)


def plot_strong_arm_pass_rate(round_events, outdir):
    rates = _stage_pass_rate_per_domain(round_events, "strong_arm")
    _plot_stage_pass_rate(rates, "Strong-arm", "06_strong_arm_pass_rate_per_domain.png", outdir)


def plot_funnel(scenarios: list, round_events: list, outdir: Path):
    entering = len(scenarios)
    passed_ever = {stage: set() for stage in STAGE_ORDER}
    for e in round_events:
        if e["passed"] and e["stage"] in passed_ever:
            passed_ever[e["stage"]].add(e["scenario_id"])

    stage_counts = [entering] + [len(passed_ever[s]) for s in STAGE_ORDER]
    stage_labels = ["Entered\n(Challenger)"] + [s.replace("_", " ").title() for s in STAGE_ORDER]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#3498DB", "#9B59B6", "#F1C40F", "#2ECC71"]
    bars = ax.bar(stage_labels, stage_counts, color=colors)
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Pipeline funnel: scenarios surviving each gate", pad=20)
    _add_headroom(ax, stage_counts, two_line_labels=True)
    for b, v in zip(bars, stage_counts):
        pct = 100 * v / entering if entering else 0
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({pct:.0f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "09_pipeline_funnel.png", dpi=150)
    plt.close(fig)


def plot_strong_weak_gap(round_events: list, outdir: Path):
    last_weak = {}
    last_strong = {}
    for e in round_events:
        if e["rollout_pass_rate"] is None:
            continue
        if e["stage"] == "weak_arm":
            last_weak[e["scenario_id"]] = e["rollout_pass_rate"]
        elif e["stage"] == "strong_arm":
            last_strong[e["scenario_id"]] = e["rollout_pass_rate"]

    gaps = []
    for sid in set(last_weak) & set(last_strong):
        gaps.append(last_strong[sid] - last_weak[sid])

    if not gaps:
        print("  WARNING: no scenarios had both weak_arm and strong_arm rollout data -- skipping graph 10")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(gaps, bins=20, color="#2ECC71", edgecolor="white")
    ax.axvline(np.mean(gaps), color="#E63946", linestyle="--", linewidth=2, label=f"mean = {np.mean(gaps):.2f}")
    ax.set_xlabel("Strong-arm pass rate \u2212 Weak-arm pass rate")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Distribution of the strong\u2212weak gap (n={len(gaps)})", pad=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "10_strong_weak_gap_distribution.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print("Usage: python make_figures.py <output_dir> <figures_out_dir>")
        sys.exit(1)

    output_dir, out_dir = sys.argv[1], sys.argv[2]
    outdir = ensure_outdir(out_dir)

    scenarios = load_all_scenarios(output_dir)
    print(f"{len(scenarios)} scenarios found across {len(find_domain_dirs(output_dir))} domain folders.\n")

    round_events = load_all_round_events(scenarios)
    print(f"{len(round_events)} round-decision events loaded.\n")

    figure_steps = [
        ("Agent-count distribution", lambda: plot_agent_count_distribution(scenarios, outdir)),
        ("Scenario types per domain", lambda: plot_scenario_types_per_domain(scenarios, outdir)),
        ("Check-type composition", lambda: plot_check_type_composition(scenarios, outdir)),
        ("Rounds to acceptance by tag", lambda: plot_rounds_to_acceptance(round_events, outdir)),
        ("Rejection-tag frequency per domain", lambda: plot_rejection_tag_frequency_per_domain(round_events, scenarios, outdir)),
        ("Strong-arm pass rate per domain", lambda: plot_strong_arm_pass_rate(round_events, outdir)),
        ("Verifier pass rate per domain", lambda: plot_verifier_pass_rate(round_events, outdir)),
        ("Weak-arm pass rate per domain", lambda: plot_weak_arm_pass_rate(round_events, outdir)),
        ("Pipeline funnel", lambda: plot_funnel(scenarios, round_events, outdir)),
        ("Strong-weak gap distribution", lambda: plot_strong_weak_gap(round_events, outdir)),
    ]
    for label, fn in tqdm(figure_steps, desc="Generating figures"):
        fn()

    print(f"\nAll figures written to {outdir}/")


if __name__ == "__main__":
    main()