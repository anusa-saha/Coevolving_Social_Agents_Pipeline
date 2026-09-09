"""
Six additional analyses, built on the same real output directory as make_figures.py,
reusing its already-tested data loading (view identical scenario/round semantics,
same domain-collision fix, same tqdm fallback).

Usage:
    python make_figures_part2.py <output_dir> <figures_out_dir>

Must be run from the same directory as make_figures.py (imports its loaders directly
rather than duplicating them).
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_figures import (
    load_all_scenarios, load_all_round_events, find_domain_dirs,
    ensure_outdir, tqdm, _vibrant, _add_headroom, TAG_COLORS, STAGE_ORDER,
)


# ---------------------------------------------------------------------------
# 1. Hardest-checks leaderboard
# ---------------------------------------------------------------------------

def plot_hardest_checks(round_events: list, outdir: Path, top_n: int = 20):
    """Aggregates content_results across every rollout in every stage-attempt into
    a per-check-id pass rate, then shows the N hardest (lowest pass rate) checks
    with a minimum sample size so a check that only ever appeared twice doesn't
    dominate the "hardest" list on pure noise."""
    tally = defaultdict(lambda: [0, 0])
    for e in round_events:
        cpt = e.get("check_pass_tally")
        if not cpt:
            continue
        for check_id, (p, t) in cpt.items():
            tally[check_id][0] += p
            tally[check_id][1] += t

    MIN_N = 10
    rates = [(cid, p / t, t) for cid, (p, t) in tally.items() if t >= MIN_N]
    if not rates:
        print("  WARNING: no checks met the minimum sample size -- skipping hardest-checks leaderboard")
        return

    rates.sort(key=lambda r: r[1])
    shown = rates[:top_n]
    labels = [f"{cid}" for cid, _, _ in shown]
    values = [r * 100 for _, r, _ in shown]
    ns = [t for _, _, t in shown]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(shown))))
    colors = plt.cm.RdYlGn(np.array(values) / 100)
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Pass rate across all rollouts (%)")
    ax.set_title(f"Hardest checks (lowest pass rate, n\u2265{MIN_N} attempts, showing {len(shown)})", pad=20)
    ax.set_xlim(0, 108)
    for b, v, n in zip(bars, values, ns):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}% (n={n})", va="center")
    fig.tight_layout()
    fig.savefig(outdir / "11_hardest_checks.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. rounds_taken distribution (overall + per domain)
# ---------------------------------------------------------------------------

def plot_rounds_taken_distribution(scenarios: list, outdir: Path):
    values = [s["rounds_taken"] for s in scenarios if s.get("rounds_taken") is not None]
    if not values:
        print("  WARNING: no rounds_taken data -- skipping distribution")
        return

    max_r = max(values)
    bins = np.arange(1, max_r + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, color="#3498DB", edgecolor="white")
    ax.axvline(np.mean(values), color="#E63946", linestyle="--", linewidth=2,
               label=f"mean = {np.mean(values):.2f}, median = {np.median(values):.0f}")
    ax.set_xlabel("Rounds taken to reach final status")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Rounds-taken distribution (n={len(values)})", pad=20)
    ax.set_xticks(range(1, max_r + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "12_rounds_taken_distribution.png", dpi=150)
    plt.close(fig)


def plot_rounds_taken_per_domain(scenarios: list, outdir: Path):
    by_domain = defaultdict(list)
    for s in scenarios:
        if s.get("rounds_taken") is not None:
            by_domain[s["domain"]].append(s["rounds_taken"])

    domains = sorted(by_domain, key=lambda d: -np.mean(by_domain[d]))
    data = [by_domain[d] for d in domains]

    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(domains)), 7))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    # Deliberately not using boxplot's labels= / tick_labels= kwarg here -- that
    # parameter was renamed between matplotlib 3.9 and 3.11 (tick_labels doesn't
    # exist before 3.9, labels is deprecated and being removed in 3.11), so neither
    # name is safe across versions. set_xticks/set_xticklabels has been stable across
    # matplotlib's entire history and sidesteps the instability entirely.
    ax.set_xticks(range(1, len(domains) + 1))
    ax.set_xticklabels(domains, rotation=35, ha="right")
    colors = _vibrant(len(domains))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel("Rounds taken")
    ax.set_title("Rounds-taken distribution per domain", pad=20)
    fig.tight_layout()
    fig.savefig(outdir / "13_rounds_taken_per_domain.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Reject-tag transition matrix
# ---------------------------------------------------------------------------

def plot_reject_tag_transitions(round_events: list, outdir: Path):
    """For each scenario, the ordered sequence of reject_tags across its FAILED
    rounds -- then counts every consecutive (tag_N -> tag_N+1) pair across all
    scenarios. Directly visualizes whether fixing one rejection tends to trigger
    a different one (the oscillation pattern), or the same one recurring."""
    by_scenario = defaultdict(list)
    for e in round_events:
        if not e["passed"] and e["reject_tag"]:
            by_scenario[e["scenario_id"]].append((e["round"], e["reject_tag"]))

    transitions = Counter()
    for sid, seq in by_scenario.items():
        seq.sort(key=lambda x: x[0])
        tags = [t for _, t in seq]
        for a, b in zip(tags, tags[1:]):
            transitions[(a, b)] += 1

    if not transitions:
        print("  WARNING: no scenarios had 2+ rejections -- skipping transition matrix")
        return

    all_tags = sorted({t for pair in transitions for t in pair})
    matrix = np.zeros((len(all_tags), len(all_tags)))
    for (a, b), count in transitions.items():
        matrix[all_tags.index(a), all_tags.index(b)] = count

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(all_tags)))
    ax.set_yticks(range(len(all_tags)))
    ax.set_xticklabels(all_tags, rotation=30, ha="right")
    ax.set_yticklabels(all_tags)
    ax.set_xlabel("Rejected as this tag next")
    ax.set_ylabel("Rejected as this tag first")
    ax.set_title("Reject-tag transitions (round N \u2192 round N+1)", pad=20)
    for i in range(len(all_tags)):
        for j in range(len(all_tags)):
            v = int(matrix[i, j])
            if v > 0:
                color = "white" if v > matrix.max() * 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Number of transitions")
    fig.tight_layout()
    fig.savefig(outdir / "14_reject_tag_transitions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Scenario-type diversity entropy per domain
# ---------------------------------------------------------------------------

def plot_diversity_entropy(scenarios: list, outdir: Path):
    """Shannon entropy of the scenario_type distribution within each domain,
    normalized to [0, 1] by dividing by log2(number of distinct types) -- a domain
    with 5 types where one type is 90% of scenarios scores low despite having a
    high raw type COUNT (which graph 02 alone can't distinguish)."""
    types_by_domain = defaultdict(list)
    for s in scenarios:
        if s["scenario_type"]:
            types_by_domain[s["domain"]].append(s["scenario_type"])

    entropy_by_domain = {}
    for domain, types in types_by_domain.items():
        counts = Counter(types)
        n = sum(counts.values())
        probs = [c / n for c in counts.values()]
        raw_entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        entropy_by_domain[domain] = raw_entropy / max_entropy if max_entropy > 0 else 0

    domains = sorted(entropy_by_domain, key=lambda d: entropy_by_domain[d])
    values = [entropy_by_domain[d] for d in domains]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(domains))))
    ax.barh(domains, values, color=_vibrant(len(domains)))
    ax.set_xlabel("Normalized diversity entropy (0 = one type dominates, 1 = perfectly even)")
    ax.set_title("Scenario-type diversity entropy per domain", pad=20)
    ax.set_xlim(0, 1.1)
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(outdir / "15_scenario_type_diversity_entropy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. rounds_taken vs agent_count / check_count
# ---------------------------------------------------------------------------

def plot_rounds_vs_agent_count(scenarios: list, outdir: Path):
    xs, ys = [], []
    for s in scenarios:
        if s.get("rounds_taken") is not None:
            xs.append(len(s["agents"]))
            ys.append(s["rounds_taken"])

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter = np.random.uniform(-0.15, 0.15, len(xs))
    ax.scatter(np.array(xs) + jitter, ys, alpha=0.4, color="#3498DB", edgecolor="none")
    means = {a: np.mean([y for x, y in zip(xs, ys) if x == a]) for a in sorted(set(xs))}
    ax.plot(list(means.keys()), list(means.values()), color="#E63946", marker="o",
            linewidth=2, markersize=8, label="mean per agent count")
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Rounds taken")
    ax.set_title("Rounds taken vs. agent count", pad=20)
    ax.set_xticks(sorted(set(xs)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "16_rounds_vs_agent_count.png", dpi=150)
    plt.close(fig)


def plot_rounds_vs_check_count(scenarios: list, outdir: Path):
    xs, ys = [], []
    for s in scenarios:
        if s.get("rounds_taken") is not None:
            xs.append(len(s["content_checks"]) + len(s["provenance_checks"]))
            ys.append(s["rounds_taken"])

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter = np.random.uniform(-0.15, 0.15, len(xs))
    ax.scatter(np.array(xs) + jitter, ys, alpha=0.4, color="#2ECC71", edgecolor="none")
    if len(set(xs)) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.02, 0.97, f"Pearson r = {corr:.2f}", transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Total checks (content + provenance)")
    ax.set_ylabel("Rounds taken")
    ax.set_title("Rounds taken vs. check count", pad=20)
    fig.tight_layout()
    fig.savefig(outdir / "17_rounds_vs_check_count.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. stage_failure_counts cross-check
# ---------------------------------------------------------------------------

def plot_stage_failure_crosscheck(scenarios: list, round_events: list, outdir: Path):
    """Two independent sources should agree: the scenario-level outcome.json's own
    stage_failure_counts field, vs. counting passed=False events per stage from the
    round-by-round walk. Filtered to round <= the scenario's official rounds_taken --
    without this filter, a small number of scenarios (confirmed: 4 of 1100 in this
    dataset) have leftover round directories on disk beyond their officially recorded
    rounds_taken, almost certainly from a retry/resume after an interrupted run.
    That's a genuine, minor data artifact worth knowing about, not a plotting bug --
    filtering here answers "do these two sources agree once measuring the same
    thing," and reports the leftover-directory count separately."""
    rounds_taken_by_scenario = {
        f"{s['domain']}::{s['scenario_id']}": s.get("rounds_taken") for s in scenarios
    }

    from_outcome = Counter()
    for s in scenarios:
        for stage, count in (s.get("stage_failure_counts") or {}).items():
            from_outcome[stage] += count

    from_events = Counter()
    leftover_scenarios = set()
    for e in round_events:
        official = rounds_taken_by_scenario.get(e["scenario_id"])
        if official is not None and e["round"] > official:
            leftover_scenarios.add(e["scenario_id"])
            continue  # beyond the official history -- excluded from the apples-to-apples count
        if not e["passed"]:
            from_events[e["stage"]] += 1

    if leftover_scenarios:
        print(f"  NOTE: {len(leftover_scenarios)} scenario(s) have round directories on disk beyond "
              f"their official rounds_taken (likely a retry/resume artifact) -- excluded from this "
              f"cross-check so both sources measure the same thing: {sorted(leftover_scenarios)}")

    stages = STAGE_ORDER
    outcome_vals = [from_outcome.get(s, 0) for s in stages]
    event_vals = [from_events.get(s, 0) for s in stages]

    mismatches = [s for s, o, e in zip(stages, outcome_vals, event_vals) if o != e]
    if mismatches:
        print(f"  NOTE: still disagrees after filtering for: {mismatches} -- this WOULD be worth "
              f"investigating further, unlike the filtered-out leftover-directory case above")

    x = np.arange(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, outcome_vals, width, label="from outcome.json", color="#3498DB")
    ax.bar(x + width / 2, event_vals, width, label="from round events (\u2264 rounds_taken)", color="#F1C40F")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in stages])
    ax.set_ylabel("Total failure count across all scenarios")
    title = "Stage-failure cross-check: two independent data sources"
    if leftover_scenarios:
        title += f"\n({len(leftover_scenarios)} scenario(s) with leftover rounds excluded, see console output)"
    ax.set_title(title, pad=20)
    _add_headroom(ax, outcome_vals + event_vals)
    ax.legend()
    for i, (o, e) in enumerate(zip(outcome_vals, event_vals)):
        ax.text(i - width / 2, o, str(o), ha="center", va="bottom")
        ax.text(i + width / 2, e, str(e), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "18_stage_failure_crosscheck.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_figures_part2.py <output_dir> <figures_out_dir>")
        sys.exit(1)

    output_dir, out_dir = sys.argv[1], sys.argv[2]
    outdir = ensure_outdir(out_dir)

    scenarios = load_all_scenarios(output_dir)
    print(f"{len(scenarios)} scenarios found across {len(find_domain_dirs(output_dir))} domain folders.\n")

    round_events = load_all_round_events(scenarios)
    print(f"{len(round_events)} round-decision events loaded.\n")

    figure_steps = [
        ("Hardest checks", lambda: plot_hardest_checks(round_events, outdir)),
        ("Rounds-taken distribution", lambda: plot_rounds_taken_distribution(scenarios, outdir)),
        ("Rounds-taken per domain", lambda: plot_rounds_taken_per_domain(scenarios, outdir)),
        ("Reject-tag transitions", lambda: plot_reject_tag_transitions(round_events, outdir)),
        ("Diversity entropy", lambda: plot_diversity_entropy(scenarios, outdir)),
        ("Rounds vs agent count", lambda: plot_rounds_vs_agent_count(scenarios, outdir)),
        ("Rounds vs check count", lambda: plot_rounds_vs_check_count(scenarios, outdir)),
        ("Stage-failure cross-check", lambda: plot_stage_failure_crosscheck(scenarios, round_events, outdir)),
    ]
    for label, fn in tqdm(figure_steps, desc="Generating figures"):
        fn()

    print(f"\nAll figures written to {outdir}/")


if __name__ == "__main__":
    main()