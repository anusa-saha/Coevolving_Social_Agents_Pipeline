"""
Generalized feedback signal: ONE function builds a uniform, evidence-based
diagnosis for the Challenger, no matter which stage rejected the scenario.

Instead of a hand-written sentence per stage ("tighten your thresholds", "add
a consultation norm"), this module reads the actual rollout results and
produces concrete, checkable evidence:
  - per-check pass rates across every rollout ("C4 passed 4/4, C2 passed 1/4")
  - which specific private facts were never revealed, and by whom
  - a real example of a failing settlement/transcript, not just a summary

Every stage's feedback comes back in the same shape:
    {
        "stage": "...",
        "reject_tag": "...",
        "diagnosis": "...",           # one or two sentence human summary
        "evidence": "...",            # concrete numbers + a real example, as text
        "evidence_data": {...},       # the same evidence as raw structured data
        "fix_instructions": "...",
    }
so callers (cli.py, pipeline.py) don't need stage-specific branching to build
the message that goes to revise_scenario().
"""
import json

import config


# ---------------------------------------------------------------------------
# Shared helpers -- these all read only from the generic rollout/scenario
# shapes, nothing stage-specific.
# ---------------------------------------------------------------------------

def _check_pass_tally(rollouts: list, key: str) -> dict:
    """key = 'content_results' or 'provenance_results'. Returns {check_id: pass_count}."""
    tally = {}
    for r in rollouts:
        for check_id, passed in r.get(key, {}).items():
            tally.setdefault(check_id, 0)
            if passed:
                tally[check_id] += 1
    return tally


def _format_check_tally(tally: dict, n: int, label: str) -> str:
    if not tally:
        return f"{label}: never evaluated (no rollout produced a usable settlement)."
    lines = [f"{label} (pass rate across {n} rollouts):"]
    for check_id in sorted(tally):
        lines.append(f"  {check_id}: {tally[check_id]}/{n}")
    return "\n".join(lines)


def _revealed_fact_tally(rollouts: list, scenario: dict) -> dict:
    """How many of the rollouts each private fact was actually revealed in."""
    private_fact_ids = list(scenario.get("private_facts", {}).keys())
    tally = {fid: 0 for fid in private_fact_ids}
    for r in rollouts:
        revealed = set(r.get("revealed", []))
        for fid in private_fact_ids:
            if fid in revealed:
                tally[fid] += 1
    return tally


def _never_revealed_decisive_facts(rollouts: list, scenario: dict) -> list:
    """
    Cross-references scenario["decisive_facts"] against what actually got
    revealed. Returns a list of dicts describing each decisive fact that never
    surfaced in ANY rollout, along with which checks it was supposed to flip
    and who owns it -- this is the single most actionable signal for
    UNCOORDINATED and LEAKED alike.
    """
    revealed_tally = _revealed_fact_tally(rollouts, scenario)
    never_revealed = []
    for df in scenario.get("decisive_facts", []):
        fid = df.get("fact_id")
        if revealed_tally.get(fid, 0) == 0:
            never_revealed.append({
                "fact_id": fid,
                "owner": df.get("owner"),
                "flips": df.get("flips", []),
                "why": df.get("why"),
            })
    return never_revealed


def _settlement_excerpt(settlement) -> str:
    if not settlement:
        return "(no settlement was produced)"
    return json.dumps(settlement, indent=2)[:500]


def _transcript_excerpt(transcript: list, max_events: int = 6) -> str:
    if not transcript:
        return "(no transcript)"
    lines = []
    for event in transcript[-max_events:]:
        speaker = event.get("speaker", "?")
        etype = event.get("type")
        if etype == "say":
            lines.append(f"{speaker} (say): {str(event.get('text', ''))[:150]}")
        elif etype == "reveal":
            lines.append(f"{speaker} (reveal {event.get('fact_id')}): {str(event.get('text', ''))[:150]}")
        elif etype == "settle":
            lines.append(f"{speaker} (settle): {json.dumps(event.get('settlement', {}))[:300]}")
    return "\n".join(lines)


def _first_failure(rollouts: list) -> dict:
    """One concrete example rollout that failed, to show the Challenger real evidence, not just a count."""
    for r in rollouts:
        if not r.get("passed"):
            return r
    return rollouts[0] if rollouts else {}


# ---------------------------------------------------------------------------
# Per-stage feedback builders
# ---------------------------------------------------------------------------

def _feedback_verifier(scenario: dict, result: dict) -> dict:
    """The verifier's own LLM call already produces diagnosis/evidence/fix_instructions."""
    return {
        "stage": "verifier",
        "reject_tag": result.get("tag", "MALFORMED"),
        "diagnosis": result.get("diagnosis", ""),
        "evidence": result.get("evidence", ""),
        "evidence_data": {"raw_verdict": result},
        "fix_instructions": result.get("fix_instructions", ""),
    }


def _feedback_weak_arm(scenario: dict, result: dict) -> dict:
    rollouts = result["rollouts"]
    n = len(rollouts)
    pass_count = result["pass_count"]

    content_tally = _check_pass_tally(rollouts, "content_results")
    never_revealed = _never_revealed_decisive_facts(rollouts, scenario)
    example = _first_failure(rollouts)

    evidence_parts = [
        _format_check_tally(content_tally, n, "Content checks"),
    ]
    if never_revealed:
        lines = ["Decisive facts never needed by the lone agent to still pass some checks:"]
        for df in never_revealed:
            lines.append(
                f"  {df['fact_id']} (owner: {df['owner']}, meant to flip {df['flips']}): {df['why']}"
            )
        evidence_parts.append("\n".join(lines))
    evidence_parts.append(f"Example lone settlement:\n{_settlement_excerpt(example.get('settlement'))}")

    diagnosis = (
        f"The lone decision-maker passed {pass_count}/{n} runs "
        f"(at most {config.WEAK_ARM_MAX_PASS} allowed). Checks that should depend on hidden "
        f"information are passable without it."
    )

    fix_instructions = (
        "Use the per-check pass rates above: any check passing too often without the hidden facts "
        "needs a tighter, more specific threshold tied exactly to what its decisive_facts entry says "
        "it should flip. Do not loosen any check -- make it harder for a generic, fact-blind decision "
        "to satisfy by luck or stereotype."
    )

    return {
        "stage": "weak_arm",
        "reject_tag": "LEAKED",
        "diagnosis": diagnosis,
        "evidence": "\n\n".join(evidence_parts),
        "evidence_data": {
            "pass_count": pass_count,
            "n_rollouts": n,
            "content_check_tally": content_tally,
            "never_revealed_decisive_facts": never_revealed,
        },
        "fix_instructions": fix_instructions,
    }


def _feedback_strong_arm(scenario: dict, result: dict) -> dict:
    rollouts = result["rollouts"]
    n = len(rollouts)
    pass_count = result["pass_count"]
    settled_count = sum(1 for r in rollouts if r.get("settled"))

    content_tally = _check_pass_tally(rollouts, "content_results")
    provenance_tally = _check_pass_tally(rollouts, "provenance_results")
    never_revealed = _never_revealed_decisive_facts(rollouts, scenario)
    example = _first_failure(rollouts)

    evidence_parts = [
        _format_check_tally(content_tally, n, "Content checks"),
        _format_check_tally(provenance_tally, n, "Provenance checks"),
        f"{settled_count}/{n} rollouts reached a settlement before the turn cap "
        f"(the rest auto-failed by running out of turns).",
    ]
    if never_revealed:
        lines = ["Decisive facts that were never revealed in ANY rollout:"]
        for df in never_revealed:
            lines.append(
                f"  {df['fact_id']} (owner: {df['owner']}, meant to flip {df['flips']}): {df['why']}"
            )
        evidence_parts.append("\n".join(lines))
    evidence_parts.append(
        f"Example failing rollout -- last few turns:\n{_transcript_excerpt(example.get('transcript', []))}"
    )
    evidence_parts.append(f"Example failing settlement:\n{_settlement_excerpt(example.get('settlement'))}")

    diagnosis = (
        f"The group only passed {pass_count}/{n} runs (at least {config.STRONG_ARM_MIN_PASS} needed). "
        f"The information gap is real, but the group could not reliably surface and use the hidden facts."
    )

    if never_revealed:
        owners = sorted({df["owner"] for df in never_revealed})
        fix_instructions = (
            f"The agents who never got their decisive facts into the conversation are: {', '.join(owners)}. "
            f"Strengthen or add a shared consultation norm in shared_context so each of them gets a "
            f"natural, unmissable opening to speak before the decision is finalized. Fix the shared "
            f"environment only -- never change an agent's personality to make them more talkative or "
            f"assertive."
        )
    else:
        fix_instructions = (
            "Facts are being revealed, but the settlement still isn't using them correctly (see the "
            "provenance/content pass rates above for which checks are failing). Check whether the "
            "settlement_schema or check wording matches what a decision-maker would naturally produce "
            "once the facts are on the table."
        )

    return {
        "stage": "strong_arm",
        "reject_tag": "UNCOORDINATED",
        "diagnosis": diagnosis,
        "evidence": "\n\n".join(evidence_parts),
        "evidence_data": {
            "pass_count": pass_count,
            "n_rollouts": n,
            "settled_count": settled_count,
            "content_check_tally": content_tally,
            "provenance_check_tally": provenance_tally,
            "never_revealed_decisive_facts": never_revealed,
        },
        "fix_instructions": fix_instructions,
    }


_STAGE_BUILDERS = {
    "verifier": _feedback_verifier,
    "weak_arm": _feedback_weak_arm,
    "strong_arm": _feedback_strong_arm,
}


def build_feedback(stage: str, scenario: dict, result: dict) -> dict:
    """
    stage: "verifier" | "weak_arm" | "strong_arm"
    scenario: the scenario dict that was just tested
    result: that stage's raw return value (run_verifier's dict, or
            run_weak_arm's/run_strong_arm's {"rollouts", "pass_count", "gate_passed"} dict)
    """
    if stage not in _STAGE_BUILDERS:
        raise ValueError(f"Unknown stage: {stage!r}")
    return _STAGE_BUILDERS[stage](scenario, result)
