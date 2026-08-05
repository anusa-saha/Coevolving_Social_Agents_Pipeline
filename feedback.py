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
from llm_clients import strong_arm_chat, extract_json


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
    """How many of the rollouts each private fact was actually revealed in. Uses the UNION of
    private_facts keys and decisive_facts fact_ids, so a fact is never silently treated as
    "never revealed" just because one of those two lists happens to be incomplete."""
    fact_ids = set(scenario.get("private_facts", {}).keys())
    fact_ids |= {df.get("fact_id") for df in scenario.get("decisive_facts", []) if df.get("fact_id")}
    tally = {fid: 0 for fid in fact_ids}
    for r in rollouts:
        revealed = set(r.get("revealed", []))
        for fid in fact_ids:
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


def _checks_linked_to_fact(scenario: dict, fact_id: str) -> list:
    """All check IDs (content or provenance) that scenario['decisive_facts'] says this fact flips."""
    for df in scenario.get("decisive_facts", []):
        if df.get("fact_id") == fact_id:
            return df.get("flips", [])
    return []


def _facts_linked_to_check(scenario: dict, check_id: str) -> list:
    """All decisive_facts entries (fact_id, owner, why) whose 'flips' list names this check."""
    return [df for df in scenario.get("decisive_facts", []) if check_id in df.get("flips", [])]


def _classify_provenance_failure(rollout: dict, fact_id: str, owner: str) -> str:
    """
    For ONE rollout and ONE fact that a failing provenance check depends on, name the EXACT
    reason it failed -- mirrors the explainer's "two frauds" distinction, plus a third real mode:
      - NEVER_SURFACED_NEVER_CITED : fact wasn't revealed AND wasn't cited (settlement didn't
        even try to use it -- the plain "didn't get there" case).
      - TELEPATHY                  : fact was CITED as justification but its owner never
        revealed it via a reveal action -- the settler assumed information nobody told them.
      - REVEALED_BUT_NOT_CITED     : owner revealed it in the conversation, but the settler's
        justification_fact_ids never names it -- heard it, didn't use it.
      - REVEALED_AND_CITED         : both conditions met -- if the check STILL failed, the
        failure is coming from something else the check requires (e.g. also needing it in
        credited_facts, or a second fact), not from disclosure/citation at all.
    """
    revealed = fact_id in set(rollout.get("revealed", []))
    settlement = rollout.get("settlement") or {}
    cited = fact_id in settlement.get("justification_fact_ids", [])
    if not revealed and not cited:
        return "NEVER_SURFACED_NEVER_CITED"
    if not revealed and cited:
        return "TELEPATHY"
    if revealed and not cited:
        return "REVEALED_BUT_NOT_CITED"
    return "REVEALED_AND_CITED"


_PROVENANCE_FAILURE_EXPLANATIONS = {
    "NEVER_SURFACED_NEVER_CITED": (
        "the settler never surfaced or used the fact at all -- a plain disclosure gap, not a "
        "reasoning error"
    ),
    "TELEPATHY": (
        "the settler cited the fact as justification even though its owner never revealed it in "
        "the conversation -- the model is pattern-matching to a plausible answer instead of "
        "actually deriving it from what was said"
    ),
    "REVEALED_BUT_NOT_CITED": (
        "the owner DID reveal the fact, but the settler's justification_fact_ids never names it -- "
        "the information reached the settler but wasn't recorded as a reason for the decision"
    ),
    "REVEALED_AND_CITED": (
        "the fact was both revealed and cited, so this check's failure is NOT a disclosure "
        "problem -- something else the check requires (a second fact, a numeric threshold, a "
        "specific commitment) is unmet; re-examine the check's other conditions"
    ),
}


def _rollout_round_trip(scenario: dict, rollouts: list) -> str:
    """One line per rollout: settled?, which checks failed, which decisive facts were revealed."""
    lines = ["Per-rollout breakdown:"]
    decisive_ids = [df["fact_id"] for df in scenario.get("decisive_facts", [])]
    for i, r in enumerate(rollouts):
        if r.get("settled") is False:
            lines.append(f"  Rollout {i}: AUTO-FAILED (hit turn cap before any settle action)")
            continue
        failed_content = [k for k, v in r.get("content_results", {}).items() if not v]
        failed_prov = [k for k, v in r.get("provenance_results", {}).items() if not v]
        revealed = set(r.get("revealed", []))
        not_revealed = [fid for fid in decisive_ids if fid not in revealed]
        status = "PASSED" if r.get("passed") else "FAILED"
        detail = f"  Rollout {i}: {status}"
        if failed_content:
            detail += f" | failed content: {failed_content}"
        if failed_prov:
            detail += f" | failed provenance: {failed_prov}"
        if not_revealed:
            detail += f" | never revealed: {not_revealed}"
        lines.append(detail)
    return "\n".join(lines)


def _per_check_root_cause(scenario: dict, rollouts: list, tally: dict, key: str, n: int) -> tuple[str, dict]:
    """
    For every check that failed at least once, name its EXACT root cause using the linked
    decisive fact(s) -- this is the specific, per-check diagnosis the Challenger actually needs,
    instead of a bare pass-rate number.
    Returns (human-readable text block, structured data for evidence_data).
    """
    lines = []
    structured = {}
    for check_id, pass_count in sorted(tally.items()):
        if pass_count >= n:
            continue  # this check never failed, nothing to diagnose
        fail_count = n - pass_count
        linked_facts = _facts_linked_to_check(scenario, check_id)

        if key == "provenance_results" and linked_facts:
            mode_counts = {}
            for df in linked_facts:
                fid, owner = df["fact_id"], df["owner"]
                for r in rollouts:
                    if r.get("provenance_results", {}).get(check_id) is False:
                        mode = _classify_provenance_failure(r, fid, owner)
                        key_str = f"{fid} ({owner}): {mode}"
                        mode_counts[key_str] = mode_counts.get(key_str, 0) + 1
            lines.append(f"{check_id} failed {fail_count}/{n} times. Root cause breakdown:")
            for label, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
                fid_owner, mode = label.rsplit(": ", 1)
                lines.append(f"    {count}x -- {fid_owner} [{mode}]: {_PROVENANCE_FAILURE_EXPLANATIONS[mode]}")
            structured[check_id] = {"fail_count": fail_count, "mode_counts": mode_counts}

        elif linked_facts:
            # Content check tied to a decisive fact -- was the fact even revealed when it failed?
            revealed_but_still_failed = 0
            never_revealed_when_failed = 0
            for df in linked_facts:
                fid = df["fact_id"]
                for r in rollouts:
                    if r.get("content_results", {}).get(check_id) is False:
                        if fid in set(r.get("revealed", [])):
                            revealed_but_still_failed += 1
                        else:
                            never_revealed_when_failed += 1
            fact_ids = ", ".join(df["fact_id"] for df in linked_facts)
            lines.append(
                f"{check_id} failed {fail_count}/{n} times (linked to {fact_ids}). "
                f"Of those failures: {revealed_but_still_failed} happened even though the fact WAS "
                f"revealed (an encoding/schema problem -- the settlement didn't translate the "
                f"disclosed info into the right field/number), and {never_revealed_when_failed} "
                f"happened because the fact was never revealed at all (a disclosure problem)."
            )
            structured[check_id] = {
                "fail_count": fail_count,
                "revealed_but_still_failed": revealed_but_still_failed,
                "never_revealed_when_failed": never_revealed_when_failed,
            }
        else:
            lines.append(
                f"{check_id} failed {fail_count}/{n} times (not linked to any decisive_facts entry -- "
                f"this may be a structural check like a sum/count constraint rather than one gated by "
                f"a specific hidden fact)."
            )
            structured[check_id] = {"fail_count": fail_count}

    return "\n".join(lines), structured


# ---------------------------------------------------------------------------
# Per-stage feedback builders
# ---------------------------------------------------------------------------

def _template_diagnosis_and_fix(pass_count, n, settled_count, auto_fail_count,
                                 content_root_cause_data, provenance_root_cause_data, never_revealed):
    """
    Deterministic fallback diagnosis/fix_instructions -- used ONLY if the LLM-generated version
    (_llm_synthesize_strong_arm_feedback) fails for some reason (network error, bad JSON, etc.),
    so a feedback-generation hiccup never crashes the cascade.
    """
    diagnosis = (
        f"The group only passed {pass_count}/{n} runs (at least {config.STRONG_ARM_MIN_PASS} needed). "
        f"{settled_count}/{n} rollouts reached a real settlement; {auto_fail_count}/{n} hit the turn "
        f"cap with no settlement at all."
    )

    fix_instructions_parts = []
    if auto_fail_count > 0:
        fix_instructions_parts.append(
            f"{auto_fail_count}/{n} rollouts never even reached a settlement -- turn_cap may be too "
            f"low, or the settle-eligibility condition rarely triggers in practice."
        )
    telepathy_facts = [
        fid_owner.split(" (")[0]
        for check_data in provenance_root_cause_data.values()
        for fid_owner in check_data.get("mode_counts", {})
        if "TELEPATHY" in fid_owner
    ]
    if telepathy_facts:
        fix_instructions_parts.append(
            f"TELEPATHY detected on: {sorted(set(telepathy_facts))} -- cited as justification without "
            f"ever being revealed. Consider whether shared_context telegraphs the answer via a trope."
        )
    encoding_problem_checks = [
        cid for cid, data in content_root_cause_data.items()
        if data.get("revealed_but_still_failed", 0) > 0
    ]
    if encoding_problem_checks:
        fix_instructions_parts.append(
            f"Checks {encoding_problem_checks} failed even when the linked fact WAS revealed -- a "
            f"schema/wording problem, not a disclosure problem."
        )
    if never_revealed:
        owners = sorted({df["owner"] for df in never_revealed})
        fact_ids = sorted({df["fact_id"] for df in never_revealed})
        fix_instructions_parts.append(
            f"Facts {fact_ids} (owned by {owners}) were NEVER revealed in any rollout. Strengthen the "
            f"shared consultation norm so each owner gets a natural opening to speak. Fix the "
            f"environment only -- never change an agent's personality to make them more talkative."
        )
    if not fix_instructions_parts:
        fix_instructions_parts.append(
            "Facts are being revealed and cited correctly in some rollouts but not reliably enough."
        )
    return diagnosis, " ".join(fix_instructions_parts)


_STRONG_ARM_FEEDBACK_SYSTEM_PROMPT = """\
You just played every agent across 4 independent rollouts of a multi-agent negotiation scenario, \
in a benchmark-generation pipeline. The scenario failed its Strong Arm gate (fewer than 3 of 4 \
rollouts passed). Your job now is to explain EXACTLY what went wrong to the Challenger -- a \
separate model that will revise the scenario based on what you say -- so it fixes the real \
problem instead of guessing.

You will be given deterministic, code-computed evidence: per-check pass rates, an exact root-cause \
classification for every failing check (whether a fact was never revealed, revealed but not cited \
as justification, cited without ever being revealed [a "telepathy" failure -- the settler \
pattern-matched to a plausible answer instead of deriving it from the conversation], or revealed \
and cited correctly with the check still failing for some other reason), which decisive facts were \
never revealed in any rollout, and a per-rollout breakdown.

Ground every claim you make STRICTLY in this evidence. Do not invent facts, checks, or events that \
aren't in it. Reference exact check IDs (e.g. "C4", "P1") and exact fact IDs (e.g. "PF2") wherever \
the evidence names them -- vague references like "some checks" are not acceptable when the evidence \
names specifics.

Respond with ONLY a JSON object of this exact shape, no other text:
{"diagnosis": "2-4 sentences: what actually happened across the rollouts and why the gate failed",
 "fix_instructions": "specific, actionable instructions for the Challenger, naming exact check IDs \
and fact IDs, distinguishing disclosure problems (fix the shared environment/norms) from encoding/\
schema problems (fix the check wording) from telepathy problems (the scenario may be too tropey/\
predictable from public facts alone)"}
"""


def _llm_synthesize_strong_arm_feedback(
    scenario: dict, evidence_text: str, pass_count: int, n: int, settled_count: int,
    auto_fail_count: int, content_root_cause_data: dict, provenance_root_cause_data: dict,
    never_revealed: list,
) -> tuple[str, str]:
    """
    Calls the strong arm's OWN model (GLM, via strong_arm_chat) to write the diagnosis and
    fix_instructions, grounded in the deterministic evidence computed above. Reasoning is enabled
    for this call (unlike the strong arm's per-turn calls) since this is an analysis task, not a
    single quick action. Raises on any failure -- the caller (_feedback_strong_arm) catches it and
    falls back to _template_diagnosis_and_fix.
    """
    user_message = (
        f"SCENARIO DESCRIPTION: {scenario.get('description', '(none)')}\n\n"
        f"DECISIVE FACTS (fact_id, owner, which checks it's supposed to flip, why):\n"
        + json.dumps(scenario.get("decisive_facts", []), indent=2)
        + "\n\nDETERMINISTIC EVIDENCE FROM THE 4 ROLLOUTS:\n\n"
        + evidence_text
    )

    raw = strong_arm_chat(
        messages=[
            {"role": "system", "content": _STRONG_ARM_FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,          # this is an analysis task, not a rollout -- want it stable, not varied
        max_tokens=900,
        reasoning_enabled=True,   # unlike per-turn calls, this benefits from GLM actually thinking
        json_mode=True,
    )
    parsed = extract_json(raw)
    diagnosis = parsed["diagnosis"]
    fix_instructions = parsed["fix_instructions"]
    if not diagnosis or not fix_instructions:
        raise ValueError(f"LLM returned empty diagnosis/fix_instructions: {parsed!r}")
    return diagnosis, fix_instructions


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
    auto_fail_count = n - settled_count

    content_tally = _check_pass_tally(rollouts, "content_results")
    provenance_tally = _check_pass_tally(rollouts, "provenance_results")
    never_revealed = _never_revealed_decisive_facts(rollouts, scenario)
    example = _first_failure(rollouts)

    content_root_cause_text, content_root_cause_data = _per_check_root_cause(
        scenario, rollouts, content_tally, "content_results", n
    )
    provenance_root_cause_text, provenance_root_cause_data = _per_check_root_cause(
        scenario, rollouts, provenance_tally, "provenance_results", n
    )

    evidence_parts = [
        _format_check_tally(content_tally, n, "Content checks"),
        _format_check_tally(provenance_tally, n, "Provenance checks"),
        f"{settled_count}/{n} rollouts reached a settlement before the turn cap; "
        f"{auto_fail_count}/{n} auto-failed by running out of turns with no settle action.",
        _rollout_round_trip(scenario, rollouts),
    ]
    if content_root_cause_text:
        evidence_parts.append("EXACT root cause per failing content check:\n" + content_root_cause_text)
    if provenance_root_cause_text:
        evidence_parts.append("EXACT root cause per failing provenance check:\n" + provenance_root_cause_text)
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

    evidence_text = "\n\n".join(evidence_parts)

    try:
        diagnosis, fix_instructions = _llm_synthesize_strong_arm_feedback(
            scenario=scenario,
            evidence_text=evidence_text,
            pass_count=pass_count,
            n=n,
            settled_count=settled_count,
            auto_fail_count=auto_fail_count,
            content_root_cause_data=content_root_cause_data,
            provenance_root_cause_data=provenance_root_cause_data,
            never_revealed=never_revealed,
        )
    except Exception as e:
        # Never let a feedback-generation hiccup (network error, bad JSON, etc.) crash the
        # cascade -- fall back to the deterministic template so revise_scenario still gets
        # something usable.
        diagnosis, fix_instructions = _template_diagnosis_and_fix(
            pass_count, n, settled_count, auto_fail_count,
            content_root_cause_data, provenance_root_cause_data, never_revealed,
        )
        fix_instructions = f"[LLM diagnosis unavailable ({e}), using template fallback] " + fix_instructions

    return {
        "stage": "strong_arm",
        "reject_tag": "UNCOORDINATED",
        "diagnosis": diagnosis,
        "evidence": evidence_text,
        "evidence_data": {
            "pass_count": pass_count,
            "n_rollouts": n,
            "settled_count": settled_count,
            "auto_fail_count": auto_fail_count,
            "content_check_tally": content_tally,
            "provenance_check_tally": provenance_tally,
            "content_root_cause": content_root_cause_data,
            "provenance_root_cause": provenance_root_cause_data,
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
