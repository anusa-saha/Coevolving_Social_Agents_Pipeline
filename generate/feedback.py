import json

import config
from llm_clients import strong_arm_chat, weak_arm_chat, extract_json


def _check_pass_tally(rollouts: list, key: str) -> dict:
    tally = {}
    for r in rollouts:
        for check_id, passed in r.get(key, {}).items():
            tally.setdefault(check_id, 0)
            if passed:
                tally[check_id] += 1
    return tally


def _facts_linked_to_check(scenario: dict, check_id: str) -> list:
    return [
        {"fact_id": df["fact_id"], "owner": df["owner"], "why": df.get("why")}
        for df in scenario.get("decisive_facts", [])
        if check_id in df.get("flips", [])
    ]


def _linked_facts_by_check(scenario: dict, check_ids) -> dict:
    return {check_id: _facts_linked_to_check(scenario, check_id) for check_id in check_ids}


WEAK_ARM_SYSTEM_PROMPT = """You are the diagnostic module of a benchmark-generation pipeline that tests whether multi-agent scenarios genuinely require several agents surfacing hidden information together, rather than being solvable alone.

A candidate scenario just failed its Weak Arm gate. In every one of the rollouts below, a lone decision-maker made a one-shot decision using ONLY the shared public facts and their own private facts -- no conversation happened, and no other agent's private information was ever available to them. The gate requires at most a small number of these lone rollouts to pass; more than that means the scenario is solvable without the hidden information, which defeats its purpose.

You are given the scenario's decisive_facts (which facts are supposed to be required, which checks they gate, and why), the per-check pass tallies, and the complete data for every rollout: its settlement (broken out into credited_facts, commitments, and justification_fact_ids for you), its per-check results, its raw model output, and which decisive facts it structurally could never have accessed.

Any content check that passed in a rollout, while being linked (via decisive_facts.flips) to a fact the lone decision-maker never had access to, is proof that check is satisfiable without the hidden information -- not a coincidence, not a lucky guess, a structural leak. Read the actual passing settlement and identify concretely what let it through: a numeric threshold that's easy to hit by generic caution or fairness, a boolean that defaults true, an equal-split heuristic, or a narrative stereotype the model is pattern-matching to.

Rules:
- Ground every claim strictly in the data given. Never invent a fact, check, or number not present in it.
- Reference exact check IDs and exact fact IDs whenever the data names them. Do not write "some checks" when the data is specific.
- Your fix_instructions must tell the Challenger exactly which check(s) to tighten and how, tied to the exact decisive fact that check is supposed to depend on.

Return ONLY a JSON object of this exact shape, no other text:
{"diagnosis": "2-4 sentences describing exactly what let the checks pass without the hidden information, naming exact check IDs and fact IDs", "fix_instructions": "specific, actionable revision instructions, naming exact check IDs and fact IDs"}"""


STRONG_ARM_SYSTEM_PROMPT = """You are the diagnostic module of a benchmark-generation pipeline that tests whether multi-agent scenarios genuinely require several agents surfacing hidden information together.
A candidate scenario just failed its Strong Arm gate. Below are complete group-conversation rollouts: full transcripts (every say/reveal/settle event, in order), the resulting settlement, every content and provenance check result, and which facts were actually revealed. The gate requires most of these rollouts to pass; too many failures means the group could not reliably surface and use the hidden information.
You are given the scenario's decisive_facts (which facts are supposed to be required, which checks they gate, and why) and the complete data for every rollout: the full transcript, the settlement broken out into credited_facts, commitments, and justification_fact_ids, which facts were revealed, whether the rollout auto-failed on the turn cap, and its turn count.
For every check that failed, read the actual transcript and settlement and determine the EXACT failure mode -- do this for EVERY failing check, not just the first or most obvious one. A diagnosis that covers 2 of 4 failing checks leaves the other 2 to fail again next round, costing an entire wasted revision cycle for a problem you already had the evidence for:
- NEVER REVEALED: the fact's owner never disclosed it via a reveal action, and the settlement never cited it in justification_fact_ids.
- TELEPATHY: the settlement cites the fact in justification_fact_ids even though it was never revealed in the transcript -- the model guessed a plausible answer instead of deriving it from the conversation.
- REVEALED BUT UNUSED: the fact was revealed by its owner, but the settlement never cited it as justification.
- ENCODING MISMATCH: the fact was both revealed and cited, the settlement's value plausibly reflects an attempt to use it, yet the check still failed because the check's threshold, field name, or wording does not match how a model naturally encodes that information once told (e.g., the check expects an exact string the fact never states verbatim, or a field name the model reasonably called something else).
- MISAPPLIED: the fact was revealed and properly cited in justification_fact_ids, the check's wording is sound, but the settlement's actual value still contradicts what the fact implies -- the group had the right information and reached the wrong conclusion anyway. Do not recommend loosening the check for this failure mode. If it's consistent across multiple rollouts, name it as a possible sign the decisive fact's implication isn't stated unambiguously enough, not as a check-wording problem.
Also identify any rollout that hit the turn cap without ever settling, and judge whether that looks like a pacing problem (turn_cap too low for every fact-holder to get a natural turn) or a structural problem (the settle-eligibility condition rarely triggers).
Rules:
- Ground every claim strictly in the data given. Never invent a fact, check, or event not present in it.
- Reference exact check IDs and exact fact IDs whenever the data names them. Do not write "some checks" when the data is specific.
- Distinguish disclosure problems (fix shared_context / the consultation norm, never an agent's personality) from encoding problems (fix the check's wording) from misapplied problems (the decisive fact's implication may need to be stated less ambiguously) from telepathy (the scenario may be too predictable from public facts alone, or a narrative trope is doing the work instead of genuine information use).
- For turn-cap auto-fails, tie the classification to a concrete fix: a pacing problem calls for a specific turn_cap increase (name a number); a structural problem calls for loosening the settle-eligibility condition, not just raising turn_cap.
- Recommend changes only to the specific checks, facts, or shared_context sentences implicated by a failure you actually found. Do not suggest improving anything that passed in all 4 rollouts -- an unrelated change to an already-working part of the scenario can silently break it, costing a full extra round of rollouts to notice.
- If a fix would loosen a check's threshold or wording to make it reachable, say so explicitly and name the exact new wording -- and flag whether that loosening risks making the check newly guessable by a lone decision-maker without the fact (a LEAKED regression). Leaving the exact rewording to the Challenger's guess risks it either undershooting (still failing) or overshooting (reopening LEAKED), either of which costs another full round.
Return ONLY a JSON object of this exact shape, no other text:
{"diagnosis": "one entry per failing check: which failure mode it was and what specifically happened -- not a single 2-4 sentence summary that only covers part of the failures", "fix_instructions": "one concrete instruction per check named in diagnosis, naming exact check IDs and fact IDs, and stating which failure mode drove each one -- addressing only some of them wastes the revision cycle on the ones left out"}"""

def _call_weak_arm_llm(scenario: dict, evidence: dict) -> tuple[str, str]:
    user_message = json.dumps({"decisive_facts": scenario.get("decisive_facts", []), **evidence}, indent=2)
    raw = weak_arm_chat(
        [
            {"role": "system", "content": WEAK_ARM_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.WEAK_ARM_MAX_TOKENS,
        temperature=config.FEEDBACK_TEMPERATURE,
    )
    parsed = extract_json(raw)
    diagnosis, fix_instructions = parsed["diagnosis"], parsed["fix_instructions"]
    if not diagnosis or not fix_instructions:
        raise ValueError(f"Empty diagnosis/fix_instructions: {parsed!r}")
    return diagnosis, fix_instructions


def _call_strong_arm_llm(scenario: dict, evidence: dict) -> tuple[str, str]:
    user_message = json.dumps({"decisive_facts": scenario.get("decisive_facts", []), **evidence}, indent=2)
    raw = strong_arm_chat(
        messages=[
            {"role": "system", "content": STRONG_ARM_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=config.FEEDBACK_TEMPERATURE,
        max_tokens=config.FEEDBACK_MAX_TOKENS,
        reasoning_enabled=False,
        json_mode=True,
    )
    parsed = extract_json(raw)
    diagnosis, fix_instructions = parsed["diagnosis"], parsed["fix_instructions"]
    if not diagnosis or not fix_instructions:
        raise ValueError(f"Empty diagnosis/fix_instructions: {parsed!r}")
    return diagnosis, fix_instructions


def _feedback_verifier(scenario: dict, result: dict) -> dict:
    return {
        "stage": "verifier",
        "reject_tag": result.get("tag", "MALFORMED"),
        "diagnosis": result.get("diagnosis", ""),
        "evidence": result.get("evidence", ""),
        "evidence_data": {"raw_verdict": result},
        "fix_instructions": result.get("fix_instructions", ""),
    }


def _weak_arm_rollout_detail(rollout: dict, index: int) -> dict:
    settlement = rollout.get("settlement") or {}
    return {
        "rollout_index": index,
        "temperature": rollout.get("temperature"),
        "passed": rollout.get("passed"),
        "content_results": rollout.get("content_results", {}),
        "settlement": settlement,
        "credited_facts": settlement.get("credited_facts", []),
        "commitments": settlement.get("commitments", []),
        "justification_fact_ids": settlement.get("justification_fact_ids", []),
        "raw_output": rollout.get("raw_output"),
    }


def _feedback_weak_arm(scenario: dict, result: dict) -> dict:
    rollouts = result["rollouts"]
    n = len(rollouts)
    pass_count = result["pass_count"]
    content_tally = _check_pass_tally(rollouts, "content_results")

    leaked_checks = {
        check_id: _facts_linked_to_check(scenario, check_id)
        for check_id, count in content_tally.items()
        if count > 0 and _facts_linked_to_check(scenario, check_id)
    }

    evidence = {
        "gate": "weak_arm",
        "pass_count": pass_count,
        "n_rollouts": n,
        "max_pass_allowed": config.WEAK_ARM_MAX_PASS,
        "content_check_tally": content_tally,
        "checks_that_leaked": leaked_checks,
        "rollouts": [_weak_arm_rollout_detail(r, i) for i, r in enumerate(rollouts)],
    }

    try:
        diagnosis, fix_instructions = _call_weak_arm_llm(scenario, evidence)
        llm_call_error = None
    except Exception as e:
        llm_call_error = f"{type(e).__name__}: {e}"
        diagnosis = (
            f"LLM diagnosis call failed ({llm_call_error}). Raw evidence: {pass_count}/{n} rollouts "
            f"passed (max {config.WEAK_ARM_MAX_PASS} allowed). Checks that leaked: {sorted(leaked_checks)}."
        )
        fix_instructions = (
            f"Tighten checks {sorted(leaked_checks)} -- each passed despite depending on a decisive "
            f"fact the lone decision-maker never had access to. See evidence_data.rollouts for the "
            f"exact passing settlements."
        )

    evidence["llm_call_error"] = llm_call_error

    return {
        "stage": "weak_arm",
        "reject_tag": "LEAKED",
        "diagnosis": diagnosis,
        "evidence": json.dumps(evidence, indent=2),
        "evidence_data": evidence,
        "fix_instructions": fix_instructions,
    }


def _strong_arm_rollout_detail(rollout: dict, index: int) -> dict:
    settlement = rollout.get("settlement") or {}
    transcript = rollout.get("transcript", [])
    return {
        "rollout_index": index,
        "temperature": rollout.get("temperature"),
        "settled": rollout.get("settled"),
        "passed": rollout.get("passed"),
        "auto_failed_on_turn_cap": rollout.get("settled") is False,
        "turn_count": len(transcript),
        "revealed_facts": rollout.get("revealed", []),
        "content_results": rollout.get("content_results", {}),
        "provenance_results": rollout.get("provenance_results", {}),
        "settlement": settlement,
        "credited_facts": settlement.get("credited_facts", []),
        "commitments": settlement.get("commitments", []),
        "justification_fact_ids": settlement.get("justification_fact_ids", []),
        "transcript": transcript,
    }


def _feedback_strong_arm(scenario: dict, result: dict) -> dict:
    rollouts = result["rollouts"]
    n = len(rollouts)
    pass_count = result["pass_count"]
    settled_count = sum(1 for r in rollouts if r.get("settled"))
    content_tally = _check_pass_tally(rollouts, "content_results")
    provenance_tally = _check_pass_tally(rollouts, "provenance_results")

    all_checks = set(content_tally) | set(provenance_tally)
    failing_checks = [c for c in all_checks if content_tally.get(c, n) < n or provenance_tally.get(c, n) < n]

    evidence = {
        "gate": "strong_arm",
        "pass_count": pass_count,
        "n_rollouts": n,
        "min_pass_required": config.STRONG_ARM_MIN_PASS,
        "settled_count": settled_count,
        "auto_failed_count": n - settled_count,
        "content_check_tally": content_tally,
        "provenance_check_tally": provenance_tally,
        "facts_linked_to_failing_checks": _linked_facts_by_check(scenario, failing_checks),
        "rollouts": [_strong_arm_rollout_detail(r, i) for i, r in enumerate(rollouts)],
    }

    try:
        diagnosis, fix_instructions = _call_strong_arm_llm(scenario, evidence)
        llm_call_error = None
    except Exception as e:
        llm_call_error = f"{type(e).__name__}: {e}"
        diagnosis = (
            f"LLM diagnosis call failed ({llm_call_error}). Raw evidence: {pass_count}/{n} rollouts "
            f"passed (min {config.STRONG_ARM_MIN_PASS} required), {settled_count}/{n} settled, "
            f"{n - settled_count}/{n} auto-failed on turn cap. Failing checks: {sorted(failing_checks)}."
        )
        fix_instructions = (
            f"Inspect evidence_data.rollouts for checks {sorted(failing_checks)} -- for each, check "
            f"whether the linked decisive fact (see facts_linked_to_failing_checks) was revealed and "
            f"cited in justification_fact_ids in the failing rollouts."
        )

    evidence["llm_call_error"] = llm_call_error

    return {
        "stage": "strong_arm",
        "reject_tag": "UNCOORDINATED",
        "diagnosis": diagnosis,
        "evidence": json.dumps(evidence, indent=2),
        "evidence_data": evidence,
        "fix_instructions": fix_instructions,
    }


_STAGE_BUILDERS = {
    "verifier": _feedback_verifier,
    "weak_arm": _feedback_weak_arm,
    "strong_arm": _feedback_strong_arm,
}


def build_feedback(stage: str, scenario: dict, result: dict) -> dict:
    if stage not in _STAGE_BUILDERS:
        raise ValueError(f"Unknown stage: {stage!r}")
    return _STAGE_BUILDERS[stage](scenario, result)
