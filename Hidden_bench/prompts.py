"""prompts.py - every prompt string in the system, and nothing else.

Three prompt families:

  AGENT   build_agent_prompt   - what a participant sees. Never privileged.
  ROUTER  build_route_prompt   - {"reason", "next_agent"}
          build_insight_prompt - {"insight"}

The router has TWO profiles that differ by exactly one block:

  student  blind       - scenario, roles, shared context, public record, schema
  teacher  privileged  - the same, PLUS every participant's private facts

That single block is the whole asymmetry, and distilling it into the student's weights
is what opd.py does. Everything else about the two prompts is byte-identical, which is
what makes the per-token reverse KL meaningful: the student and teacher are scoring the
same completion under contexts that differ only in privilege.

Ported verbatim from check/OPD_check_one/{helper,teacher_ctx}.py. The +0.136 checks_frac
result was measured against these exact strings; changing them invalidates that baseline.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Set


# ============================================================
# CONFIG
# ============================================================

class PROMPT_CFG:
    # How much dialogue history to show. None = all of it. 12 keeps the router prompt
    # under ~2.5k tokens on the longest scenarios, which is what MAX_TRAIN_LEN assumes.
    MAX_HISTORY_LINES = 12

    # Word cap written into the insight task text. Not enforced by the sampler - it is a
    # request to the model, and INSIGHT_MAX_TOKENS in opd.py is the hard limit.
    INSIGHT_MAX_WORDS = 120

    # The route output carries NO reason field. Measured on the first V2 run, the id was
    # a deterministic readout of the reason it followed: route_logp came back at -0.000
    # across 53 scored spans, i.e. P(chosen id) ~= 1.0 and the policy gradient on the
    # routing decision was identically zero. The reason had already committed the choice.
    #
    # The router is not left without deliberation - the INSIGHT is its chain-of-thought.
    # It rewrites a slot table every turn naming what the record still cannot determine,
    # it persists across turns, and it is the head distillation trains. So the split is:
    #   insight = deliberation, distilled from a privileged teacher
    #   route   = action, reinforced from reward
    # and the routing decision now lives at the one token position RL scores.
    #
    # Kept only for reference; nothing reads it any more.
    REASON_MAX_WORDS = 25


VALID_ACTIONS = ("free", "reveal", "settle")


# ============================================================
# SYSTEM PROMPTS
# ============================================================

SYS_AGENT = ("You are one participant in a multi-agent meeting. Each participant sees "
             "different information. You act by emitting exactly ONE JSON object and "
             "nothing else: no prose, no markdown fences, no explanation.")

SYS_ROUTER = ("You are the chair of a multi-agent meeting. You never speak in the meeting "
              "and you never decide the outcome. You maintain a running insight about the "
              "meeting and you decide who speaks next. You reply with exactly ONE JSON "
              "object and nothing else: no prose, no markdown fences, no explanation.")


# ============================================================
# FORMATTERS
# ============================================================

def fmt_shared(sc: Dict[str, str]) -> str:
    return "\n".join("  {}: {}".format(k, v) for k, v in sc.items()) or "  (none)"


def fmt_own(scen, aid: str, revealed: Set[str]) -> str:
    own = scen.owned_facts(aid)
    if not own:
        return "  (you hold no private facts)"
    return "\n".join("  {} [{}]: {}".format(
        f, "ALREADY REVEALED" if f in revealed else "NOT YET REVEALED",
        scen.private_facts[f]["text"]) for f in own)


def fmt_revealed(scen, revealed: Set[str]) -> str:
    if not revealed:
        return "  (nothing has been formally revealed yet)"
    return "\n".join("  {} (revealed by {}): {}".format(
        f, scen.fact_owner(f), scen.private_facts[f]["text"]) for f in sorted(revealed))


def fmt_history(transcript: Sequence[str], max_lines: Optional[int]) -> str:
    if not transcript:
        return "  (the meeting has just started)"
    lines = list(transcript)
    pre = []
    if max_lines and len(lines) > max_lines:
        pre = ["  ... earlier turns omitted ..."]
        lines = lines[-max_lines:]
    return "\n".join(pre + ["  " + l for l in lines])


def fmt_all_private(scen, revealed: Set[str]) -> str:
    """TEACHER ONLY. Every private fact, revealed or not."""
    if not scen.private_facts:
        return "  (no private facts exist in this scenario)"
    return "\n".join("  {} (owner: {}) [{}]: {}".format(
        f, v.get("owner", "?"),
        "FORMALLY REVEALED" if f in revealed else "NOT YET REVEALED", v.get("text", ""))
        for f, v in scen.private_facts.items())


# ============================================================
# THE PRIVILEGED BLOCK  (teacher profile only - the thing OPD distils)
# ============================================================

PRIVILEGED_INSIGHT_RULE = """
[PRIVILEGED VIEW - READ THIS RULE CAREFULLY]
{facts}

You can see every participant's private facts, including ones that have NOT been formally
revealed. Use them ONLY to judge what the public record is still missing and to decide
what the insight should emphasise or de-emphasise. You are SHAPING the existing insight,
never adding to it.
HARD CONSTRAINT: the insight you write is shown to every participant. It must be entirely
supported by the public record above. Never state, quote, paraphrase, summarise, allude to
or hint at the content of any fact marked NOT YET REVEALED - not its subject, its timing,
its numbers, nor its implication. If a needed fact is still hidden, the insight may say
only that a topic remains unresolved, never what the hidden answer is.
"""

PRIVILEGED_ROUTE_RULE = """
[PRIVILEGED VIEW]
{facts}

You can see every participant's private facts. Use them to judge which participant still
holds information the public record needs, and route to them. This view informs your
choice only - the "reason" field you write is logged and must not disclose the content of
any fact marked NOT YET REVEALED.
"""


# ============================================================
# ROUTER: INSIGHT
# ============================================================

INSIGHT_TASK = """[YOUR TASK]
Rewrite the CURRENT INSIGHT. It is the shared working memory handed to every participant
before they speak, and the only thing the decision-maker gets besides the raw record.
Write it in exactly three parts:

1. SETTLED VALUES - for each decision field in the schema above, the value the public
   record now determines, copied VERBATIM from a revealed fact or the shared context.
   Write UNKNOWN for any field the record does not determine. Never guess.
2. STILL OPEN - name each UNKNOWN field and state what information would settle it,
   phrased as a direct question to the room.
3. NEXT - what the decision-maker still needs before a settlement would pass its checks.

Under {words} words. Concrete over fluent. Do not invent facts.

[OUTPUT FORMAT] - one JSON object, nothing else:
{{"insight": "<the rewritten insight>"}}"""


def build_insight_prompt(scen, insight: str, transcript: Sequence[str],
                         revealed: Set[str], priv: str = "") -> str:
    return """[MEETING] {desc}

[SHARED CONTEXT] (public)
{shared}

[SETTLEMENT SCHEMA] (public - the decision-maker must fill every field)
{schema}

[FORMALLY REVEALED FACTS] (public record - only these count for the settlement)
{revealed}

[DIALOGUE SO FAR] (public)
{history}

[CURRENT INSIGHT]
{insight}
{priv}
{task}""".format(
        desc=scen.description, shared=fmt_shared(scen.shared_context),
        schema="  " + json.dumps(scen.settlement_schema, ensure_ascii=False),
        revealed=fmt_revealed(scen, revealed),
        history=fmt_history(transcript, PROMPT_CFG.MAX_HISTORY_LINES),
        insight=insight or "  (no insight yet - this is the first turn)",
        priv=priv, task=INSIGHT_TASK.format(words=PROMPT_CFG.INSIGHT_MAX_WORDS))


# ============================================================
# ROUTER: ROUTE
# ============================================================

def build_route_prompt(scen, insight: str, transcript: Sequence[str], revealed: Set[str],
                       budget: Dict[str, int], turn: int, left: int, priv: str = "") -> str:
    roster = "\n".join("  {} - {} ({}){}{}".format(
        a["agent_id"], a.get("name", a["agent_id"]), a.get("role", ""),
        "  [DECISION MAKER - the only one who may settle]"
        if a["agent_id"] == scen.decision_maker else "",
        "  [NO TURNS LEFT - cannot be chosen]" if budget.get(a["agent_id"], 0) <= 0 else "")
        for a in scen.agents)
    available = [a for a in scen.agent_ids if budget.get(a, 0) > 0]
    return """[MEETING] {desc}

[PARTICIPANTS]
{roster}

[SHARED CONTEXT] (public)
{shared}

[SETTLEMENT SCHEMA] (public - what {dm} must eventually fill)
{schema}

[FORMALLY REVEALED FACTS] (public record)
{revealed}

[DIALOGUE SO FAR] (public)
{history}

[CURRENT INSIGHT]
{insight}
{priv}
[TURN] {turn} - about {left} turn(s) left before the meeting is cut off.

[YOUR TASK]
Choose who speaks next. The meeting ends the moment {dm} settles, so every turn you spend
is a turn {dm} does not get. Route to whoever most advances the decision: someone whose
role plausibly covers a decision field the record cannot yet determine, or {dm} once the
record is complete enough to settle correctly. Too early wastes the settlement; too late
runs out the clock.

Your reasoning is already written down: CURRENT INSIGHT above is your own running note of
which decision fields the record cannot yet determine and what would settle them. Read it
and answer with the choice alone.

You may choose ONLY from: {available}

[OUTPUT FORMAT] - one JSON object, nothing else, no explanation, no other keys:
{{"next_agent": "<agent_id>"}}""".format(
        desc=scen.description, roster=roster, shared=fmt_shared(scen.shared_context),
        schema="  " + json.dumps(scen.settlement_schema, ensure_ascii=False),
        revealed=fmt_revealed(scen, revealed),
        history=fmt_history(transcript, PROMPT_CFG.MAX_HISTORY_LINES),
        insight=insight or "  (no insight yet)", priv=priv,
        turn=turn, left=left, dm=scen.decision_maker, available=", ".join(available))


# ============================================================
# ROUTER PROFILES
# ============================================================

class Profile:
    """`privileged` is the ONLY difference between the student and the teacher."""

    def __init__(self, name: str, privileged: bool = False):
        self.name = name
        self.privileged = privileged

    def _priv(self, scen, revealed, kind):
        if not self.privileged:
            return ""
        tmpl = PRIVILEGED_INSIGHT_RULE if kind == "insight" else PRIVILEGED_ROUTE_RULE
        return tmpl.format(facts=fmt_all_private(scen, revealed))

    def insight_prompt(self, scen, insight, transcript, revealed):
        return build_insight_prompt(scen, insight, transcript, revealed,
                                    self._priv(scen, revealed, "insight"))

    def route_prompt(self, scen, insight, transcript, revealed, budget, turn, left):
        return build_route_prompt(scen, insight, transcript, revealed, budget, turn, left,
                                  self._priv(scen, revealed, "route"))


PROFILES = {
    "student": Profile("student", privileged=False),
    "teacher": Profile("teacher", privileged=True),
}


# ============================================================
# AGENT
# ============================================================

def build_agent_prompt(scen, aid: str, insight: str, transcript: Sequence[str],
                       revealed: Set[str], valid: Sequence[str], turn: int,
                       left: int) -> str:
    """The participant's view. Never privileged - an agent sees only its OWN facts.

    This is check/OPD_check_one's scaffold, deliberately, not onlyRl's. onlyRl's prompt
    hands the agent an explicit "choose one action, in this order of preference" list,
    which drives reveal_selection_rate to 1.00 and leaves RL nothing to learn. Here the
    agent sits at ~0.53 decisive-reveal, which is the headroom rl.py is aimed at.
    """
    spec = scen.agent_spec(aid)
    is_dm = aid == scen.decision_maker
    unrevealed = [f for f in scen.owned_facts(aid) if f not in revealed]
    roster = "\n".join("  {} - {} ({}){}{}".format(
        a["agent_id"], a.get("name", a["agent_id"]), a.get("role", ""),
        "  [YOU]" if a["agent_id"] == aid else "",
        "  [DECISION MAKER]" if a["agent_id"] == scen.decision_maker else "")
        for a in scen.agents)
    fmt = []
    if "free" in valid:
        fmt.append('  {"action": "free", "content": "<what you say out loud>"}')
    if "reveal" in valid:
        fmt.append('  {"action": "reveal", "fact_id": "<PFx you own>", '
                   '"content": "<how you state the fact out loud>"}')
    if "settle" in valid:
        fmt.append('  {"action": "settle", "settlement": {<must match the settlement schema>}}')
    settle_block = ""
    if is_dm:
        settle_block = (
            "\n[SETTLEMENT SCHEMA] (your settlement must fill in every field)\n"
            "  {}\n"
            "  credited_facts / justification_fact_ids may cite ONLY shared-context ids and\n"
            "  facts in the FORMALLY REVEALED list above.\n\n"
            "[SETTLEMENT DECISION RULE]\n"
            "Do NOT settle merely because you can produce valid JSON. Before settling verify:\n"
            "  - the settlement follows the schema above;\n"
            "  - the public record contains what the decision needs;\n"
            "  - every private fact you cite has been formally revealed;\n"
            "  - you have not relied on another agent's hidden information.\n"
            "If something important is still missing, continue the meeting instead.\n"
        ).format(json.dumps(scen.settlement_schema, ensure_ascii=False))
    return """[AGENT] {aid} ({name})
[ROLE] {role}
[DECISION MAKER] {dm_line}
[TURN] {turn} - about {left} turn(s) left in this meeting.

[MEETING] {desc}

[PARTICIPANTS] (you cannot see anyone else's private information)
{roster}

[SHARED CONTEXT] (everyone can see this)
{shared}

[MEETING INSIGHT] (the chair's running summary of the public record, shared with everyone)
{insight}

[YOUR PRIVATE INFORMATION] (nobody else can see this until you formally reveal it)
{own}

[PRIVATE FACT DECISION RULE]
For each fact marked NOT YET REVEALED, ask: could it affect the final settlement, or an
important requirement, or cause the decision-maker to decide wrongly without it? If yes,
strongly prefer REVEAL over FREE this turn. Do not reveal an irrelevant fact.

[FORMALLY REVEALED FACTS] (the public record - only these count for the settlement)
{revealed}

[DIALOGUE SO FAR]
{history}

[OBJECTIVE]
{dm} must produce a settlement satisfying the meeting's acceptance criteria. Those criteria
depend on private facts {dm} cannot see. The team succeeds or fails together, on the
settlement alone.

[VALID ACTIONS THIS TURN]
  {valid}{reveal_hint}
{settle_block}
[NO-TELEPATHY]
  - Only the information above exists for you. Never invent or assume a fact.
  - A private fact counts ONLY if its owner emits a "reveal" action for it. Describing it
    in "free" chat does NOT put it on the record and does NOT count.
  - Never reveal a fact you do not own, and never repeat a reveal already on the record.
  - The insight above summarises the public record only. It is not a substitute for a
    formal reveal and it never contains another participant's hidden facts.

[OUTPUT FORMAT] - one JSON object, nothing else:
{fmt}

Now emit your JSON action for {aid}:""".format(
        aid=aid, name=spec.get("name", aid), role=spec.get("role", ""),
        dm_line=("YES - you are the only agent allowed to settle." if is_dm else
                 "NO - only {} may settle. If you emit \"settle\" it is INVALID.".format(
                     scen.decision_maker)),
        turn=turn, left=left, desc=scen.description, roster=roster,
        shared=fmt_shared(scen.shared_context),
        insight=insight or "  (no insight yet - this is the first turn)",
        own=fmt_own(scen, aid, revealed), revealed=fmt_revealed(scen, revealed),
        history=fmt_history(transcript, PROMPT_CFG.MAX_HISTORY_LINES),
        dm=scen.decision_maker, valid=", ".join(valid),
        reveal_hint=("\n  reveal is available for: " + ", ".join(unrevealed)) if unrevealed else "",
        settle_block=settle_block, fmt="\n".join(fmt))
