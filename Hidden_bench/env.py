"""env.py - the environment. Scenarios, parsing, the verifier, and the episode loop.

This is the world both learners act in, and it is deliberately separate from either
learner: rl.py owns the agent's objective, opd.py owns the router's, and neither of them
decides what a meeting IS. main.py and eval.py both drive episodes, so the loop cannot
live in either without a circular import.

One episode:

    seed insight
    repeat until settle or the turn cap:
        router.route    -> {"reason", "next_agent"}     [an RL decision]
        agent[chosen]   -> {"action", ...}              [an RL text turn]
        apply it: free / reveal <fact_id> / settle <settlement>
        router.insight  -> {"insight"}                  [an OPD step]
    verify the settlement against the scenario's checks

WHAT CHANGED FROM train_V1, and why (all four are measured, not guessed):

 1. LAST CALL.  If the record is complete, or this is the final turn, the floor goes to
    the decision-maker unconditionally.  In V1, 104/360 training episodes revealed EVERY
    fact (mean 4.00, vs 2.28 for episodes that settled), ran the full 12 turns, ended on
    a turn where `free` was the only legal action - and scored exactly 0.000, because the
    chair never called on the one agent allowed to write the answer down.  `settled` was
    0.711 and `checks_frac` was 0.519 == 0.722 x 0.719 (settle-rate x quality-given-
    settle).  This is an environment bug, not a learning problem.

 2. CREDIT SPAN INVERTED.  V1 scored `decision_prefix_chars` - the JSON prefix
    {"action": "free" - and explicitly discarded the `content` field.  Measured, that
    prefix has sequence probability 0.9994 (6 tokens, mean logp -0.0001/tok): there is
    nothing to learn there, and 61% of turns had exactly one legal action anyway.  The
    settlement object has sequence probability 0.0023 (191 tokens, -0.032/tok).  V2
    scores the TEXT (content / settlement) and never the action keyword.

 3. ROUTE DECISIONS ARE RECORDED FOR RL.  `settled | DM-routed` was 1.000 with no
    exceptions in V1 - the router alone decides whether an episode scores at all.  The
    route call now carries the char span of its `next_agent` value so rl.py can score
    exactly that, and forced/fallback picks are flagged so they are NEVER trained on.

 4. ROUTING IS SHAPED.  A route that lands on an agent still holding an undisclosed
    decisive fact earns a small per-turn bonus.  Nine routing decisions per episode
    sharing one terminal scalar is weak credit assignment; this makes it dense without
    being farmable (each fact can only be revealed once).

The router never sees a private fact (unless it is the teacher profile, training only).
An agent sees only its OWN facts. A fact enters the public record ONLY through a formal
`reveal` action by its owner - describing it in `free` chat does not count, and the
provenance checks are what enforce that.
"""

from __future__ import annotations

import ast
import functools
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import prompts
import rl

try:                                            # torch is optional at import time
    import torch as _torch
    _OOM = getattr(_torch.cuda, "OutOfMemoryError", RuntimeError)
except Exception:                               # noqa: BLE001
    _OOM = RuntimeError


# ============================================================
# CONFIG
# ============================================================

class ENV_CFG:
    # Hard cap on turns. A scenario may request a lower turn_cap of its own; the smaller
    # of the two wins.
    T_MAX = 12

    # Per-agent speaking budget. Stops the router from parking on one participant.
    MAX_TURNS_PER_AGENT = 3

    # Write an insight before turn 0, so the first speaker is not staring at a blank
    # working memory.
    SEED_INSIGHT = True

    # The dataset-construction solver (strong_arm.py::_settle_allowed_now) could not
    # settle until EVERY non-decision-maker had spoken at least once. We do not enforce
    # that, deliberately:
    #   * "when is the record complete enough to settle" is the router's central
    #     decision - gating it away would remove the thing OPD is meant to teach; and
    #   * check/OPD_check_one's baseline was measured without the gate, so turning it on
    #     makes those numbers non-comparable.
    # Set True to reproduce the generator's protocol exactly. Expect `success` to rise
    # and the router's job to get easier.
    SETTLE_REQUIRES_ALL_SPOKEN = False

    # ---- LAST CALL (see item 1 in the module docstring) ----
    # A meeting whose record is complete has nothing left to elicit; every further turn
    # is a `free` no-op that can only lose to the turn penalty. And a meeting that never
    # settles scores 0.000 no matter how well it gathered - measured over all four V1
    # eval arms, checks_frac on an unsettled episode was 0.000 in 360/360 cases.
    FORCE_DM_WHEN_COMPLETE = True     # every decisive fact revealed -> hand over
    FORCE_DM_ON_LAST_TURN = True      # one turn left -> hand over

    # MAX_TURNS_PER_AGENT exists to stop the router parking on one participant. It must
    # never be able to make settling IMPOSSIBLE - and it could: a decision-maker that
    # spends its three turns on `free` is then unavailable forever, and the episode is
    # guaranteed to score 0.000 however well it gathered. Measured on a mock run this hit
    # 2/60 episodes. The reserve exempts the DM from the cap when, and only when, a
    # hand-over is due.
    DM_RESERVE_TURN = True

    # A forced hand-over is not a decision the router made, so it is excluded from the
    # router's RL batch. Set False only to reproduce V1's behaviour.
    TRAIN_ON_FORCED_ROUTES = False

    # ---- routing shaping (item 4) ----
    # Paid per turn the router picks an agent that still holds an unrevealed DECISIVE
    # fact. Bounded by construction: a fact can only be revealed once, so the maximum
    # total is ROUTE_HIT_BONUS * n_decisive and it cannot be farmed by stalling.
    ROUTE_HIT_BONUS = 0.05


VALID_ACTIONS = prompts.VALID_ACTIONS


# ============================================================
# SCENARIO
# ============================================================

@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str = ""
    description: str = ""
    agents: List[dict] = field(default_factory=list)
    shared_context: Dict[str, str] = field(default_factory=dict)
    private_facts: Dict[str, dict] = field(default_factory=dict)
    views: Dict[str, List[str]] = field(default_factory=dict)
    decision_maker: str = ""
    interaction_config: Dict[str, Any] = field(default_factory=dict)
    settlement_schema: Dict[str, Any] = field(default_factory=dict)
    acceptance_conditions: List[str] = field(default_factory=list)
    content_checks: Dict[str, str] = field(default_factory=dict)
    provenance_checks: Dict[str, str] = field(default_factory=dict)
    decisive_facts: List[dict] = field(default_factory=list)
    domain: str = ""
    num_agents: int = 0
    uid: int = -1                    # dataset index; scenario_id is NOT unique

    @property
    def agent_ids(self) -> List[str]:
        return [a["agent_id"] for a in self.agents]

    def fact_owner(self, fid: str) -> Optional[str]:
        return (self.private_facts.get(fid) or {}).get("owner")

    def agent_spec(self, aid: str) -> dict:
        for a in self.agents:
            if a.get("agent_id") == aid:
                return a
        return {"agent_id": aid, "name": aid, "role": ""}

    def owned_facts(self, aid: str) -> List[str]:
        return [f for f, v in self.private_facts.items() if v.get("owner") == aid]

    def decisive_fact_ids(self) -> List[str]:
        """Facts that FLIP at least one check. Without these revealed, the settlement
        cannot pass - which is what makes this a social task rather than a solo one."""
        return [d["fact_id"] for d in self.decisive_facts
                if isinstance(d, dict) and d.get("fact_id") in self.private_facts]

    def n_checks(self) -> int:
        return len(self.content_checks) + len(self.provenance_checks)


def _valid_scenario(s: Scenario) -> bool:
    return bool(s.agents and s.decision_maker
                and s.decision_maker in s.agent_ids
                and (s.content_checks or s.provenance_checks))


def load_scenarios(path: str, n: int = 0):
    """Every valid scenario in a pre-split file, in file order.

    The split lives in the files (data/450_train.json, data/450_test.json: 80/20 within
    each of the 9 domains), not in a runtime shuffle. Each scenario carries its index in
    450D/dataset_450.json as `uid`, so ids are unique across both files; a file without
    uids falls back to its own row index.
    """
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    known = Scenario.__dataclass_fields__.keys()
    scen = []
    for i, s in enumerate(raw):
        obj = Scenario(**{k: v for k, v in s.items() if k in known})
        if not isinstance(s.get("uid"), int):
            obj.uid = i
        scen.append(obj)
    uids = [s.uid for s in scen]
    if len(set(uids)) != len(uids):
        raise ValueError("{}: duplicate uids".format(path))
    scen = [s for s in scen if _valid_scenario(s)]
    return scen[:n] if n else scen


# ============================================================
# PARSING
# ============================================================

@dataclass
class ParsedAction:
    action: Optional[str]
    content: str = ""
    fact_id: Optional[str] = None
    settlement: Optional[dict] = None
    parse_error: Optional[str] = None


def extract_json(text: str) -> Optional[dict]:
    """First balanced {...} that parses as a dict. String-aware brace-depth scan, so a
    JSON object with braces inside a string value still parses."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> ParsedAction:
    obj = extract_json(text)
    if obj is None:
        return ParsedAction(action=None, parse_error="no parseable JSON object")
    action = obj.get("action")
    if action not in VALID_ACTIONS:
        return ParsedAction(action=None, parse_error="unknown action {!r}".format(action))
    content = obj.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    settlement = obj.get("settlement")
    if action == "settle" and not isinstance(settlement, dict):
        return ParsedAction(action=None, parse_error="settle without a settlement object")
    fid = obj.get("fact_id")
    if action == "reveal" and not isinstance(fid, str):
        return ParsedAction(action=None, parse_error="reveal without a fact_id string")
    return ParsedAction(action=action, content=content, fact_id=fid, settlement=settlement)


# ============================================================
# CREDIT SPANS  --  which characters of a generation RL is allowed to score
# ============================================================
#
# This is the single most consequential function in the file, and train_V1 had it exactly
# backwards. Measured over train_V1's 3,205 scored turns:
#
#   span                          tokens   mean logp/tok   P(whole span)
#   {"action": "free"                  6       -0.000105          0.9994
#   {"action":"reveal","fact_id":..   14       -0.000006          0.99992
#   the settlement object            191       -0.0316            0.0023
#
# The action keyword carries no entropy because the ENVIRONMENT already decided it: 61%
# of turns offered exactly one legal action, and of the 1,000 turns where `reveal` was
# offered the agent took it 1,000 times. Scoring it computes a gradient of ~0 and adds
# pure noise to the batch. The text is where every bit of reward variance lives - the
# decision-maker passes only 0.735 of its content checks even when the record is COMPLETE
# and every fact is sitting in its context, a flat ~25% per-field error rate that does not
# vary with settlement size (r = -0.048).
#
# So: score the text, never the keyword.


def _value_span(text: str, key: str) -> Optional[tuple]:
    """(start, end) character span of the VALUE of a top-level JSON string key.

    Hand-rolled rather than json.loads-based because we need offsets into the RAW
    generation - the tokeniser works on the raw string, and re-serialising a parsed
    object would shift every offset.
    """
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', text)
    if not m:
        return None
    i = m.end()                      # first char INSIDE the opening quote
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return (i, j) if j > i else None
        j += 1
    return (i, len(text)) if len(text) > i else None


def _object_span(text: str, key: str) -> Optional[tuple]:
    """(start, end) character span of the VALUE of a top-level JSON object key."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\{', text)
    if not m:
        return None
    i = m.end() - 1
    depth, j, in_str, esc = 0, i, False, False
    while j < len(text):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
        j += 1
    return (i, len(text))


def text_span_chars(text: str, action: Optional[str]) -> Optional[tuple]:
    """The span RL scores on an AGENT turn: what the agent chose to SAY.

      free / reveal -> the `content` string (the sentence spoken into the transcript)
      settle        -> the `settlement` object (what the verifier actually grades)

    Returns None when the span cannot be located, in which case the turn is dropped from
    the batch rather than scored on the wrong characters.
    """
    if action == "settle":
        return _object_span(text, "settlement")
    if action in ("free", "reveal"):
        return _value_span(text, "content")
    return None


def next_agent_span_chars(text: str) -> Optional[tuple]:
    """The span RL scores on a ROUTE call: the agent id, and nothing else.

    train_V1 distilled the whole route completion, which is ~33 tokens of which
    `next_agent` is ~2 - so ~95% of the routing gradient was spent matching the teacher's
    prose style for the `reason` field. That is why kl_route fell 61% while `settled`
    went DOWN and routing_precision went 0.642 -> 0.599. Score the choice, not the essay.
    """
    return _value_span(text, "next_agent")


# ============================================================
# TELEPATHY GUARDRAIL
# ============================================================
#
# The OPD teacher can see every hidden fact and its insights DO leak them (0.105 vs a
# blind student's 0.006, measured in check/OPD_check_one/archive/ablation_run1.log). If a
# trained checkpoint's private leak rises materially above vanilla's, the student has
# learned to FABRICATE values it cannot see and any checks_frac gain is telepathy, not
# routing. This is the single number that decides whether insight distillation worked.
#
# Adopted verbatim from check/OPD_check_one/engine.py::private_leak - the lexical
# leak_score it replaces fires on public schema slot names, so any schema-aware insight
# inflates it. This one subtracts everything a blind router legitimately knows first.

_STOP = set("""the a an and or but if then than that this these those of to in on for with
by at from as is are was were be been being it its their there here what which who whom
whose when where why how all any both each few more most other some such no nor not only
own same so too very can will just should now must may might would could shall about into
over under again further once during before after above below up down out off has have had
do does did doing done we you they he she i me my our your his her them us will need must
meeting decision settle settlement agent turn record public""".split())

_PROC_STOP = set("""required requires still pending completed complete confirmed
confirm recorded record records already yet outstanding missing needed needs open closed
determine determines determined settle settled settlement decide decided decision
information item items topic topics field fields value values answer question questions
participant participants holding holds hold reveal revealed revealing chair route routing
turn turns schedule scheduled must should would could shall cannot note noted state
states stated provide provided given give report reported update updated""".split())


def _toks(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 3 and w not in _STOP}


def _public_vocab(scen, revealed: Set[str]) -> Set[str]:
    """Everything a blind router legitimately knows. Schema slot names live here, which is
    why naming a slot is not a leak."""
    parts = [scen.description,
             json.dumps(scen.settlement_schema, ensure_ascii=False),
             " ".join(str(v) for v in scen.shared_context.values()),
             " ".join("{} {}".format(a.get("name", ""), a.get("role", ""))
                      for a in scen.agents),
             " ".join(scen.private_facts[f].get("text", "")
                      for f in revealed if f in scen.private_facts)]
    return _toks(" ".join(parts))


def private_leak(scen, insight: str, revealed: Set[str]):
    """(worst overlap, which fact). Scores only vocabulary UNIQUE to a still-hidden fact."""
    pub = _public_vocab(scen, revealed) | _PROC_STOP
    ins = _toks(insight)
    best, who = 0.0, ""
    for fid, v in scen.private_facts.items():
        if fid in revealed:
            continue
        ft = _toks(v.get("text", "")) - pub
        if len(ft) < 3:
            continue
        o = len(ft & ins) / len(ft)
        if o > best:
            best, who = o, fid
    return best, who


def schema_slot_frac(scen, insight: str) -> Optional[float]:
    """Fraction of schema decision fields the insight names. Slot-table ~1.0, prose ~0.
    This is the metric that moved 0.291 -> 0.984 in check/OPD_check_one - i.e. the thing
    insight distillation actually teaches."""
    slots = list((scen.settlement_schema or {}).get("decisions", {}) or {})
    if not slots:
        return None
    low = (insight or "").lower()
    return sum(1 for s in slots
               if s.lower() in low or s.replace("_", " ").lower() in low) / len(slots)


# ============================================================
# VERIFIER  --  the reference grader, adopted verbatim
# ============================================================
#
# This is grader.py from the dataset-construction pipeline, not a re-implementation.
# Every scenario in dataset.json was ADMITTED by running a strong model against these
# exact semantics and requiring it to pass all content AND all provenance checks; so
# grading with anything else measures a different thing than the dataset guarantees.
#
# All 3,659 checks in dataset.json validate against the whitelist below with zero
# rejections. They are plain Python - there is no DSL, no `exists X where`, no
# `reveal(...)` predicate, no P_TELEPATHY. An earlier re-implementation here carried a
# rewrite layer for those forms and, in doing so, regex-substituted uppercase OR -> or
# INSIDE STRING LITERALS, turning `decisions['operating_room'] == 'OR 4'` into
# `== 'or 4'` and failing a correct settlement. Hence: adopt, do not re-implement.

ALLOWED_NAMES = {"decisions", "credited_facts", "commitments",
                 "justification_fact_ids", "revealed"}
ALLOWED_BUILTINS = {"any": any, "all": all, "len": len, "sum": sum}
ALLOWED_METHODS = {"values", "keys", "items", "get"}


def validate_check(check_str: str) -> ast.AST:
    """Reject anything outside the whitelist. eval() below is only as safe as this."""
    tree = ast.parse(check_str, mode="eval")

    bound_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                for target_node in ast.walk(generator.target):
                    if isinstance(target_node, ast.Name):
                        bound_names.add(target_node.id)

    allowed = ALLOWED_NAMES | bound_names

    allowed_attribute_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ALLOWED_METHODS:
                allowed_attribute_nodes.add(node.func)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed and node.id not in ALLOWED_BUILTINS:
            raise ValueError("Check uses disallowed name: {!r} in {!r}".format(node.id, check_str))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_BUILTINS:
                raise ValueError("Check calls disallowed function: {!r} in {!r}".format(
                    node.func.id, check_str))
            if isinstance(node.func, ast.Attribute) and node.func not in allowed_attribute_nodes:
                raise ValueError("Check calls disallowed method: .{}() in {!r}".format(
                    node.func.attr, check_str))
        if isinstance(node, ast.Attribute) and node not in allowed_attribute_nodes:
            raise ValueError("Disallowed attribute access: .{} in {!r}".format(
                node.attr, check_str))
        if isinstance(node, ast.IfExp):
            raise ValueError("Ternary expressions are not allowed: {!r}".format(check_str))
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda)):
            raise ValueError("Disallowed construct in check: {!r}".format(check_str))
    return tree


@functools.lru_cache(maxsize=8192)
def _compiled(check_str: str):
    """Validate once, compile once. Every episode re-grades the same ~8 expressions, and
    an AST walk per check per episode is pure waste over 360 scenarios."""
    return compile(validate_check(check_str), "<check>", "eval")


def eval_check(check_str, decisions, credited_facts, commitments,
               justification_fact_ids, revealed) -> bool:
    namespace = {
        "decisions": decisions,
        "credited_facts": credited_facts,
        "commitments": commitments,
        "justification_fact_ids": justification_fact_ids,
        "revealed": revealed,
    }
    return bool(eval(_compiled(check_str), {"__builtins__": ALLOWED_BUILTINS}, namespace))


class TerminalVerifier:
    """CONTENT checks grade the settlement alone. PROVENANCE checks additionally require
    that every cited fact reached the public record through a formal reveal - that is what
    separates "the decision-maker guessed right" from "the team surfaced it".

    Note `revealed=[]` for content, matching the reference grader: content checks are a
    property of the settlement text, not of the record. Exactly one content check in the
    450 scenarios references `revealed` at all, and passing the real set there would make
    us more lenient than the benchmark.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def _grade(self, block, settlement, revealed):
        out = {}
        for cid, expr in (getattr(self.scenario, block) or {}).items():
            try:
                out[cid] = eval_check(
                    expr,
                    decisions=settlement.get("decisions", {}) or {},
                    credited_facts=settlement.get("credited_facts", []) or [],
                    commitments=settlement.get("commitments", []) or [],
                    justification_fact_ids=settlement.get("justification_fact_ids", []) or [],
                    revealed=revealed)
            except Exception:
                # a missing key, a wrong type, a malformed settlement -> the check simply
                # does not pass. Never crash the episode on a bad generation.
                out[cid] = False
        return out

    def verify(self, settlement: Optional[dict], revealed: Set[str]):
        if not settlement:
            return {}, {}
        content = self._grade("content_checks", settlement, [])
        prov = self._grade("provenance_checks", settlement, sorted(revealed))
        return content, prov


# ============================================================
# EPISODE
# ============================================================

def _settle_allowed(scen: Scenario, spoken: Set[str]) -> bool:
    """Whether `settle` is offered to the decision-maker this turn. See
    ENV_CFG.SETTLE_REQUIRES_ALL_SPOKEN."""
    if not ENV_CFG.SETTLE_REQUIRES_ALL_SPOKEN:
        return True
    non_dm = {a for a in scen.agent_ids if a != scen.decision_maker}
    return non_dm <= spoken


# ============================================================
# EPISODE  --  batched lockstep
# ============================================================
#
# Everything runs through run_batch(). run_episode() and run_group() are thin wrappers
# over it, so training and evaluation execute LITERALLY the same code and cannot drift.
#
# WHY BATCHED. train_V1 issued three batch-1 generate() calls per turn (route, action,
# insight) and measured 152 seconds per episode - roughly 7s per generation. At batch 1
# autoregressive decode is entirely memory-bandwidth bound: the weights are streamed once
# per token no matter how many sequences ride along, so a batch of 8 costs barely more
# wall time than a batch of 1 on a 1.7B/4B model. Running the rollouts of a group (and of
# several groups) in LOCKSTEP - all of them take their route step together, then all take
# their action step together - turns N x 21 sequential calls into 21 batched ones.
# check/OPD_check_one's README called this out: "Batched lockstep rollout would be ~4x
# faster and is the first thing to add if wall-time hurts."
#
# The episodes de-synchronise (one settles at turn 3, another runs to 9). The driver just
# shrinks the batch as episodes retire, so late turns are cheaper than early ones.


class _Ep:
    """One in-flight episode. Plain state; all the decision logic lives in the driver."""

    __slots__ = ("scen", "transcript", "revealed", "insight", "settlement", "settled",
                 "budget", "order", "t_max", "decisive", "rows", "agent_turns",
                 "route_turns", "route_seq", "spoken", "invalid_actions", "reveals",
                 "unnecessary_free", "routing_hits", "routing_opps", "forced_routes",
                 "route_invalid", "leaks", "slot_fracs", "rr", "done", "t0",
                 "router_steps", "_pend")

    def __init__(self, scen):
        self.scen = scen
        self.transcript = []
        self.revealed = set()
        self.insight = ""
        self.settlement = None
        self.settled = False
        self.budget = {a: ENV_CFG.MAX_TURNS_PER_AGENT for a in scen.agent_ids}
        self.order = list(scen.agent_ids)
        self.t_max = min(scen.interaction_config.get("turn_cap", ENV_CFG.T_MAX),
                         ENV_CFG.T_MAX)
        self.decisive = set(scen.decisive_fact_ids())
        self.rows = []
        self.agent_turns = []
        self.route_turns = []
        self.route_seq = []
        self.spoken = set()
        self.invalid_actions = self.reveals = self.unnecessary_free = 0
        self.routing_hits = self.routing_opps = 0
        self.forced_routes = self.route_invalid = 0
        self.leaks = []
        self.slot_fracs = []
        self.rr = 0
        self.done = False
        self.t0 = time.time()
        self.router_steps = []
        self._pend = {}          # scratch for the current turn


def run_batch(scens, router, agent, student_profile=None, teacher_profile=None,
              record_router=False, record_agent=False, chunk=None):
    """Run len(scens) episodes concurrently, one lockstep turn at a time.

    `scens` may repeat the same Scenario (that is what run_group does for RLOO) or hold
    different ones (that is what main.py does to fill the batch). Returns one result dict
    per input, in order.

    chunk caps how many episodes are in flight at once, so a large batch degrades into
    several smaller lockstep passes rather than an OOM.
    """
    if chunk and len(scens) > chunk:
        out = []
        for i in range(0, len(scens), chunk):
            out.extend(run_batch(scens[i:i + chunk], router, agent, student_profile,
                                 teacher_profile, record_router, record_agent))
        return out

    # OOM RECOVERY. Peak memory is set by the widest batch times the longest sequence,
    # and both vary with the scenario: a 5-agent meeting with a long transcript needs
    # several times the KV cache of a short one. Rather than let the unluckiest batch of
    # a 15-hour unattended run kill it, halve the batch and retry. Worst case this
    # degrades to one episode at a time, which is slow but finishes.
    if len(scens) > 1:
        try:
            return _run_batch_inner(scens, router, agent, student_profile,
                                    teacher_profile, record_router, record_agent)
        except _OOM as exc:
            half = max(1, len(scens) // 2)
            print("[env] OOM at batch {} ({}); retrying at {}".format(
                len(scens), str(exc)[:80], half), flush=True)
            _empty_cache(router, agent)
            out = []
            for i in range(0, len(scens), half):
                out.extend(run_batch(scens[i:i + half], router, agent, student_profile,
                                     teacher_profile, record_router, record_agent))
            return out
    return _run_batch_inner(scens, router, agent, student_profile, teacher_profile,
                            record_router, record_agent)


def _empty_cache(router, agent):
    for lm in (router, agent):
        try:
            lm.release()
        except Exception:                                       # noqa: BLE001
            pass


def _run_batch_inner(scens, router, agent, student_profile=None, teacher_profile=None,
                     record_router=False, record_agent=False):

    student_profile = student_profile or prompts.PROFILES["student"]
    want_teacher = record_router and teacher_profile is not None
    eps = [_Ep(s) for s in scens]

    def emit(ep, kind, sp, tp, raw, ids, meta):
        if record_router:
            ep.router_steps.append({
                "kind": kind, "uid": ep.scen.uid, "student_prompt": sp,
                "teacher_prompt": tp, "completion": raw, "completion_ids": ids,
                "meta": meta})

    # ---------- seed insight (before turn 0) ----------
    if ENV_CFG.SEED_INSIGHT:
        sps = [student_profile.insight_prompt(e.scen, "", e.transcript, e.revealed)
               for e in eps]
        outs = router.ask_json_batch(sps, router.cfg.INSIGHT_MAX_TOKENS, "insight")
        for e, sp, (val, raw, ids, _) in zip(eps, sps, outs):
            e.insight = val.strip() if isinstance(val, str) else ""
            tp = (teacher_profile.insight_prompt(e.scen, "", e.transcript, e.revealed)
                  if want_teacher else None)
            emit(e, "insight", sp, tp, raw, ids, {"t": -1})

    # ---------- lockstep turns ----------
    for t in range(max(e.t_max for e in eps)):
        active = [e for e in eps if not e.done and t < e.t_max]
        if not active:
            break

        # ===== 1. LAST CALL, then the router for whoever still has a choice =====
        for e in active:
            avail = [a for a in e.scen.agent_ids if e.budget[a] > 0]
            left = e.t_max - t
            complete = e.decisive <= e.revealed
            due = ((ENV_CFG.FORCE_DM_WHEN_COMPLETE and complete)
                   or (ENV_CFG.FORCE_DM_ON_LAST_TURN and left <= 1))
            dm_free = (e.scen.decision_maker in avail
                       or (ENV_CFG.DM_RESERVE_TURN and due))
            e._pend = {"avail": avail, "left": left, "complete": complete,
                       "forced": due and dm_free}
            if not avail and not e._pend["forced"]:
                e.done = True

        active = [e for e in active if not e.done]
        if not active:
            break

        ask = [e for e in active if not e._pend["forced"]]
        if ask:
            sps = [student_profile.route_prompt(
                e.scen, e.insight, e.transcript, e.revealed, e.budget,
                t + 1, e._pend["left"]) for e in ask]
            outs = router.ask_json_batch(sps, router.cfg.ROUTE_MAX_TOKENS, "next_agent")
            for e, sp, (pick, raw, ids, reason) in zip(ask, sps, outs):
                e._pend.update({"sp": sp, "pick": pick, "raw": raw, "ids": ids,
                                "reason": reason})

        for e in active:
            p = e._pend
            if p["forced"]:
                p["chosen"] = e.scen.decision_maker
                p["reason"] = ("record complete -> last call" if p["complete"]
                               else "final turn -> last call")
                p["bad_pick"] = False
                p["span"] = None
                e.forced_routes += 1
            else:
                pick, avail = p["pick"], p["avail"]
                chosen = pick if isinstance(pick, str) and pick in avail else None
                p["bad_pick"] = chosen is None
                if chosen is None:
                    # NEVER hand a failed parse to the decision-maker: agent_ids[0] is
                    # the DM in 450/450 scenarios, so a round-robin from index 0 turned
                    # every parse failure into a premature settle.
                    pool = ([a for a in e.order
                             if a in avail and a != e.scen.decision_maker] or avail)
                    chosen = pool[e.rr % len(pool)]
                    e.rr += 1
                    e.route_invalid += 1
                    p["span"] = None
                else:
                    p["span"] = next_agent_span_chars(p["raw"])
                p["chosen"] = chosen

                tp = (teacher_profile.route_prompt(
                    e.scen, e.insight, e.transcript, e.revealed, e.budget,
                    t + 1, p["left"]) if want_teacher else None)
                emit(e, "route", p["sp"], tp, p["raw"], p["ids"],
                     {"t": t, "available": list(avail), "chosen": chosen,
                      "bad_pick": int(p["bad_pick"]), "dm": e.scen.decision_maker,
                      "complete": p["complete"]})

                # Only a real, parseable, multi-option choice enters the RL batch. A
                # forced hand-over, a parse failure and a single-candidate turn are all
                # things the router did not decide - training on them is exactly the
                # mistake train_V1 made on the agent side.
                if (record_router and p["span"] is not None and len(avail) > 1
                        and (ENV_CFG.TRAIN_ON_FORCED_ROUTES or not p["bad_pick"])):
                    e.route_turns.append({
                        "uid": e.scen.uid, "t": t, "system": prompts.SYS_ROUTER,
                        "user": p["sp"], "generation": p["raw"], "span": p["span"],
                        "chosen": chosen, "n_options": len(avail)})

            holders = {e.scen.fact_owner(f) for f in (e.decisive - e.revealed)}
            p["hit"] = int(p["chosen"] in holders)
            if holders:
                e.routing_opps += 1
                e.routing_hits += p["hit"]
            e.route_seq.append(p["chosen"])
            e.budget[p["chosen"]] -= 1

        # ===== 2. the agents act =====
        for e in active:
            p = e._pend
            chosen = p["chosen"]
            valid = ["free"]
            if [f for f in e.scen.owned_facts(chosen) if f not in e.revealed]:
                valid.append("reveal")
            if chosen == e.scen.decision_maker and _settle_allowed(e.scen, e.spoken):
                valid.append("settle")
                # NOW OR NEVER. Routing to the decision-maker does not make it settle -
                # in train_V1 it complied 256/260 times, and the 4 that did not still
                # scored 0.000. On the final turn `free` is removed. This does NOT
                # fabricate a settlement: the agent still generates the whole object,
                # and that text is exactly what RL trains.
                if ENV_CFG.FORCE_DM_ON_LAST_TURN and p["left"] <= 1:
                    valid = ["settle"]
            p["valid"] = valid
            p["user"] = prompts.build_agent_prompt(
                e.scen, chosen, e.insight, e.transcript, e.revealed, valid,
                t + 1, p["left"])
            p["a_max"] = (rl.RL_CFG.SETTLE_MAX_TOKENS
                          if chosen == e.scen.decision_maker
                          else rl.RL_CFG.AGENT_MAX_TOKENS)

        # Group by token budget: a batch runs until its longest member stops, so mixing
        # a 400-token settlement in with 160-token chat would pay the settlement's
        # length for every sequence. Usually one or two groups.
        by_max = {}
        for e in active:
            by_max.setdefault(e._pend["a_max"], []).append(e)
        for a_max, grp in by_max.items():
            outs = agent.generate_batch([e._pend["user"] for e in grp],
                                        prompts.SYS_AGENT, a_max)
            for e, (rendered, raw) in zip(grp, outs):
                e._pend["rendered"] = rendered
                e._pend["agent_raw"] = raw

        for e in active:
            _apply_action(e, t, record_agent)

        # ===== 3. the router rewrites the insight =====
        sps = [student_profile.insight_prompt(e.scen, e.insight, e.transcript, e.revealed)
               for e in active]
        prevs = [e.insight for e in active]
        outs = router.ask_json_batch(sps, router.cfg.INSIGHT_MAX_TOKENS, "insight")
        for e, sp, prev, (val, raw, ids, _) in zip(active, sps, prevs, outs):
            if isinstance(val, str) and val.strip():
                e.insight = val.strip()
            tp = (teacher_profile.insight_prompt(e.scen, prev, e.transcript, e.revealed)
                  if want_teacher else None)
            emit(e, "insight", sp, tp, raw, ids, {"t": t})

            pleak, _fid = private_leak(e.scen, e.insight, e.revealed)
            sfrac = schema_slot_frac(e.scen, e.insight)
            e.leaks.append(pleak)
            if sfrac is not None:
                e.slot_fracs.append(sfrac)

            p = e._pend
            e.rows.append({
                "uid": e.scen.uid, "scenario_id": e.scen.scenario_id, "t": t,
                "routed_agent": p["chosen"], "route_reason": p.get("reason", ""),
                "route_forced": int(p["forced"]), "route_invalid": int(p["bad_pick"]),
                "routing_hit": p["hit"],
                "is_decision_maker": int(p["chosen"] == e.scen.decision_maker),
                "valid_actions": "|".join(p["valid"]),
                "executed_action": p["executed"] or "INVALID",
                "fact_id": p.get("arg") or "", "invalid_reason": p.get("note", ""),
                "agent_response": p.get("response", ""),
                "agent_raw": (p["agent_raw"] or "").strip(), "insight": e.insight,
                "revealed_after": "|".join(sorted(e.revealed)),
            })
            if e.settled:
                e.done = True

    return [_finish(e) for e in eps]


def _apply_action(e, t, record_agent):
    """Parse and apply one agent generation. Pure state mutation - no model calls."""
    p = e._pend
    chosen, valid, raw = p["chosen"], p["valid"], p["agent_raw"]
    parsed = parse_action(raw)
    executed = arg = None
    note = ""
    had_unrevealed = bool([f for f in e.scen.owned_facts(chosen) if f not in e.revealed])

    if parsed.action is None or parsed.action not in valid:
        note = parsed.parse_error or "{} not valid for this agent/state".format(
            parsed.action)
        line = "{} (invalid action - ignored)".format(chosen)
        e.invalid_actions += 1
    elif parsed.action == "settle":
        if isinstance(parsed.settlement, dict) and parsed.settlement:
            e.settlement, e.settled, executed = parsed.settlement, True, "settle"
            line = "{} (settle): {}".format(chosen, json.dumps(parsed.settlement)[:400])
        else:
            note = "empty settlement object"
            line = "{} (invalid action - ignored)".format(chosen)
            e.invalid_actions += 1
    elif parsed.action == "reveal":
        fid = parsed.fact_id
        if fid not in e.scen.private_facts:
            note = "unknown fact_id {!r}".format(fid)
        elif e.scen.fact_owner(fid) != chosen:
            note = "{} is owned by {}".format(fid, e.scen.fact_owner(fid))
        elif fid in e.revealed:
            note = "{} was already revealed".format(fid)
        if note:
            line = "{} (invalid action - ignored)".format(chosen)
            e.invalid_actions += 1
        else:
            e.revealed.add(fid)
            e.reveals += 1
            executed, arg = "reveal", fid
            line = "{} (reveal {}): {}".format(
                chosen, fid, parsed.content or e.scen.private_facts[fid]["text"])
    else:
        executed = "free"
        line = "{} (free): {}".format(chosen, parsed.content)
        if had_unrevealed:
            e.unnecessary_free += 1

    e.transcript.append(line)
    if chosen != e.scen.decision_maker:
        e.spoken.add(chosen)

    # The span is what the agent SAID, never the action keyword it was funnelled into.
    # An unparseable generation yields no span and is dropped: scoring the wrong
    # characters is worse than scoring none.
    if record_agent:
        span = text_span_chars(raw, executed)
        if span is not None:
            e.agent_turns.append({
                "uid": e.scen.uid, "t": t, "agent_id": chosen,
                "is_dm": int(chosen == e.scen.decision_maker),
                "rendered_prompt": p["rendered"], "generation": raw,
                "span": span, "executed_action": executed or "INVALID"})

    p["executed"] = executed
    p["arg"] = arg
    p["note"] = note
    p["response"] = (json.dumps(parsed.settlement or {}) if parsed.action == "settle"
                     else parsed.content)


def _finish(e):
    """Terminal verification, rewards and the summary for one episode."""
    scen = e.scen
    content, prov = TerminalVerifier(scen).verify(e.settlement, e.revealed)
    n_content = sum(content.values())
    n_prov = sum(prov.values())
    n_pass = n_content + n_prov
    n_total = scen.n_checks()
    frac = round(n_pass / n_total, 4) if n_total else 0.0
    turns = len(e.route_seq)

    outcome = rl.EpisodeOutcome(
        settled=e.settled, content_results=content, provenance_results=prov,
        invalid_action_count=e.invalid_actions, valid_reveal_count=e.reveals,
        unnecessary_free_count=e.unnecessary_free, timed_out=not e.settled,
        episode_length=turns, decisive_facts_total=len(e.decisive),
        decisive_facts_revealed=len(e.decisive & e.revealed))
    success = rl.compute_official_reward(outcome)
    train_r, breakdown = rl.compute_training_reward(outcome, rl.REWARD, success)

    # TWO rewards, one environment.
    #   agents  see the task reward. What they say is what gets graded.
    #   router  sees the task reward PLUS dense routing credit, because it makes ~9
    #           decisions per episode and one terminal scalar cannot say which was good.
    route_r = train_r + ENV_CFG.ROUTE_HIT_BONUS * e.routing_hits
    for tr in e.agent_turns:
        tr["train_reward"] = round(train_r, 4)
        tr["scenario_id"] = scen.scenario_id
    for tr in e.route_turns:
        tr["train_reward"] = round(route_r, 4)
        tr["scenario_id"] = scen.scenario_id

    summary = {
        "uid": scen.uid, "scenario_id": scen.scenario_id, "domain": scen.domain,
        "n_agents": scen.num_agents or len(scen.agents),
        "content_passed": n_content, "content_total": len(scen.content_checks),
        "prov_passed": n_prov, "prov_total": len(scen.provenance_checks),
        "checks_passed": n_pass, "checks_total": n_total, "checks_frac": frac,
        "success": success,
        "reveals": e.reveals,
        "decisive_revealed": len(e.decisive & e.revealed),
        "decisive_total": len(e.decisive),
        "settled": int(e.settled),
        "turns_used": turns, "turns_to_settle": turns if e.settled else "",
        # routing diagnostics (train_V1 computed bad_pick and threw it away, which is why
        # "the router got worse" stayed invisible for 17 hours)
        "routing_precision": (round(e.routing_hits / e.routing_opps, 4)
                              if e.routing_opps else ""),
        "routing_hits": e.routing_hits, "routing_opportunities": e.routing_opps,
        "route_invalid": e.route_invalid, "forced_routes": e.forced_routes,
        "route_decisions": len(e.route_turns),
        "insight_private_leak_max": round(max(e.leaks), 4) if e.leaks else "",
        "insight_slot_frac": round(e.slot_fracs[-1], 4) if e.slot_fracs else "",
        "train_reward": round(train_r, 4), "route_reward": round(route_r, 4),
        "route_sequence": "|".join(e.route_seq),
        "wall_time": round(time.time() - e.t0, 1),
    }
    episode = {
        "uid": scen.uid, "scenario_id": scen.scenario_id,
        "transcript": e.transcript, "revealed": sorted(e.revealed),
        "settlement": e.settlement, "content_results": content,
        "provenance_results": prov, "final_insight": e.insight,
        "route_sequence": e.route_seq, "reward_breakdown": breakdown,
        "turns": e.rows, "summary": summary,
    }
    return {"summary": summary, "rows": e.rows, "episode": episode,
            "agent_turns": e.agent_turns, "route_turns": e.route_turns,
            "router_steps": e.router_steps,
            "train_reward": train_r, "route_reward": route_r}


def run_episode(scen: Scenario, router, agent, student_profile=None,
                teacher_profile=None, record_router=None, record_agent=False) -> dict:
    """One meeting. A batch of one - eval and training therefore run the same code.

    `record_router` is accepted as a truthy flag or as the old callback; either way the
    recorded steps come back on the result as out["router_steps"].
    """
    out = run_batch([scen], router, agent, student_profile, teacher_profile,
                    record_router=bool(record_router), record_agent=record_agent)[0]
    if callable(record_router):
        for d in out["router_steps"]:
            record_router(d)
    return out


def run_group(scen: Scenario, router, agent, group_size: int,
              record_router: bool = False, chunk=None, **kw):
    """G independent rollouts of the SAME scenario, run concurrently.

    This is the whole reason V2 can learn where V1 could not. Measured across train_V1's
    four eval arms on the same 90 held-out scenarios:

        BETWEEN-scenario variance (how hard the scenario is)  0.1204   84.2%
        WITHIN-scenario  variance (how good the policy is)    0.0225   15.8%

    train_V1 subtracted the mean over a batch of four DIFFERENT scenarios, so ~84% of
    every advantage it computed was "which scenario did I draw" rather than "did my policy
    do well". A group baseline subtracts the mean over rollouts of the SAME scenario,
    where difficulty is identical by construction and cancels exactly.

    Sampling is stochastic (agent temperature 0.8, router 0.3), so the G rollouts
    genuinely differ: a settlement object has measured sequence probability ~0.0023.
    """
    return run_batch([scen] * group_size, router, agent,
                     record_router=record_router, chunk=chunk, **kw)


def run_groups(scens, router, agent, group_size: int, record_router: bool = False,
               chunk=None, **kw):
    """Several scenarios' groups in ONE lockstep batch.

    Returns a list of per-scenario lists. Filling the batch across scenarios as well as
    across rollouts is free throughput: the decode is bandwidth-bound either way, and a
    wider batch simply amortises the weight streaming over more sequences.
    """
    flat = [s for s in scens for _ in range(group_size)]
    out = run_batch(flat, router, agent, record_router=record_router, chunk=chunk, **kw)
    return [out[i * group_size:(i + 1) * group_size] for i in range(len(scens))]


# ============================================================
# CSV SCHEMAS
# ============================================================

TURN_COLS = ["uid", "scenario_id", "t", "routed_agent", "route_reason",
             "route_forced", "route_invalid", "routing_hit",
             "is_decision_maker", "valid_actions", "executed_action", "fact_id",
             "invalid_reason", "agent_response", "agent_raw", "insight",
             "revealed_after"]

SUM_COLS = ["uid", "scenario_id", "domain", "n_agents",
            "content_passed", "content_total", "prov_passed", "prov_total",
            "checks_passed", "checks_total", "checks_frac", "success",
            "reveals", "decisive_revealed", "decisive_total",
            "settled", "turns_used", "turns_to_settle",
            "routing_precision", "routing_hits", "routing_opportunities",
            "route_invalid", "forced_routes", "route_decisions",
            "insight_private_leak_max", "insight_slot_frac",
            "train_reward", "route_reward", "route_sequence", "wall_time"]

RL_TURN_COLS = ["round", "phase", "gstep", "uid", "scenario_id", "t", "who",
                "agent_id", "is_dm", "executed_action", "n_chars", "n_tokens",
                "reward", "group_mean", "advantage", "logp_mean"]


def rl_turn_row(tr: dict) -> dict:
    """One scored span, reduced to the numbers you would actually look at when the RL
    curve misbehaves. Prompt and generation are deliberately NOT here - they are already
    in episodes.jsonl and would make this file unreadable.

    What each column tells you when things go wrong:

      n_tokens ~= 6         the credit span slipped back onto the JSON keyword. In V1
                            that was the whole bug: 6 tokens at logp -0.0001 each.
      logp_mean ~= 0        the span has no entropy, so the gradient is ~0. A healthy
                            settlement span sits near -0.03/token; a healthy route
                            choice near -0.1 to -0.5 on the agent-id token.
      advantage == 0        this turn's group was degenerate (every rollout scored the
                            same). Watch `degenerate_frac` in train_steps.csv - if it
                            climbs above ~0.3 the group size is too small or the reward
                            has saturated.
      group_mean flat       the policy is not moving. Compare against `reward`.
    """
    ad = tr.get("advantage")
    lp = tr.get("logp_mean")
    sp = tr.get("span") or (0, 0)
    return {
        "uid": tr.get("uid"), "scenario_id": tr.get("scenario_id", ""),
        "t": tr.get("t"), "who": tr.get("who", ""),
        "agent_id": tr.get("agent_id", tr.get("chosen", "")),
        "is_dm": tr.get("is_dm", ""),
        "executed_action": tr.get("executed_action", ""),
        "n_chars": sp[1] - sp[0],
        "n_tokens": tr.get("n_tokens", ""),
        "reward": tr.get("train_reward", ""),
        "group_mean": round(tr["group_mean"], 4) if tr.get("group_mean") is not None else "",
        "advantage": round(ad, 6) if ad is not None else "",
        "logp_mean": round(lp, 6) if lp is not None else "",
    }


# The reported metric set, and whether higher is better.
METRICS = [
    ("checks_frac", True),
    ("checks_passed", True),
    ("success", True),
    ("content_passed", True),
    ("prov_passed", True),
    ("reveals", True),
    ("decisive_revealed", True),
    ("settled", True),
    ("turns_to_settle", False),
    ("turns_used", False),
    ("routing_precision", True),
    ("route_invalid", False),
    # Guardrail, not an objective. If this rises materially above vanilla's, the student
    # has learned to fabricate hidden values and any checks_frac gain is telepathy.
    ("insight_private_leak_max", False),
    ("insight_slot_frac", True),
    ("train_reward", True),
]


# ============================================================
# PREFLIGHT
# ============================================================

def preflight(router, agent, scenarios, log, abort: bool = True) -> bool:
    """Cheap checks that fail loudly NOW instead of silently wasting a 10-hour run.

    The one that matters most: Qwen3 reasons by default. If `enable_thinking=False`
    does not take effect, every generation opens with a <think> block, the 256-token
    route budget is consumed before "next_agent" is ever emitted, EVERY route parses as
    invalid, and the run produces a curve that looks like training and means nothing.
    """
    ok = True
    log("")
    log("-" * 78)
    log("PREFLIGHT")
    log("-" * 78)

    # --- 1. every check in this split is gradeable ---
    bad = []
    for s in scenarios:
        for block in ("content_checks", "provenance_checks"):
            for cid, expr in (getattr(s, block) or {}).items():
                try:
                    validate_check(expr)
                except Exception as exc:
                    bad.append((s.uid, cid, str(exc)[:70]))
    n_checks = sum(s.n_checks() for s in scenarios)
    if bad:
        ok = False
        log("  [FAIL] {} of {} checks do not validate".format(len(bad), n_checks))
        for b in bad[:5]:
            log("         uid={} {} {}".format(*b))
    else:
        log("  [ok]   {} checks over {} scenarios all validate".format(
            n_checks, len(scenarios)))

    # --- 2. the student really is blind, the teacher really is privileged ---
    sc = scenarios[0]
    bud = {a: ENV_CFG.MAX_TURNS_PER_AGENT for a in sc.agent_ids}
    p_s = prompts.PROFILES["student"].route_prompt(sc, "", [], set(), bud, 1, 12)
    p_t = prompts.PROFILES["teacher"].route_prompt(sc, "", [], set(), bud, 1, 12)
    hidden = [v.get("text", "") for v in sc.private_facts.values() if v.get("text")]
    leak_s = sum(1 for h in hidden if h in p_s)
    seen_t = sum(1 for h in hidden if h in p_t)
    if leak_s or seen_t != len(hidden):
        ok = False
        log("  [FAIL] blindness broken: student sees {}/{} hidden facts, "
            "teacher sees {}/{}".format(leak_s, len(hidden), seen_t, len(hidden)))
    else:
        log("  [ok]   student sees 0/{} hidden facts, teacher sees {}/{}".format(
            len(hidden), seen_t, len(hidden)))

    # --- 3. does enable_thinking actually change the rendered template? ---
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for name, lm in (("router", router), ("agent", agent)):
        try:
            on = lm.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=True)
            off = lm.tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
            if on == off:
                log("  [warn] {}: enable_thinking has NO effect on this template "
                    "(harmless on a non-reasoning model, fatal on Qwen3/Qwen3.5)".format(name))
            else:
                log("  [ok]   {}: enable_thinking changes the template".format(name))
        except TypeError:
            log("  [warn] {}: tokenizer rejects enable_thinking "
                "(pre-Qwen3 template)".format(name))

    # --- 4. one real generation from each, on a real prompt ---
    pick, raw, _ids, _reason = router.ask_json(
        p_s, router.cfg.ROUTE_MAX_TOKENS, "next_agent")
    thinking = "<think>" in (raw or "")
    valid_pick = isinstance(pick, str) and pick in sc.agent_ids
    if thinking or not valid_pick:
        ok = False
        log("  [FAIL] router route: thinking_tokens={}  parsed_next_agent={!r}".format(
            thinking, pick))
        log("         raw: {}".format((raw or "")[:220].replace(chr(10), " ")))
    else:
        log("  [ok]   router route -> {!r}   ({} chars, no thinking tokens)".format(
            pick, len(raw or "")))

    aid = sc.agent_ids[1] if len(sc.agent_ids) > 1 else sc.agent_ids[0]
    a_user = prompts.build_agent_prompt(sc, aid, "", [], set(), ["free", "reveal"], 1, 12)
    _rendered, a_raw = agent.generate(a_user, prompts.SYS_AGENT,
                                      rl.RL_CFG.AGENT_MAX_TOKENS)
    parsed = parse_action(a_raw)
    a_thinking = "<think>" in (a_raw or "")
    if a_thinking or parsed.action is None:
        ok = False
        log("  [FAIL] agent action: thinking_tokens={}  parsed={!r}  err={}".format(
            a_thinking, parsed.action, parsed.parse_error))
        log("         raw: {}".format((a_raw or "")[:220].replace(chr(10), " ")))
    else:
        log("  [ok]   agent action -> {!r}   ({} chars, no thinking tokens)".format(
            parsed.action, len(a_raw or "")))

    # --- 5. the credit span lands on TEXT, not on the action keyword ---
    # This is the check that would have caught train_V1's central bug on minute one.
    span = text_span_chars(a_raw, parsed.action)
    if span is None:
        ok = False
        log("  [FAIL] credit span: could not locate the text span in the agent's "
            "generation")
        log("         raw: {}".format((a_raw or "")[:220].replace(chr(10), " ")))
    else:
        scored = a_raw[span[0]:span[1]]
        if '"action"' in scored:
            ok = False
            log("  [FAIL] credit span covers the action keyword - that span has "
                "measured probability 0.9994 and cannot be trained")
        else:
            log("  [ok]   credit span -> {} chars of text, no action keyword: {!r}".format(
                len(scored), scored[:60] + ("..." if len(scored) > 60 else "")))

    rspan = next_agent_span_chars(raw or "")
    if rspan is None:
        ok = False
        log("  [FAIL] route span: no next_agent value in the router's generation; every "
            "route would be dropped from the RL batch")
        log("         raw: {}".format((raw or "")[:200].replace(chr(10), " ")))
    else:
        log("  [ok]   route span -> {!r}".format((raw or "")[rspan[0]:rspan[1]]))

    # The route output must carry NOTHING but the choice. A `reason` field ahead of it
    # makes the id a deterministic readout of prose the credit span does not cover: the
    # first V2 run measured route_logp = -0.000 over 53 spans, i.e. no policy gradient on
    # the routing decision at all. This is the check that would have caught it in minute
    # one instead of hour three.
    if '"reason"' in (raw or "") or "reason" in (p_s or "").lower().split(
            "[output format]")[-1]:
        ok = False
        log("  [FAIL] the route output still carries a `reason` field. The agent id then "
            "follows from the prose and route_logp collapses to ~0.")
        log("         raw: {}".format((raw or "")[:200].replace(chr(10), " ")))
    else:
        log("  [ok]   route output is choice-only ({} chars) - the decision sits inside "
            "the credit span".format(len(raw or "")))

    # --- 6. batched generation actually batches, and left-pads correctly ---
    # A right-padded decoder-only batch continues from a run of PAD tokens and produces
    # garbage for every short prompt. That failure is silent - the JSON just stops
    # parsing - so it is checked here rather than discovered eight hours in.
    try:
        bs = [p_s, p_s]
        bouts = router.ask_json_batch(bs, router.cfg.ROUTE_MAX_TOKENS, "next_agent")
        ok_b = (len(bouts) == 2
                and all(isinstance(b[0], str) and b[0] in sc.agent_ids for b in bouts))
        if not ok_b:
            ok = False
            log("  [FAIL] batched route: {} of 2 parsed. Check tokenizer padding_side "
                "- a right-padded batch generates from PAD.".format(
                    sum(1 for b in bouts if isinstance(b[0], str)
                        and b[0] in sc.agent_ids)))
            log("         raw[0]: {}".format((bouts[0][1] or "")[:180]
                                             .replace(chr(10), " ")))
        else:
            log("  [ok]   batched route x2 -> {}   (left padding works)".format(
                [b[0] for b in bouts]))
    except Exception as exc:                                    # noqa: BLE001
        ok = False
        log("  [FAIL] router.ask_json_batch raised {}: {}".format(
            type(exc).__name__, str(exc)[:160]))

    try:
        bouts = agent.generate_batch([a_user, a_user], prompts.SYS_AGENT,
                                     rl.RL_CFG.AGENT_MAX_TOKENS)
        n_ok = sum(1 for _r, g in bouts if parse_action(g).action is not None)
        if len(bouts) != 2 or n_ok < 2:
            ok = False
            log("  [FAIL] batched agent: {}/2 parsed as actions".format(n_ok))
            log("         raw[0]: {}".format((bouts[0][1] or "")[:180]
                                             .replace(chr(10), " ")))
        else:
            log("  [ok]   batched agent x2 -> {}   (left padding works)".format(
                [parse_action(g).action for _r, g in bouts]))
    except Exception as exc:                                    # noqa: BLE001
        ok = False
        log("  [FAIL] agent.generate_batch raised {}: {}".format(
            type(exc).__name__, str(exc)[:160]))

    # --- 7. last call is armed ---
    if ENV_CFG.FORCE_DM_WHEN_COMPLETE and ENV_CFG.FORCE_DM_ON_LAST_TURN:
        log("  [ok]   last call armed: complete-record AND final-turn hand the floor "
            "to the decision-maker")
    else:
        log("  [warn] last call is DISABLED. train_V1 lost 104/360 episodes to exactly "
            "this: complete record, 12 turns used, score 0.000.")

    # --- 8. memory headroom, with both models resident ---
    try:
        import torch
        if torch.cuda.is_available():
            g = 1024.0 ** 3
            # per ACTUAL device: the no-arg forms report the current device, which is
            # cuda:0 regardless of where the models were placed - so on cuda:1 they
            # would cheerfully report an empty card while the real one filled up.
            seen = []
            for name, lm in (("router", router), ("agent", agent)):
                d = str(getattr(lm, "device", ""))
                if "cuda" not in d:
                    continue
                if d in seen:
                    log("  [info] {:<7} also on {}".format(name, d))
                    continue
                seen.append(d)
                free, total = torch.cuda.mem_get_info(d)
                log("  [info] {:<7} on {}: {:.1f} GiB allocated | {:.1f} free of "
                    "{:.1f} GiB".format(name, d, torch.cuda.memory_allocated(d) / g,
                                        free / g, total / g))
    except Exception:
        pass

    log("-" * 78)
    if not ok:
        log("PREFLIGHT FAILED." + (" Aborting." if abort else " Continuing anyway."))
        log("  If thinking tokens appeared: check ENABLE_THINKING in opd.OPD_CFG and")
        log("  rl.RL_CFG, or raise ROUTE_MAX_TOKENS / AGENT_MAX_TOKENS above the model's")
        log("  reasoning budget. Do NOT start a long run with this failing.")
    else:
        log("PREFLIGHT PASSED")
    log("")
    return ok
