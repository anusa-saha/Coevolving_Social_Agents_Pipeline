"""prepare_hiddenbench.py - HiddenBench as an OUT-OF-DISTRIBUTION test set.

    python data/prepare_hiddenbench.py

Downloads HiddenBench (65 scenarios) and rewrites it into the scenario schema env.py
already reads, so `inf.py` can evaluate the 1100D checkpoint on it without a single
change to env.py / opd.py / rl.py / main.py / prompts.py.

NOTHING IS TRAINED ON THIS FILE. It is a held-out benchmark the run has never seen, in
any phase, which is the whole point: 1100D's `unseen` split measures held-out DOMAINS
from the same generator, and this measures a held-out BENCHMARK from a different one.
Every scenario is tagged eval_group="unseen" for that reason.

WHY THE MAPPING IS EXACT

HiddenBench is the Stasser hidden-profile paradigm: the shared information is a
deliberate decoy pointing at the wrong option, and the hidden facts only identify the
right one when pooled. Its construction rule - remove any one hidden item and the answer
goes ambiguous - is the same invariant 1100D enforces as `decisive_facts`, which is not a
filter there but is equal to `private_facts` in 1100/1100 scenarios across both splits.
So the fields line up:

    HiddenBench                      1100D
    ---------------------------------------------------------------------------
    hidden_information[i]        ->  private_facts["PF{i+1}"], owner A{i+2}
    shared_information[i]        ->  shared_context["S{i+1}"]
    possible_answers             ->  settlement_schema.decisions.answer (an enum)
    correct_answer               ->  content_checks C1  (exactly one check)
    (implied by construction)    ->  provenance_checks P1..Pn, one per hidden fact
    (implied by construction)    ->  decisive_facts = every private fact
    len(hidden_information) + 1  ->  num_agents, DM = A1 and holds NO private fact

The last row is what makes it faithful. In 1100D the decision-maker owns zero private
facts (1100/1100) and is always agents[0]; giving A1 nothing to reveal preserves the
property that the DM cannot solve the scenario alone. n_hidden 4 -> 5 agents and
n_hidden 3 -> 4 agents both land inside 1100D's own distribution (5-agent: 237,
4-agent: 254 of 720), so the checkpoint is not being asked for a shape it never saw.

Because there is exactly ONE content check, `content_frac` in the eval report IS
HiddenBench accuracy. No new metric, no code change.

TWO PLACES WHERE JUDGEMENT ENTERS, AND WHAT WE CHOSE

 1. ROLES. HiddenBench has no agent roster, so one has to be minted, and the roster is
    visible to the blind router AND to every agent. Minting roles from each holder's
    hidden fact would make routing more legible - the router was trained to route to
    "whoever's role plausibly covers a field the record cannot determine" - but it would
    also (a) leak the hidden information into a public field and (b) make HiddenBench
    strictly easier than it is, because its participants are SYMMETRIC by design: no
    role signals what anyone knows.

    So every agent gets the same role. That is faithful to the benchmark, needs no model
    at prep time, and cannot leak. It is also the harder setting for the trained router,
    which must now route off the record rather than off the roster - worth stating when
    reporting, because it is a floor on the transfer result, not a ceiling. Both arms see
    the identical roster, so the vanilla/co-evolved comparison is unaffected either way.
    CFG.ROLE_FROM_FACTS = True switches to the other convention; the leak gate below
    applies to both and will reject roles that carry hidden vocabulary.

 2. DESCRIPTIONS. HiddenBench descriptions are second-person and carry two mechanics
    this environment does not implement: a $1/$2 payoff for individual/unanimous
    correctness, and a "the chat will last at most 15 minutes, you don't know when it
    ends" clock. The turn budget is already stated to every agent by prompts.py
    ("[TURN] t - about N turn(s) left"), and there is no per-agent payoff here.

    Those lines are STRIPPED, a line left dangling by the strip ("After the
    discussion:" once its bullet list is gone) is dropped with them, and the stale
    participant COUNT is deleted in place - adding a fact-less chair turns a
    4-participant scenario into 5 agents, so "three other participants" would
    contradict the [PARTICIPANTS] roster printed in the same prompt.

    Nothing else is touched - no paraphrase, no voice change, no reordering. Rewriting
    prose would risk changing the task, and the second-person voice reads correctly
    under prompts.build_agent_prompt anyway. Every stripped line and every count edit
    is reported by the self-check so the whole diff is auditable.

The file hard-fails rather than writing a subtly wrong test set. It round-trips through
env.load_scenarios, runs env.validate_check over every emitted check expression, runs
env.TerminalVerifier on a known-good, a known-bad and an unsettled settlement, checks
that no payoff/clock residue survived and that normalisation only ever deleted, checks
that the three scenarios sharing a description stay distinct, and runs a leak gate over
the only text it mints - the roster. Each gate is verified in BOTH directions: a role
carrying a hidden fact verbatim scores 1.000 and one carrying a single distinctive word
scores 0.250, against a 0.15 limit.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import env  # noqa: E402  (after sys.path; env imports no torch at module level)

URL = ("https://huggingface.co/datasets/YuxuanLi1225/HiddenBench/"
       "resolve/main/benchmark.json")
RAW = os.path.join(HERE, "hiddenbench.json")            # vendored: no hub dependency
OUT = os.path.join(HERE, "hiddenbench_ood.json")


class CFG:
    # One role for everyone. See item 1 in the module docstring.
    ROLE = "Panel Member"
    ROLE_FROM_FACTS = False     # ablation hook; the leak gate applies either way

    # Deterministic by index, so a re-run produces a byte-identical file.
    NAMES = ["Avery", "Blair", "Casey", "Devon", "Ellis", "Finley", "Harper", "Indigo"]

    # 1100D's own n_agents -> turn_cap modes: (4, 14) in 177/720, (5, 18) in 174/720,
    # i.e. about 3.5 turns per agent. Anything outside the table falls back to that
    # rule rather than raising, so a benchmark revision with more hidden facts still
    # converts. env.ENV_CFG.T_MAX (12) caps it either way - the smaller of the two wins.
    TURN_CAP = {3: 10, 4: 14, 5: 18, 6: 21}

    # Ceiling on how much of a hidden fact's distinctive vocabulary the minted roster
    # may share. Scored only over the part of the roster that VARIES by scenario - see
    # self-check 7, which explains why the obvious formulation is circular.
    LEAK_MAX = 0.15

    # Mechanics this environment does not implement (see item 2).
    STRIP_PAYOFF = True
    STRIP_CLOCK = True
    # Adding a fact-less chair changes the head count the prose quotes.
    FIX_PARTICIPANT_COUNT = True


# Published hidden-profile studies. Their answers are plausibly in any base model's
# pretraining, so they are reported as their own domain rather than pooled: if the
# vanilla arm already solves them, those five measure recall, not information pooling.
LIT_REPLICATIONS = {
    "toma_butera_2009", "baker_2010", "schulz_hardt_mojzisch_2012",
    "graetz_et_al_1998", "Stasser_Stewart_1992",
}

# First match wins. `domain` drives inf.py's per-domain tables only; it is descriptive.
FAMILY_RULES = [
    ("deduction_mystery", r"theft|murder|missing|mystery|outbreak|investigat|stolen|"
                          r"culprit|elusive|deduction|who\s+killed"),
    ("evacuation_shelter", r"evacuat|shelter|safe\s*haven|refuge|storm|disaster|"
                           r"flood|earthquake|spill|hurricane|blackout|outage"),
    ("logistics_delivery", r"delivery|deliver|transport|route|supply|supplies|drop|"
                           r"shipment|move|moving|migration|relocat|transfer"),
    ("site_selection", r"."),      # the remainder: venue / base / station / site choice
]

# The benchmark uses a curly U+2019 apostrophe, not an ASCII one. A bare `don'?t` misses
# every contraction in the file, which is how "You don't know how much time you have
# left" survived a first pass of this script.
_APOS = r"['’ʼ]"

PAYOFF_RE = re.compile(
    r"(earn\s+\$|\$\d|additional\s+\$|maximi[sz]e\s+your\s+reward|"
    r"you\s+each\s+earn|for\s+a\s+total\s+of)", re.I)
CLOCK_RE = re.compile(
    r"(\b\d+\s*minutes?\b|when\s+the\s+(?:chat|discussion)\s+will\s+end|"
    r"exact\s+time\s+when|do(?:es)?n" + _APOS + r"?t\s+know\s+(?:when|how\s+much\s+time)|"
    r"speed\s+matters)", re.I)

# Sentence boundary. Removal is per sentence, not per line - see normalise_description.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# The participant COUNT in the prose is stale once A1 is added as a fact-less chair:
# a 4-participant HiddenBench scenario becomes 5 agents here, so "three other
# participants" would contradict the [PARTICIPANTS] roster the same prompt prints.
# The count word is deleted ("the other three community leaders" -> "the other community
# leaders"); nothing else in the sentence is touched, so no task content is lost. The
# lookahead is deliberately narrow - it fires only before a person noun, so "three
# options", "three suspects" and "12 people fell ill" are all left alone.
_PERSON = (r"(?:participants?|leaders?|members?|investigators?|colleagues?|panelists?|"
           r"employees?|experts?|scientists?|officers?|staff)")
_COUNT = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
COUNT_RE = re.compile(
    r"\b" + _COUNT + r"\s+(?=(?:other\s+)?(?:\w+\s+){0,2}" + _PERSON + r"\b)", re.I)

# A line left dangling by the strip - "After the discussion:" with its bullet list gone.
DANGLING_RE = re.compile(r":\s*$")


# ============================================================
# SOURCE
# ============================================================

def fetch():
    """The raw benchmark, downloaded once and vendored into data/.

    A plain URL read, not `datasets.load_dataset`: this adds no huggingface_hub or
    pyarrow dependency to a repo that otherwise needs only torch/transformers/peft, and
    it pins the artefact - a later re-run reads the committed file rather than whatever
    the hub is serving that day.
    """
    if not os.path.isfile(RAW):
        print("[fetch] {} -> {}".format(URL, os.path.relpath(RAW, ROOT)))
        with urllib.request.urlopen(URL, timeout=120) as r:
            body = r.read()
        with open(RAW, "wb") as f:
            f.write(body)
    with open(RAW, encoding="utf-8-sig") as f:
        raw = json.load(f)
    print("[fetch] {} scenarios from {} ({:,} bytes)".format(
        len(raw), os.path.relpath(RAW, ROOT), os.path.getsize(RAW)))
    return raw


# ============================================================
# TRANSFORMS
# ============================================================

def normalise_description(desc):
    """Strip the payoff and clock mechanics, and de-stale the participant count.

    Returns (text, [dropped sentences], n_count_fixes). Removal is per SENTENCE, not per
    line: the clock phrase is often welded to real task content -

        "Now, the rain has temporarily stopped, giving you and the other community
         leaders a short window to decide on the safest evacuation route before the rain
         resumes. You don't know how much time you have left to make this decision."

    - and dropping that whole line would take the task statement with it. Nothing is
    paraphrased, reordered or reworded; surviving sentences are the benchmark's own.
    """
    kept, dropped = [], []
    for line in desc.split("\n"):
        if not line.strip():
            kept.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        sents = _SENT_SPLIT.split(line.strip())
        keep = []
        for s in sents:
            if (CFG.STRIP_PAYOFF and PAYOFF_RE.search(s)) or \
               (CFG.STRIP_CLOCK and CLOCK_RE.search(s)):
                dropped.append(s.strip())
                continue
            keep.append(s)
        if keep:                                   # else the line was pure mechanics
            kept.append(indent + " ".join(keep))

    # Drop a line whose list was just stripped out from under it ("After the
    # discussion:"), i.e. one ending in a colon with nothing left below it.
    while kept and (not kept[-1].strip() or DANGLING_RE.search(kept[-1])):
        if DANGLING_RE.search(kept[-1]):
            dropped.append(kept[-1].strip())
        kept.pop()

    text = "\n".join(kept)
    text, n_fix = (COUNT_RE.subn("", text) if CFG.FIX_PARTICIPANT_COUNT else (text, 0))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, dropped, n_fix


def family(rec):
    if rec["name"] in LIT_REPLICATIONS:
        return "literature_replication"
    hay = "{} {}".format(rec["name"], rec["description"]).lower()
    for name, pat in FAMILY_RULES:
        if re.search(pat, hay):
            return name
    return "site_selection"


def answer_literals(answer):
    """The answer string plus punctuation variants, as Python literals for the check.

Covers curly-vs-straight punctuation and letter case.

    A model that answers "west city" has not made a reasoning error, and with exactly ONE
    content check a single case slip costs a whole scenario. 1100D already accepts format
    alternatives for a correct value - `closing_date == '2026-12-15' or closing_date ==
    'December 15, 2026'` - so alternatives are the house convention, not a new leniency,
    and the expression stays inside env.validate_check's whitelist (there is no .lower()
    available to a check). Both arms are scored identically, so this cannot favour either.

    Only one option in the benchmark carries a non-ASCII character (a U+2019 apostrophe)
    and no correct_answer does, so the punctuation half is insurance rather than a fix.
    """
    def _ascii(x):
        return (x.replace("’", "'").replace("‘", "'")
                 .replace("“", '"').replace("”", '"')
                 .replace("–", "-").replace("—", "-"))

    out = []
    for base in (answer, _ascii(answer)):
        for v in (base, base.lower(), base.upper()):
            if v not in out:
                out.append(v)
    return out


def content_check(answer):
    return " or ".join("decisions['answer'] == {!r}".format(v)
                       for v in answer_literals(answer))


def roles_for(rec, agent_ids):
    """One role string per agent. Symmetric by default - see item 1 in the docstring."""
    if not CFG.ROLE_FROM_FACTS:
        return {a: CFG.ROLE for a in agent_ids}
    # Ablation path: name the holder after the first capitalised noun phrase in the fact
    # it owns. Deliberately crude - the leak gate is what decides whether it is usable.
    out = {agent_ids[0]: CFG.ROLE}
    for k, fact in enumerate(rec["hidden_information"]):
        m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", fact)
        out[agent_ids[k + 1]] = "{} Liaison".format(m.group(1)) if m else CFG.ROLE
    return out


def convert(rec, uid):
    hidden = rec["hidden_information"]
    n_agents = len(hidden) + 1                       # DM holds nothing: 1100D convention
    agent_ids = ["A{}".format(k + 1) for k in range(n_agents)]
    roles = roles_for(rec, agent_ids)
    desc, dropped, n_fix = normalise_description(rec["description"])

    private = {"PF{}".format(k + 1): {"owner": agent_ids[k + 1], "text": text}
               for k, text in enumerate(hidden)}
    shared = {"S{}".format(k + 1): text
              for k, text in enumerate(rec["shared_information"])}

    # Who may see what. The DM sees the shared context and nothing else - that is the
    # property that forces elicitation.
    views = {agent_ids[0]: sorted(shared)}
    for k in range(len(hidden)):
        views[agent_ids[k + 1]] = sorted(shared) + ["PF{}".format(k + 1)]

    scen = {
        "uid": uid,
        "scenario_id": rec["name"],
        "scenario_type": "hidden_profile_choice",
        "description": desc,
        "agents": [{"agent_id": a,
                    "name": CFG.NAMES[k % len(CFG.NAMES)],
                    "role": roles[a]} for k, a in enumerate(agent_ids)],
        "shared_context": shared,
        "private_facts": private,
        "views": views,
        "decision_maker": agent_ids[0],
        "interaction_config": {"turn_order": agent_ids,
                               "turn_cap": CFG.TURN_CAP.get(
                                   n_agents, int(round(3.5 * n_agents)))},
        "settlement_schema": {
            "decisions": {"answer": "one of: " + " | ".join(rec["possible_answers"])},
            "credited_facts": ["fact_id"],
            "justification_fact_ids": ["fact_id"],
        },
        "acceptance_conditions": [
            "The answer must be exactly {!r}.".format(rec["correct_answer"]),
            "Every hidden fact must be formally revealed and cited in the settlement.",
        ],
        "content_checks": {"C1": content_check(rec["correct_answer"])},
        "provenance_checks": {
            "P{}".format(k + 1):
                "{0!r} in justification_fact_ids and {0!r} in revealed".format(f)
            for k, f in enumerate(private)},
        "decisive_facts": [
            {"fact_id": f, "owner": v["owner"], "flips": ["C1"],
             "why": "HiddenBench construction: without this fact the remaining record "
                    "does not uniquely determine the answer."}
            for f, v in private.items()],
        "domain": family(rec),
        "num_agents": n_agents,
        "eval_group": "unseen",          # OOD by definition; never trained on
        # provenance of the row itself, and what the poll/native metric would need
        "hb_id": rec["id"],
        "possible_answers": rec["possible_answers"],
        "correct_answer": rec["correct_answer"],
        "hb_rationale": rec.get("rationale") or "",
        "dropped_lines": dropped,
        "count_fixes": n_fix,
    }
    return scen


# ============================================================
# SELF-CHECK  --  hard-fails instead of writing a wrong test set
# ============================================================

def fail(msg):
    raise SystemExit("[FAIL] {}".format(msg))


def selfcheck(raw, rows):
    print("\n" + "=" * 74)
    print("SELF-CHECK")
    print("=" * 74)

    # ---- 1. source integrity ----
    for rec in raw:
        for k in ("name", "description", "shared_information", "hidden_information",
                  "possible_answers", "correct_answer"):
            if k not in rec:
                fail("source record {!r} is missing {!r}".format(rec.get("name"), k))
        if rec["correct_answer"] not in rec["possible_answers"]:
            fail("{}: correct_answer not among possible_answers".format(rec["name"]))
        if len(rec["hidden_information"]) < 2:
            fail("{}: fewer than 2 hidden facts".format(rec["name"]))
    print("  source records ................ {} ok".format(len(raw)))

    # ---- 2. uid uniqueness and tagging ----
    uids = [r["uid"] for r in rows]
    if len(set(uids)) != len(uids):
        fail("duplicate uid in the converted set")
    if any(r["eval_group"] != "unseen" for r in rows):
        fail("every converted scenario must be tagged eval_group='unseen'")
    print("  uids unique, all eval_group=unseen")

    # ---- 3. round-trip through the real loader ----
    tmp = OUT + ".check"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    try:
        scens = env.load_scenarios(tmp)
    finally:
        os.remove(tmp)
    if len(scens) != len(rows):
        fail("env.load_scenarios dropped {} scenario(s) as invalid".format(
            len(rows) - len(scens)))
    print("  env.load_scenarios ............ {}/{} loaded, 0 dropped".format(
        len(scens), len(rows)))

    # ---- 4. structural invariants 1100D enforces ----
    for s in scens:
        owners = {v["owner"] for v in s.private_facts.values()}
        if s.decision_maker in owners:
            fail("{}: decision-maker owns a private fact".format(s.scenario_id))
        if s.decision_maker != s.agent_ids[0]:
            fail("{}: decision-maker is not agents[0]".format(s.scenario_id))
        if set(s.decisive_fact_ids()) != set(s.private_facts):
            fail("{}: decisive_facts != private_facts".format(s.scenario_id))
        if len(s.private_facts) != s.num_agents - 1:
            fail("{}: n_private != n_agents - 1".format(s.scenario_id))
    print("  DM fact-less / DM==A1 / decisive==private ... {} ok".format(len(scens)))

    # ---- 5. every check expression survives the AST whitelist ----
    n_expr = 0
    for s in scens:
        for cid, expr in list(s.content_checks.items()) + list(s.provenance_checks.items()):
            try:
                env.validate_check(expr)
            except Exception as exc:                            # noqa: BLE001
                fail("{} {}: {}".format(s.scenario_id, cid, exc))
            n_expr += 1
    print("  env.validate_check ............ {} expressions, 0 rejected".format(n_expr))

    # ---- 6. the verifier actually separates right from wrong ----
    # env.Scenario keeps only its own dataclass fields, so correct_answer and
    # possible_answers do NOT survive load_scenarios - they are carried for provenance
    # and for a consensus poll, not for the verifier. Read them from the row instead.
    row_of = {r["uid"]: r for r in rows}
    for s in scens:
        row = row_of[s.uid]
        allf = sorted(s.private_facts)
        good = {"decisions": {"answer": row["correct_answer"]},
                "credited_facts": allf, "justification_fact_ids": allf}
        wrong = dict(good, decisions={
            "answer": next(o for o in row["possible_answers"]
                           if o != row["correct_answer"])})
        c, p = env.TerminalVerifier(s).verify(good, set(allf))
        if not (all(c.values()) and all(p.values())):
            fail("{}: a perfect settlement does not pass its own checks".format(
                s.scenario_id))
        c2, _ = env.TerminalVerifier(s).verify(wrong, set(allf))
        if any(c2.values()):
            fail("{}: a wrong answer still passes the content check".format(
                s.scenario_id))
        c3, p3 = env.TerminalVerifier(s).verify(None, set())
        if c3 or p3:
            fail("{}: an unsettled episode graded non-empty".format(s.scenario_id))
    print("  TerminalVerifier .............. right/wrong/unsettled separated on {}"
          .format(len(scens)))

    # ---- 7. leak gate: does anything WE MINTED carry hidden vocabulary? ----
    #
    # env.private_leak cannot be pointed at the scenario's own public text, and an
    # earlier version of this check did exactly that and was vacuous. It scores a
    # CANDIDATE INSIGHT against a FIXED public context, computing each fact's unique
    # vocabulary as `_toks(fact) - _public_vocab(scen)` - and `_public_vocab` includes
    # the agent roster. Put a hidden fact verbatim into a role and that vocabulary joins
    # the public set, is subtracted from the fact, drops it under the 3-token floor, and
    # the fact is skipped: the leak erases its own evidence. Verified - a verbatim paste
    # scored 0.000 and the "blind" count rose from 8 to 143.
    #
    # The non-circular test fixes the reference to the BENCHMARK's own public text
    # (description + shared information + option names, none of which we author) and
    # probes only the text this script mints: the roster. The normalised description
    # cannot add vocabulary because normalisation only deletes.
    #
    # Only the part of the roster that VARIES BY SCENARIO can carry information about
    # that scenario's facts. Vocabulary present in every roster - the fixed name pool,
    # and the role string itself when roles are symmetric - is constant and encodes
    # nothing, so scoring it only produces false positives: the default roster hit
    # overlap 0.125 against two facts purely because "Panel Member" shares the word
    # "member" with them. The constant vocabulary is therefore subtracted, which makes
    # the gate exactly zero for a symmetric roster (correct - it cannot leak) while
    # leaving the ROLE_FROM_FACTS path fully exposed, since those roles vary.
    minted_of = {
        s.uid: env._toks(" ".join("{} {}".format(a.get("name", ""), a.get("role", ""))
                                  for a in s.agents))
        for s in scens}
    constant = (set.intersection(*minted_of.values())
                if len(minted_of) > 1 else set())

    worst, blind = (0.0, "", ""), 0
    for s in scens:
        rec = raw[s.uid]
        base = env._toks(" ".join([rec["description"],
                                   " ".join(rec["shared_information"]),
                                   " ".join(rec["possible_answers"])])) | env._PROC_STOP
        minted = minted_of[s.uid] - constant
        for fid, v in s.private_facts.items():
            uniq = env._toks(v.get("text", "")) - base
            if len(uniq) < 3:
                blind += 1
                continue
            lk = len(uniq & minted) / len(uniq)
            if lk > worst[0]:
                worst = (lk, s.scenario_id, fid)
            if lk > CFG.LEAK_MAX:
                fail("{}: the minted roster leaks {} (overlap {:.3f} > {:.2f}) - "
                     "shared vocabulary: {}".format(
                         s.scenario_id, fid, lk, CFG.LEAK_MAX,
                         ", ".join(sorted(uniq & minted)[:8])))

    # Normalisation must only ever delete, never introduce.
    for r in rows:
        added = env._toks(r["description"]) - env._toks(raw[r["uid"]]["description"])
        if added:
            fail("{}: normalisation INVENTED vocabulary: {}".format(
                r["scenario_id"], sorted(added)[:8]))

    print("  roster leak gate .............. max {:.3f} ({} / {}), limit {:.2f}".format(
        worst[0], worst[1] or "-", worst[2] or "-", CFG.LEAK_MAX))
    print("  normalisation only deletes .... {} descriptions ok".format(len(rows)))
    if blind:
        print("     note: {} of {} fact(s) have <3 tokens of vocabulary not already in "
              "the benchmark's own public text".format(
                  blind, sum(len(s.private_facts) for s in scens)))

    # ---- 8. no mechanics survived the normalisation ----
    # These fired on a first pass: a curly U+2019 apostrophe defeated `don'?t`, and a
    # line-level strip left "After the discussion:" dangling over a deleted list. Both
    # are cheap to reintroduce and invisible in aggregate metrics, so they are asserted.
    for r in rows:
        d = r["description"]
        for label, rx in (("payoff", PAYOFF_RE), ("clock", CLOCK_RE)):
            m = rx.search(d)
            if m:
                fail("{}: {} mechanics survived normalisation: ...{}...".format(
                    r["scenario_id"], label, d[max(0, m.start() - 40):m.end() + 40]))
        if d.rstrip().endswith(":"):
            fail("{}: description ends on a dangling ':' - a stripped list left its "
                 "lead-in behind".format(r["scenario_id"]))
        if len(d.strip()) < 80:
            fail("{}: description is {} chars after normalisation - too much was "
                 "removed".format(r["scenario_id"], len(d.strip())))
    print("  no payoff / clock / dangling residue .. {} descriptions ok".format(len(rows)))

    # ---- 9. scenarios sharing a description must still be distinct ----
    # evacuation_west_city / _north_hill / _east_town are byte-identical cover stories
    # whose hidden information differs, so each has a DIFFERENT correct answer. That is
    # a controlled triplet - it separates reading the record from pattern-matching the
    # prose - and it breaks any analysis keyed on the description. Assert the conversion
    # keeps them apart.
    by_desc = collections.defaultdict(list)
    for r in rows:
        by_desc[r["description"]].append(r)
    twins = {k: v for k, v in by_desc.items() if len(v) > 1}
    for group in twins.values():
        ids = [r["scenario_id"] for r in group]
        if len({r["uid"] for r in group}) != len(group):
            fail("{}: share a description AND a uid".format(ids))
        if len({r["correct_answer"] for r in group}) == 1:
            fail("{}: share a description and the same answer - not a real twin "
                 "set".format(ids))
        if len({json.dumps(r["private_facts"], sort_keys=True) for r in group}) != len(group):
            fail("{}: share a description and identical hidden facts".format(ids))
        if len({json.dumps(r["content_checks"], sort_keys=True) for r in group}) != len(group):
            fail("{}: share a description and the same content check".format(ids))
    print("  shared-description twins ...... {} group(s), all distinct by uid / facts / "
          "answer".format(len(twins)))
    for group in twins.values():
        print("     {} -> {}".format(
            " / ".join(r["scenario_id"] for r in group),
            ", ".join(r["correct_answer"] for r in group)))


def report(rows):
    print("\n" + "=" * 74)
    print("CONVERTED SET")
    print("=" * 74)
    fam = collections.Counter(r["domain"] for r in rows)
    ag = collections.Counter(r["num_agents"] for r in rows)
    ans = collections.Counter(len(r["possible_answers"]) for r in rows)
    drop = sum(len(r["dropped_lines"]) for r in rows)
    print("  scenarios ..... {}".format(len(rows)))
    print("  n_agents ...... {}".format(dict(sorted(ag.items()))))
    print("  n_options ..... {}".format(dict(sorted(ans.items()))))
    npv = sorted({len(r["provenance_checks"]) for r in rows})
    print("  checks ........ 1 content + {} provenance (mean {:.1f} total)".format(
        "-".join(str(n) for n in (npv if len(npv) < 3 else [npv[0], npv[-1]])),
        sum(1 + len(r["provenance_checks"]) for r in rows) / len(rows)))
    print("  domains (for the per-domain tables):")
    for k, v in fam.most_common():
        print("      {:<26s} {}".format(k, v))
    fixes = sum(r["count_fixes"] for r in rows)
    print("  sentences stripped: {} (payoff / clock mechanics)".format(drop))
    print("  participant counts de-staled: {} (a fact-less chair adds one agent)"
          .format(fixes))
    ex = next((r for r in rows if r["dropped_lines"]), None)
    if ex:
        print("      e.g. {}:".format(ex["scenario_id"]))
        for line in ex["dropped_lines"][:3]:
            print("         - {}".format(line[:96]))


def main():
    raw = fetch()
    rows = [convert(rec, i) for i, rec in enumerate(raw)]
    selfcheck(raw, rows)
    report(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print("\n  wrote {}  ({:,} bytes)".format(
        os.path.relpath(OUT, ROOT), os.path.getsize(OUT)))
    print("\n  evaluate it with:   python run_hb_eval.py")


if __name__ == "__main__":
    main()
