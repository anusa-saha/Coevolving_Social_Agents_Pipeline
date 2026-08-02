In each scenario, a group of people must make a decision. The information needed to decide correctly is split across the
group thus no single person has it all. You only write the scenario. You never write conversations, transcripts, or grades.

## The two tests every scenario must pass

- Alone: one person (the decision-maker) sees only the shared facts and their own private facts. No conversation.
  They decide right away. They should usually get it wrong and not satisfy all the people involved in the scenario.
- Together: everyone sees the shared facts, their own private facts, and everything said so far. They can talk and
  share what they know. They should usually get it right and the final decision satisfies everyone.

## Rules for a good scenario

- Pick a scenario where the group of people should make a concrete decision 
- Split the needed information across the group. Each non-decision-maker knows at least one private fact that changes
  the correct decision, once they say it and it gets used.
- There should be only one final decision maker.
- Never write a sentence that hints that some information is hidden or missing. It must read like an ordinary situation, not a
  puzzle.
- Works for any number of people. Don't hardcode a fixed count anywhere.

## Shared facts

Shared facts (`S1`, `S2`, ...) are things everyone already knows. Write them like plain facts a person in the room would
know, there should never be a sentence that describes the scenario's own structure.

Include one shared fact that gives everyone a normal reason to speak before the decision is final. Write it as an ordinary process step to prompt the group of people to have a conversation and share information

## Private facts

Private facts (`PF1`, `PF2`, ...) each belong to one agent, marked with an `"owner"` field. Every private fact must
change something concrete about the final decision — a number, an assignment, a date, a required commitment, etc. If it has no
effect on the final decision, cut it.

### Decisive facts

A decisive fact is a private fact that flips the outcome: some check fails without it, and can pass once it's revealed
and used. List these in `decisive_facts`. Each entry has:

- `fact_id`, `owner`
- `flips`: which check ID(s) it turns from fail to pass
- `why`: one sentence naming exactly what changes (a number, a choice) — not "this fact matters"

Every non-decision-maker must own at least one decisive fact.

Make sure the decision-maker can't just pass by being generous, fair, or by guessing a stereotype. 
The private facts need to be specific enough that generic good behavior isn't enough on its
own.

## Views

Each agent's view = every shared fact + only their own private facts. Never let one agent's view leak another agent's
private fact. The decision-maker's view alone defines the "alone" test.

## Interaction setup

Set `decision_maker`, `turn_order`, and `turn_cap` (about 3-4 turns per agent — enough for everyone to speak plus one to
settle). Don't hardcode the turn cap to a fixed number of agents.

## Checks

Turn every acceptance condition into a check — a boolean expression, not a sentence.

- Content check: can be verified by reading the settlement JSON alone.
- Provenance check: additionally needs proof the fact was said out loud by its owner AND cited by the
  decision-maker. Rule of thumb: crediting a *public* fact → content check. Crediting a *private* fact out loud →
  provenance check.

Write at least 4 `content_checks` and at least 1 `provenance_check`.

If a private fact implies a number or threshold, compile it exactly as stated — never round it down to make the scenario
easier.

### Every single content check and provenance check must be valid Python that returns True or False

Each check is one line of Python. It is evaluated with exactly these five variables available, and nothing else:

- `decisions` — the decision object (shaped like `settlement_schema`)
- `credited_facts` — fact IDs the decision-maker says they credited
- `commitments` — list of commitment objects the decision-maker made
- `justification_fact_ids` — fact IDs the decision-maker cited as their reasoning
- `revealed` — fact IDs actually said out loud during the conversation

The check must be a single expression that evaluates to `True` or `False` — a boolean. Not a string, not a number, not
`None`.

Allowed: the five names above, plain literals, `==` `!=` `<` `<=` `>` `>=`, `in` / `not in`, `and` / `or` / `not`,
`any()` / `all()` / `len()` / `sum()`, and the dict methods `.values()` / `.keys()` / `.items()` / `.get()`. Not
allowed: `if/else` ternaries, imports, lambdas, any other method call, or any name besides the five above (plus a
comprehension's own loop variable, e.g. the `c` in `any(c['type'] == 'x' for c in commitments)`).

## Before you return it, check your own work

- Solvable: at least one settlement passes every check.
- Failable: at least one plausible settlement fails at least one check.
- Not trivial: a generic fair/equal/cautious settlement does not pass everything.
- Faithful: no threshold from a private fact was softened.
- No leaks: shared facts never hint at hidden information.
- Decisive facts complete: one per non-decision-maker, each `why` names a real change.
- Enough checks: at least 4 `content_checks`, at least 1 `provenance_check`.
- Valid Python, returns a boolean: every check only uses the five allowed names, only the allowed
  operators/functions/methods, every string is quoted, and it evaluates to `True` or `False`.

## Output format

Return JSON only — no Markdown, no preamble, no explanation.

```json
{
    "scenario_id": "...",
    "scenario_type": "...",
    "description": "...",
    "agents": [],
    "shared_context": {},
    "private_facts": {},
    "views": {},
    "decision_maker": "...",
    "interaction_config": { "turn_order": [], "turn_cap": },
    "settlement_schema": {},
    "acceptance_conditions": [],
    "content_checks": {},
    "provenance_checks": {},
    "decisive_facts": []
}
```

## Few-shot examples

### Example 1 — Workplace bonus allocation

```json
{
  "scenario_id": "seed_bonus_001",
  "scenario_type": "resource_allocation",
  "description": "A team lead must divide a project bonus among three team members after a successful product launch, with a finance partner also weighing in.",
  "agents": [
    {"agent_id": "A1", "name": "Maya", "role": "Team Lead"},
    {"agent_id": "A2", "name": "Raj", "role": "Senior Engineer"},
    {"agent_id": "A3", "name": "Lin", "role": "Junior Engineer"},
    {"agent_id": "A4", "name": "Priya", "role": "Finance Partner"}
  ],
  "shared_context": {
    "S1": "The team successfully launched the product and received a $30,000 bonus.",
    "S2": "Maya determines the final bonus allocation.",
    "S3": "Raj delivered the live demonstration that convinced leadership.",
    "S4": "Lin is junior and was relatively quiet during project meetings.",
    "S5": "Before the allocation is finalized, each team member normally explains what work they believe the decision should recognize."
  },
  "private_facts": {
    "PF1": {"owner": "A2", "text": "Raj believes his contribution warrants at least half of the bonus."},
    "PF2": {"owner": "A3", "text": "Lin independently fixed a launch-critical database deadlock shortly before deployment."},
    "PF3": {"owner": "A3", "text": "Lin currently holds a competing job offer."},
    "PF4": {"owner": "A4", "text": "Finance approved an extra $5,000 top-up to the pool, but only if a written retention commitment is filed for any flight-risk team member."}
  },
  "views": {
    "A1": ["S1","S2","S3","S4","S5"],
    "A2": ["S1","S2","S3","S4","S5","PF1"],
    "A3": ["S1","S2","S3","S4","S5","PF2","PF3"],
    "A4": ["S1","S2","S3","S4","S5","PF4"]
  },
  "decision_maker": "A1",
  "interaction_config": {"turn_order": ["A1","A2","A3","A4"], "turn_cap": 14},
  "settlement_schema": {
    "decisions": {"allocations": {"A1": "integer", "A2": "integer", "A3": "integer"}},
    "credited_facts": ["fact_id"],
    "commitments": [{"type": "string", "target": "agent_id", "detail": "string"}],
    "justification_fact_ids": ["fact_id"]
  },
  "acceptance_conditions": [
    "The full available pool must be allocated.",
    "A2 must receive at least $15,000.",
    "A3 must receive at least $7,500.",
    "A2's publicly visible demo contribution S3 must be credited.",
    "A3's hidden database contribution PF2 must be explicitly acknowledged.",
    "A retention commitment targeting A3 must exist.",
    "The pool used must reflect the $5,000 top-up (i.e. sum to $35,000) only if the retention commitment is present."
  ],
  "content_checks": {
    "C1": "sum(decisions['allocations'].values()) in (30000, 35000)",
    "C2": "decisions['allocations']['A2'] >= 15000",
    "C3": "decisions['allocations']['A3'] >= 7500",
    "C4": "'S3' in credited_facts",
    "C5": "any(c['type'] == 'retention' and c['target'] == 'A3' for c in commitments)",
    "C6": "(any(c['type'] == 'retention' for c in commitments) and sum(decisions['allocations'].values()) == 35000) or (not any(c['type'] == 'retention' for c in commitments) and sum(decisions['allocations'].values()) == 30000)"
  },
  "provenance_checks": {
    "P1": "'PF2' in credited_facts and 'PF2' in justification_fact_ids and 'PF2' in revealed"
  },
  "decisive_facts": [
    {"fact_id": "PF1", "owner": "A2", "flips": ["C2"], "why": "Maya has no independent way to know Raj privately expects at least half; without it, a generic visible-merit split can land near $12,000, failing C2. Once revealed, the allocation to A2 moves up to clear the $15,000 floor."},
    {"fact_id": "PF2", "owner": "A3", "flips": ["C5", "P1"], "why": "Without PF2 there is no reason to credit a database fix; the generic 'reward visible demo work' settlement fails C5 and P1. Revealed and cited, both become satisfiable."},
    {"fact_id": "PF3", "owner": "A3", "flips": ["C5"], "why": "PF3 supplies the concrete leaving-risk that makes a retention commitment necessary rather than optional; without it, C5 has no motivating reason even if PF2 alone is known."},
    {"fact_id": "PF4", "owner": "A4", "flips": ["C6"], "why": "Nothing tells Maya a $5,000 top-up exists; without PF4, allocations sum to $30,000 and adding a retention commitment on top would fail C6. Revealed, the pool correctly grows to $35,000 alongside the commitment."}
  ]
}
```

### Example 2 — Procurement / vendor selection

```json
{
  "scenario_id": "seed_procurement_security_001",
  "scenario_type": "procurement",
  "description": "A procurement lead must select a vendor for a business-critical analytics deployment, balancing cost, security posture, budget ceiling, and a compliance deadline.",
  "agents": [
    {"agent_id": "A1", "name": "Grace", "role": "Procurement Lead"},
    {"agent_id": "A2", "name": "Leo", "role": "Security Engineer"},
    {"agent_id": "A3", "name": "Nina", "role": "Finance Manager"},
    {"agent_id": "A4", "name": "Priti", "role": "Operations Lead"}
  ],
  "shared_context": {
    "S1": "Vendor X costs $82,000 and Vendor Y costs $91,000.",
    "S2": "Both vendors satisfy the published functional requirements.",
    "S3": "Vendor X has the shorter standard implementation timeline.",
    "S4": "Grace makes the final vendor selection.",
    "S5": "The deployment supports a compliance program with a fiscal Q3 close of September 30.",
    "S6": "Before finalizing, Grace requests a sign-off comment from Leo, Nina, and Priti."
  },
  "private_facts": {
    "PF1": {"owner": "A2", "text": "Vendor X failed an internal security review because its audit-log retention cannot meet company policy."},
    "PF2": {"owner": "A3", "text": "Finance has an approved procurement ceiling of $97,000 for this deployment."},
    "PF3": {"owner": "A4", "text": "Vendor Y's standard implementation timeline would miss the September 30 compliance close by two weeks unless a $3,000 expedited-onboarding option is purchased."}
  },
  "views": {
    "A1": ["S1","S2","S3","S4","S5","S6"],
    "A2": ["S1","S2","S3","S4","S5","S6","PF1"],
    "A3": ["S1","S2","S3","S4","S5","S6","PF2"],
    "A4": ["S1","S2","S3","S4","S5","S6","PF3"]
  },
  "decision_maker": "A1",
  "interaction_config": {"turn_order": ["A1","A2","A3","A4"], "turn_cap": 14},
  "settlement_schema": {
    "decisions": {"selected_vendor": "string", "approved_budget": "integer"},
    "credited_facts": [],
    "commitments": [{"type": "string", "target": "string", "detail": "string"}],
    "justification_fact_ids": ["fact_id"]
  },
  "acceptance_conditions": [
    "Vendor Y must be selected, because Vendor X failed the hidden security review.",
    "Approved spending must not exceed the finance ceiling of $97,000.",
    "Approved spending must cover at least Vendor Y's base cost plus the expedited-onboarding fee ($94,000).",
    "An expedite commitment targeting Vendor Y must exist so the compliance deadline is met."
  ],
  "content_checks": {
    "C1": "decisions['selected_vendor'] == 'Vendor Y'",
    "C2": "decisions['approved_budget'] <= 97000",
    "C3": "decisions['approved_budget'] >= 94000",
    "C4": "any(c['type'] == 'expedite' and c['target'] == 'Vendor Y' for c in commitments)"
  },
  "provenance_checks": {
    "P1": "'PF1' in justification_fact_ids and 'PF1' in revealed",
    "P2": "'PF3' in justification_fact_ids and 'PF3' in revealed"
  },
  "decisive_facts": [
    {"fact_id": "PF1", "owner": "A2", "flips": ["C1", "P1"], "why": "Publicly Vendor X looks preferable (cheaper, faster). Only PF1 (X's hidden security failure) makes selecting Vendor Y the correct, checkable answer instead of an arbitrary or cost-driven guess."},
    {"fact_id": "PF2", "owner": "A3", "flips": ["C2"], "why": "Nothing public caps the budget; without PF2, an approved_budget above $97,000 (e.g. padded contingency) would not be recognized as invalid. Revealed, spending is correctly capped."},
    {"fact_id": "PF3", "owner": "A4", "flips": ["C4", "P2", "C3"], "why": "Nothing public ties Vendor Y to a timeline risk; only PF3 makes the expedite commitment (and the extra $3,000 in the approved budget) necessary rather than optional padding."}
  ]
}
```

---

## Feedback prompt (used when a scenario is rejected)

After you submit a scenario, it gets tested. If it fails, you'll get a message like this, plus your full
scenario JSON. Revise the same scenario — don't start over unless told to.

```
Your previous scenario was REJECTED. Tag: {REJECT_TAG}

What went wrong: {DIAGNOSIS}

Evidence: {EVIDENCE}

What to fix: {FIX_INSTRUCTIONS}

Rules for your revision:
- Keep everything that already works. Change only what caused the rejection.
- Return the complete scenario object again, in the same JSON format as before.
- Return JSON only — no explanation.
```

The three rejection tags:

- MALFORMED
  - What it means: a check is broken — impossible to pass, always passes, or `shared_context`
    accidentally leaks the setup.
  - How to fix it: rewrite the leaking sentence, or rewrite the check so it's genuinely conditional.

- LEAKED
  - What it means: the lone decision-maker passed too often. The hidden information wasn't needed.
  - How to fix it: tighten the checks back to exactly what the private fact implies. Make sure no
    generic, fair, or stereotype-based settlement can pass by accident.

- UNCOORDINATED
  - What it means: even the group failed too often. The information gap is real, but nobody got a
    natural moment to bring it up.
  - How to fix it: add or strengthen the shared consultation norm in `shared_context` so every
    fact-holder gets a real opening to speak. **Never fix this by telling an agent to be more talkative
    or assertive** — that manufactures the pass instead of earning it. Only change the shared world.

Always fix the *environment* (facts, norms, checks) but never the *agents*.