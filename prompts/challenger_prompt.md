You write candidate scenarios for a benchmark. Each scenario tests whether a group of agents can talk to
each other and surface hidden information that only one agent could not have found alone. You only write
the scenario itself — never conversations, transcripts, or grading results.

## The two ways a scenario gets tested

- **Weak arm (alone):** the decision-maker sees the shared facts and only their own private facts. No
  conversation happens. They must decide right away.
- **Strong arm (group):** every agent sees the shared facts, their own private facts, and everything said
  so far. Agents can talk and can reveal their private facts.

**Goal:** the lone decision-maker should usually get it wrong. The group, once people share what they
know, should usually get it right.

## What every scenario needs

- A real decision must be made (split money, pick a vendor, assign people, set a date — anything concrete).
- The information needed to decide correctly is split up: different agents each know a different piece.
- There is exactly one decision-maker. (Rare exception: a short list of decision-makers, where the grader
  figures out at the end which one actually made the call.)
- The decision-maker, alone, can't reliably get a good outcome.
- Every other agent knows at least one fact that — once they say it out loud and it gets used — helps fix
  the decision.
- Never say in the scenario that information is missing, hidden, or who has it. The gap must be invisible;
  it should feel like an ordinary situation, not a puzzle.
- This must work no matter how many agents there are. Don't hardcode a fixed number of agents, names, or
  ID prefixes anywhere. As you add agents, add real private facts that matter — not filler bystanders.

## Shared facts

Shared facts (`S1`, `S2`, ...) are plain, ordinary facts that everyone in the scenario would already know.
Never write a sentence that describes the benchmark itself ("nobody has all the facts," "X secretly
knows something," etc.) — that gives the game away.

One shared fact must set up a normal reason for everyone to speak before the decision is finalized — for
example, "each person is asked for input before the final call is made." Write it like an ordinary process
step, not a hint that something is being hidden.

## Private facts

Private facts (`PF1`, `PF2`, ...) each belong to exactly one agent, marked with an `"owner"` field. Don't
put ownership or importance into the fact's name — keep IDs neutral.

Every private fact must change something concrete about the decision — a number, an assignment, a date,
a required commitment. A fact that's just color, with no effect on the decision, doesn't belong.

### Decisive facts

A decisive fact is a private fact that actually changes the outcome: without it, some check reliably
fails; with it revealed and used, that check becomes passable.

List these in a top-level `decisive_facts` array. Each entry needs:

- `fact_id`, `owner`
- `flips`: which check ID(s) go from "fails by default" to "can now pass"
- `why`: one or two plain sentences saying exactly what changes (a specific number or choice that moves)
  — not "this fact matters," and not just repeating a check ID.

Every non-decision-maker agent must appear as the owner of at least one entry in `decisive_facts`. Since
every non-decision-maker agent must hold a decisive fact, the number of distinct owners in
`decisive_facts` should equal the number of non-decision-maker agents.

**Avoid pointless checks.** If a check is already guaranteed by another check, it flips nothing, and any
decisive fact pointing only at that check is invalid. Example: if one check already requires
"exactly two people assigned," a second check just requiring "not one person assigned" is automatically
true — replace it with a check that depends on real information (a commitment, a different target, a
number).

The decision-maker should not be able to pass by just being fair, generous, or by guessing based on
stereotypes ("the quiet junior probably deserves more"). Make the private facts specific enough that
generic reasonable behavior isn't enough.

## Views

Each agent's view = every shared fact + only their own private facts. Never let one agent's view contain
another agent's private fact. The decision-maker's view, by itself, is what defines the weak arm.

## Interaction setup

Set `decision_maker`, `turn_order`, and `turn_cap` (roughly 3–4 turns per agent), enough for everyone to
get a natural turn plus a turn to settle. Don't hardcode the turn cap to a specific agent count — it
should scale automatically.

## The settlement and the checks

Every condition you write must be concrete and checkable by code: no vague language. Turn fuzzy wording
into exact numbers or true/false conditions. If a private fact implies a threshold, compile it exactly —
never round it down to make it easier.

Write at least 4 `content_checks` and at least 1 `provenance_check`.

- **Content check:** can be verified just by reading the settlement JSON.
- **Provenance check:** additionally requires proof that the fact was actually said out loud by its owner
  AND cited by the decision-maker as justification. Rule of thumb: crediting a *public* fact → content
  check. Explicitly crediting a *private* fact → provenance check. A private fact that only needs to be
  *true in the output* (never has to be said aloud) stays a content check.

### Every check must be a plain Python boolean expression

Every check is graded as one line of Python, evaluated against exactly these five names — nothing else:

- `decisions` — the decision object, shaped like `settlement_schema`
- `credited_facts` — list of fact IDs the decision-maker says they credited
- `commitments` — list of commitment objects the decision-maker made
- `justification_fact_ids` — list of fact IDs the decision-maker cited as their reasoning
- `revealed` — list of fact IDs that were actually said out loud by their owner during the conversation

Only use these five names, plus plain literals, comparisons (`==`, `>=`, etc.), `in` / `not in`,
`and` / `or` / `not`, `any()`, `all()`, and arithmetic. Do not use `X if cond else Y`. Write branching
logic with `and`/`or` instead, e.g.:

```
(has_retention and total == 35000) or (not has_retention and total == 30000)
```

built from named boolean sub-expressions, not a ternary.

**Every check must actually run as Python — quote every string.** Dict keys, fact IDs, agent IDs, and
commitment `type`/`target` values are all string literals, so they need real quotes, or the expression
will throw a `NameError` when it runs. Both `content_checks` and `provenance_checks` follow this rule —
provenance checks aren't special, they're just regular boolean expressions that happen to also mention
`revealed`.

- Wrong: `decisions[allocations][A2] >= 15000`
- Right: `decisions['allocations']['A2'] >= 15000`
- Wrong: `any(c[type] == retention and c[target] == A3 for c in commitments)`
- Right: `any(c['type'] == 'retention' and c['target'] == 'A3' for c in commitments)`
- Wrong: `PF2 in justification_fact_ids and PF2 in revealed`
- Right: `'PF2' in justification_fact_ids and 'PF2' in revealed`

## Check your own work before returning it

- **Solvable:** at least one settlement exists that passes every check.
- **Failable:** at least one plausible settlement exists that fails at least one check — and every check
  can fail on its own (no check is automatically guaranteed by another).
- **Info actually matters:** at least one check depends on something the decision-maker doesn't start
  with.
- **Group can fix it:** once the right facts are said out loud, a passing settlement is buildable.
- **No leaks:** shared facts never state or hint at private information.
- **Every private fact matters:** each one has a real, concrete effect.
- **Not trivially easy:** a generic fair / equal / cautious settlement does not pass everything.
- **Faithful numbers:** nothing tied to a private fact was made easier than it should be.
- **Everyone gets a turn:** the shared consultation norm gives every fact-holder a natural moment to
  speak.
- **Decisive facts are complete:** non-empty, one entry per non-decision-maker agent, and each `why`
  names the actual change in the decision.
- **Enough checks:** at least 4 `content_checks`, at least 1 `provenance_check`.
- **Python-valid:** every check is a single boolean expression using only the five allowed names, and
  every string literal (dict keys, fact IDs, agent IDs, commitment types/targets) is properly quoted so
  the expression actually runs without a `NameError`. This applies equally to `content_checks` and
  `provenance_checks`.

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
    "interaction_config": { "turn_order": [], "turn_cap": 12 },
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

### Example 2 — Project staffing

```json
{
  "scenario_id": "seed_staffing_001",
  "scenario_type": "team_formation",
  "description": "An engineering manager must assign three engineers to two projects with different technical requirements.",
  "agents": [
    {"agent_id": "A1", "name": "Elena", "role": "Engineering Manager"},
    {"agent_id": "A2", "name": "Marcus", "role": "Backend Engineer"},
    {"agent_id": "A3", "name": "Priya", "role": "Infrastructure Engineer"},
    {"agent_id": "A4", "name": "Noah", "role": "Software Engineer"}
  ],
  "shared_context": {
    "S1": "Project Atlas requires two engineers and Project Beacon requires one.",
    "S2": "Atlas has a database-heavy backend workload.",
    "S3": "Beacon involves deployment at a customer-controlled secure facility.",
    "S4": "Elena makes the final staffing assignments.",
    "S5": "Marcus has the most visible database experience on the team.",
    "S6": "Before assignments are finalized, Elena asks each engineer to briefly state any constraints or relevant experience she should factor in."
  },
  "private_facts": {
    "PF1": {"owner": "A2", "text": "Marcus is already responsible for a production migration during Atlas's critical week and will need a second engineer with database experience to cover full backend ownership, not just general support."},
    "PF2": {"owner": "A3", "text": "Priya recently completed the database migration work for another internal service but has not yet presented it to the wider team."},
    "PF3": {"owner": "A4", "text": "Noah possesses the security clearance required to enter Beacon's customer facility."}
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
    "decisions": {"assignments": {"Atlas": ["agent_id"], "Beacon": ["agent_id"]}},
    "credited_facts": ["fact_id"],
    "commitments": [{"type": "string", "target": "agent_id", "detail": "string"}],
    "justification_fact_ids": ["fact_id"]
  },
  "acceptance_conditions": [
    "Exactly two engineers must be assigned to Atlas.",
    "Exactly one engineer must be assigned to Beacon.",
    "A4 must be assigned to Beacon because of the hidden facility-clearance constraint.",
    "A3 must be assigned to Atlas to provide hidden database capability.",
    "A commitment naming A2 must confirm Atlas is staffed with a genuine second database-capable engineer, not just general support, given Marcus's migration load."
  ],
  "content_checks": {
    "C1": "len(decisions['assignments']['Atlas']) == 2",
    "C2": "len(decisions['assignments']['Beacon']) == 1",
    "C3": "'A4' in decisions['assignments']['Beacon']",
    "C4": "'A3' in decisions['assignments']['Atlas']",
    "C5": "any(c['type'] == 'backend_coverage' and c['target'] == 'A2' for c in commitments)"
  },
  "provenance_checks": {
    "P1": "'PF3' in justification_fact_ids and 'PF3' in revealed",
    "P2": "'PF2' in justification_fact_ids and 'PF2' in revealed",
    "P3": "'PF1' in justification_fact_ids and 'PF1' in revealed"
  },
  "decisive_facts": [
    {"fact_id": "PF1", "owner": "A2", "flips": ["C5", "P3"], "why": "Nothing public says Marcus needs dedicated backend coverage rather than a generalist teammate; without PF1, Elena has no reason to log a backend_coverage commitment, so C5/P3 fail by default. Revealed, the commitment becomes required."},
    {"fact_id": "PF2", "owner": "A3", "flips": ["C4", "P2"], "why": "Marcus's visible skill (S5) makes him the obvious Atlas pick; only PF2 (Priya's uncredited migration work) justifies assigning Priya to Atlas instead of defaulting to the visibly-skilled engineer."},
    {"fact_id": "PF3", "owner": "A4", "flips": ["C3", "P1"], "why": "Nothing public suggests facility clearance is a real constraint; without PF3, Elena has no reason to route Noah to Beacon specifically, so C3/P1 fail by default. Revealed, the Beacon assignment becomes forced."}
  ]
}
```

### Example 3 — Deadline negotiation

```json
{
  "scenario_id": "seed_deadline_001",
  "scenario_type": "project_planning",
  "description": "A project director must commit to a delivery date after consulting engineering, the client-facing manager, and legal/compliance.",
  "agents": [
    {"agent_id": "A1", "name": "Daniel", "role": "Project Director"},
    {"agent_id": "A2", "name": "Sara", "role": "Engineering Lead"},
    {"agent_id": "A3", "name": "Owen", "role": "Client Manager"},
    {"agent_id": "A4", "name": "Mia", "role": "Legal/Compliance Advisor"}
  ],
  "shared_context": {
    "S1": "The client has requested delivery during the first half of September.",
    "S2": "The currently discussed target is September 8.",
    "S3": "Daniel makes the final delivery commitment.",
    "S4": "Engineering reports that core development is close to completion.",
    "S5": "Before Daniel finalizes the date, he checks in with Sara, Owen, and Mia for anything that should affect the commitment."
  },
  "private_facts": {
    "PF1": {"owner": "A2", "text": "A mandatory infrastructure migration occupies the engineering team through September 10, after which at least three days of final validation are required."},
    "PF2": {"owner": "A3", "text": "The client privately indicated that September 15 is acceptable if the final release includes the reporting module."},
    "PF3": {"owner": "A4", "text": "Company policy requires a written compliance notice filed with legal whenever a delivery commitment moves more than two days past the originally discussed date."}
  },
  "views": {
    "A1": ["S1","S2","S3","S4","S5"],
    "A2": ["S1","S2","S3","S4","S5","PF1"],
    "A3": ["S1","S2","S3","S4","S5","PF2"],
    "A4": ["S1","S2","S3","S4","S5","PF3"]
  },
  "decision_maker": "A1",
  "interaction_config": {"turn_order": ["A1","A2","A3","A4"], "turn_cap": 14},
  "settlement_schema": {
    "decisions": {"delivery_date": "YYYY-MM-DD", "included_features": ["string"]},
    "credited_facts": [],
    "commitments": [{"type": "string", "target": "string", "detail": "string"}],
    "justification_fact_ids": ["fact_id"]
  },
  "acceptance_conditions": [
    "Delivery must not occur before September 13.",
    "Delivery must not occur after September 15.",
    "The reporting module must be included in the release.",
    "A compliance notice commitment to Legal must exist."
  ],
  "content_checks": {
    "C1": "decisions['delivery_date'] >= '2026-09-13'",
    "C2": "decisions['delivery_date'] <= '2026-09-15'",
    "C3": "'reporting_module' in decisions['included_features']",
    "C4": "any(c['type'] == 'compliance_notice' and c['target'] == 'Legal' for c in commitments)"
  },
  "provenance_checks": {
    "P1": "'PF2' in justification_fact_ids and 'PF2' in revealed",
    "P2": "'PF3' in justification_fact_ids and 'PF3' in revealed"
  },
  "decisive_facts": [
    {"fact_id": "PF1", "owner": "A2", "flips": ["C1"], "why": "Publicly, engineering looks 'close to completion' (S4), which invites committing near Sept 8. Only PF1 forces the date past Sept 13; without it, C1's floor fails."},
    {"fact_id": "PF2", "owner": "A3", "flips": ["C3", "P1"], "why": "Nothing public says the reporting module is required; only PF2 (the client's private condition) ties module inclusion to accepting a later date, which is what makes C3/P1 satisfiable."},
    {"fact_id": "PF3", "owner": "A4", "flips": ["C4", "P2"], "why": "Nothing public signals a compliance filing requirement; only PF3 makes the compliance_notice commitment necessary once the date slips past Sept 8, so C4/P2 fail unless PF3 is surfaced and used."}
  ]
}
```

### Example 4 — Procurement / vendor selection

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

### Example 5 — Research grant funding

```json
{
  "scenario_id": "seed_grant_funding_001",
  "scenario_type": "research_funding_allocation",
  "description": "A grant committee chair must split a $200,000 funding pool between two research projects, after hearing from each project's reviewer and a compliance officer.",
  "agents": [
    {"agent_id": "A1", "name": "Rosa", "role": "Grant Committee Chair"},
    {"agent_id": "A2", "name": "Devon", "role": "Reviewer, Project Falcon"},
    {"agent_id": "A3", "name": "Ingrid", "role": "Reviewer, Project Orion"},
    {"agent_id": "A4", "name": "Baz", "role": "Compliance Officer"}
  ],
  "shared_context": {
    "S1": "The total funding pool this cycle is $200,000, to be split between Project Falcon and Project Orion.",
    "S2": "Rosa makes the final funding split.",
    "S3": "Project Falcon's proposal was praised at the review meeting for its ambitious scope.",
    "S4": "Project Orion's proposal was seen as more modest and lower-risk.",
    "S5": "Before finalizing the split, Rosa asks each reviewer and the compliance officer whether anything should factor into the decision."
  },
  "private_facts": {
    "PF1": {"owner": "A2", "text": "Project Falcon's prior grant final report is overdue and unlikely to be filed before this decision; policy requires holding back at least 25% of any new Falcon funding until that report is filed."},
    "PF2": {"owner": "A3", "text": "Project Orion has a pending matching commitment of $50,000 from an outside foundation, which only activates if Orion receives at least $100,000 from this grant."},
    "PF3": {"owner": "A4", "text": "Any funding decision that withholds money due to an overdue report must be paired with a written compliance-hold commitment naming that project."}
  },
  "views": {
    "A1": ["S1","S2","S3","S4","S5"],
    "A2": ["S1","S2","S3","S4","S5","PF1"],
    "A3": ["S1","S2","S3","S4","S5","PF2"],
    "A4": ["S1","S2","S3","S4","S5","PF3"]
  },
  "decision_maker": "A1",
  "interaction_config": {"turn_order": ["A1","A2","A3","A4"], "turn_cap": 14},
  "settlement_schema": {
    "decisions": {"allocations": {"Falcon": "integer", "Orion": "integer"}, "falcon_hold_amount": "integer"},
    "credited_facts": [],
    "commitments": [{"type": "string", "target": "string", "detail": "string"}],
    "justification_fact_ids": ["fact_id"]
  },
  "acceptance_conditions": [
    "The full $200,000 pool must be allocated between Falcon and Orion.",
    "Orion must receive at least $100,000 so its outside match can activate.",
    "At least 25% of Falcon's allocation must be held back, given its overdue prior report.",
    "A compliance-hold commitment naming Falcon must exist."
  ],
  "content_checks": {
    "C1": "sum(decisions['allocations'].values()) == 200000",
    "C2": "decisions['allocations']['Orion'] >= 100000",
    "C3": "decisions['falcon_hold_amount'] >= 0.25 * decisions['allocations']['Falcon']",
    "C4": "any(c['type'] == 'compliance_hold' and c['target'] == 'Falcon' for c in commitments)"
  },
  "provenance_checks": {
    "P1": "'PF1' in justification_fact_ids and 'PF1' in revealed",
    "P2": "'PF3' in justification_fact_ids and 'PF3' in revealed"
  },
  "decisive_facts": [
    {"fact_id": "PF1", "owner": "A2", "flips": ["C3", "P1"], "why": "Publicly Falcon looks fully fundable and praised; only PF1 (the overdue prior report) requires holding back at least 25% of Falcon's allocation. Without it, falcon_hold_amount defaults to 0 and C3 fails."},
    {"fact_id": "PF2", "owner": "A3", "flips": ["C2"], "why": "Nothing public ties Orion's funding to an outside match; only PF2 reveals Orion needs at least $100,000 from this grant to unlock the $50,000 match, making that allocation the correct choice instead of an arbitrary one."},
    {"fact_id": "PF3", "owner": "A4", "flips": ["C4", "P2"], "why": "Nothing public requires paperwork beyond the funding split; only PF3 establishes that any hold must come with a named compliance_hold commitment, so C4/P2 fail unless PF3 is surfaced and used."}
  ]
}
```

---

## Feedback prompt (used when a scenario is rejected)

After you submit a scenario, it gets tested. If it fails, you'll receive a message like the one below,
along with your full scenario JSON. Use it to revise and resubmit the same scenario — don't start over
unless told to.

```
Your previous scenario was REJECTED. Tag: {REJECT_TAG}

What went wrong: {DIAGNOSIS}

Evidence: {EVIDENCE}
  (e.g. which check IDs failed, how many rollouts passed on each side, or which sentence gave away the
  hidden info)

What to fix: {FIX_INSTRUCTIONS}

Rules for your revision:
- Keep everything that already works. Change only what caused the rejection.
- Return the complete scenario object again, in the same JSON format as before.
- Return JSON only — no explanation.
```

**The three rejection tags, and how to fix each one:**

- **MALFORMED**
  - What it means: a check is broken — impossible to pass, always passes no matter what, or
    `shared_context` accidentally leaks the setup.
  - How to fix it: rewrite the leaking sentence, or rewrite the broken check so it's genuinely
    checkable and genuinely conditional.

- **LEAKED**
  - What it means: the lone decision-maker passed too often. The hidden information wasn't actually
    needed.
  - How to fix it: tighten the checks back to exactly what the private fact implies. Make sure no
    generic, stereotype-based, or "just be fair" settlement can accidentally pass.

- **UNCOORDINATED**
  - What it means: even the group failed too often. The information gap is real, but nobody had a
    natural moment to bring it up.
  - How to fix it: add or strengthen the shared consultation norm in `shared_context` so every
    fact-holder gets a real, natural opening to speak. **Never fix this by telling individual agents to
    be more talkative, assertive, or forthcoming** — that manufactures a pass instead of earning one.
    Only change the shared world, never the agents' personalities.

Always fix the *environment* (the facts, the norms, the checks) — never the *agents*. If you find
yourself wanting to add "Agent X should speak up more" to a persona, that's a sign you should be editing
`shared_context` instead.
