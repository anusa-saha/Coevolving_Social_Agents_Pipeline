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
- The scenario must contain 3, 4, or 5 agents only. Do not write scenario logic that depends on a specific count. The scenario should remain valid for any agent count within this range.

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

Set `decision_maker`, `turn_order`, and set `turn_cap` dynamically based on the number of agents (approximately 3–4 turns per agent, enough for everyone to speak once or twice and for the decision-maker to settle). Do not use the same constant turn cap for every scenario.

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

## Feedback prompt (used when a scenario is rejected)

After you submit a scenario, it gets tested. If it fails, you'll get a message like this, plus your full
scenario JSON. Revise the same scenario — don't start over unless told to.


Your previous scenario was REJECTED. Tag: {REJECT_TAG}

What went wrong: {DIAGNOSIS}

Evidence: {EVIDENCE}

What to fix: {FIX_INSTRUCTIONS}

Rules for your revision:
- Keep everything that already works. Change only what caused the rejection.
- Return the complete scenario object again, in the same JSON format as before.
- Return JSON only — no explanation.


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