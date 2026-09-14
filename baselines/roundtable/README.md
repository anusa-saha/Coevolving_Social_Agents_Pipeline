# Round Table

The method-free baseline. N agents sit at a table, take turns, and converge on one
settlement. No planner, no strategist, no act taxonomy, no reasoning scaffold, no stall
detector, and nothing is trained.

It exists because the other five arms need a floor that is *theirs*. Every one of them is
"CSA plus a mechanism"; without this, a reported gain is measured against nothing in
particular. Here the only machinery is the view filtering that the benchmark itself
requires — each agent sees its own facts and no one else's — and the shared verifier.

## Standalone

This folder imports nothing from the rest of the repo. Copy it anywhere, point it at the
scenarios, and it runs:

```bash
cp -r roundtable /somewhere/else && cd /somewhere/else/roundtable
export CSA_RAW_DIR=/path/to/raw      # or let it fetch from the Hub
python selftest.py
```

The five `_*.py` files are **generated copies** of the shared modules — the split, the
disclosure detector, the verifier, the paths logic, the version shims. Never edit them.
They exist so the folder can travel, and the cost is that this arm's scoring rule could
drift from the one the other five use, which would not crash anything: it would quietly
make the numbers incomparable.

So the copies are checked. `selftest.py` re-renders each one from `csa_core/` and fails if
they differ; when `csa_core/` is not reachable — the normal state once this folder has been
lifted out — it says so rather than passing silently.

```bash
python vendor.py --check   # report drift, exit 1 if any
python vendor.py           # refresh from csa_core after changing the contract
```

To change the scoring contract, edit `csa_core/` and re-run `vendor.py`. Editing a `_*.py`
directly is the one thing that breaks the guarantee.

## What it does

```
for round in 1..rounds:
    for agent in turn_order:
        agent speaks once, having seen the transcript and its own view
then: turn N opinions into one settlement
```

That last step is the only real design choice, and there are three, because they answer
different questions:

| `--decide` | how one answer emerges | what it tests |
|---|---|---|
| `chair` | the `decision_maker` writes the settlement | the floor that lines up against every other arm, all of which settle this way |
| `vote` | every agent proposes; each decision field goes to the majority | whether the table collectively knows more than its chair |
| `converge` | the chair proposes, the others reply `AGREE` or return a correction, last correction wins | closest to "they reach a common conclusion"; also the most expensive |

**`chair` is the row for the comparison table.** The other two are interesting on their own
terms but change the protocol, so they are not like-for-like against PPDPP or EPO.

Two details in `vote` worth knowing. Ties break towards the chair's own proposal, so a
split table degrades to `chair` rather than to an arbitrary agent. And `credited_facts` /
`justification_fact_ids` are **unioned, not voted** — a fact some agent genuinely relied on
was relied on, and majority-voting provenance would throw away real disclosures.

## Models

```bash
# the row that belongs in the table: same model as every other arm
python run_rt.py --backend local --decide chair --split test

# a frontier reference through an OpenAI-compatible endpoint
export OPENAI_API_KEY=...        # never put a key in a file
python run_rt.py --backend api --api_model gpt-5.4-luna --decide chair --split test
```

> The `api` backend is **not comparable with the other five arms**, which are all
> Qwen2.5-7B. A win there partly means "the frontier model is better than a 7B", which is
> not a finding about information pooling. It answers a different and genuinely useful
> question — how much of CSA is hard versus how much is the 7B — so run it *alongside*
> local, never instead.

Every agent at the table uses the same backend instance. A mixed table (some agents
frontier, some 7B) would confound any conclusion with which agent got which model; that is
a different experiment and this arm does not run it.

## Flags

| flag | default | |
|---|---|---|
| `--decide` | `chair` | `chair` / `vote` / `converge` |
| `--rounds` | 2 | passes around the table before settling |
| `--converge_rounds` | 2 | extra passes when `--decide converge`; stops early on unanimous agreement |
| `--backend` | `local` | `local` / `api` |
| `--api_model` | `gpt-5.4-luna` | also `CSA_RT_API_MODEL` |
| `--api_base` | — | OpenAI-compatible base URL; also `CSA_RT_API_BASE` |
| `--limit` | 0 | first N scenarios, for a smoke run |

## Check it first

```bash
python selftest.py
```

Eleven CPU checks, about a minute, no model. The ones that matter:

* **every agent sees only its own facts** — all agents, all 550 scenarios. This arm has no
  other safeguard, so a filtering bug here would silently turn the benchmark into a
  reading-comprehension task.
* **vote / converge behave as documented** — majority wins, ties go to the chair, facts
  union, converge halts on unanimous agreement instead of burning `converge_rounds`.
* **record schema matches the other arms** — a missing key would silently drop this arm
  out of whole tables in `analysis/compute_extended_metrics.py`.

## Output

`logs/Record-rt-<decide>-<backend>-<split>.txt`, same schema as every other arm, so:

```bash
cd ../analysis && python compute_extended_metrics.py
```

picks it up with no changes and scores it beside the rest as `RT chair/local`.

## Cost

`chair`: `rounds` × N agents calls per scenario, plus one settlement call. At N=4 and
rounds=2 that is ~9 calls, against ~14 for the steered arms — so if it scores close to
them, read the `dca/100call` column before concluding anything.

`vote` adds one settlement call per agent. `converge` adds up to `converge_rounds` × (N−1),
though unanimous agreement short-circuits it.
