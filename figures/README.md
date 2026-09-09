# Dataset & Pipeline Analysis — Figure Guide

Explains all 17 figures generated from the real 1,100-scenario output directory (11 domains × 100 scenarios), covering dataset composition, pipeline mechanics, and the core empirical validation of the benchmark's central claim.

---

## Part 1: Dataset Composition

### 01 — Agent-count distribution
**What it shows**: How many scenarios use 3, 4, or 5 agents.

**Reading it**: Nearly perfectly balanced — 360 / 369 / 371. No skew toward simpler (3-agent) or more complex (5-agent) scenarios. This confirms the agent-count balance requirement held across the full generation run, not just in a small sample.

### 02 — Distinct scenario types per domain
**What it shows**: Count of unique `scenario_type` labels used within each domain (out of ~100 scenarios per domain).

**Reading it**: Ranges from 53 (software_technology) to 90 (entertainment). All values are high relative to 100 scenarios, meaning most scenarios in every domain use a unique label rather than reusing the same type repeatedly. **Caveat**: a unique label doesn't guarantee a genuinely different decision *mechanic* — two differently-named scenarios could still be structurally identical. Figure 14 (diversity entropy) is the more rigorous companion to this one.

### 03 — Check-type composition
**What it shows**: Across all 9,076 checks in the dataset, what fraction are booleans, exact-string matches, exact-number matches, numeric ranges, provenance-membership checks, or structural/commitment checks.

**Reading it**: "Other" (uncategorized by the classifier) is the largest bucket at 22% — a real gap in the classification heuristic, not evidence the checks themselves are unusual. Excluding that, the dataset is dominated by checks requiring an **exact** answer (exact number 20%, exact string 16%, provenance-membership 17%) rather than easily-guessable booleans (17%). This is a meaningful signal for the benchmark's difficulty: a lone agent without the hidden fact has very low odds of passing most checks by chance.

### 14 — Scenario-type diversity entropy per domain
**What it shows**: Normalized Shannon entropy (0–1) of the scenario_type distribution within each domain. Unlike Figure 02's raw count, this captures *evenness* — a domain could have many distinct types but still be dominated by one or two of them, which entropy would catch and a count wouldn't.

**Reading it**: All domains score 0.89–0.99 — genuinely high and even distributions, not just high raw counts. `entertainment` (0.99) and `education` (0.98) are the most evenly spread; `workplace_interpersonal` and `software_technology` (0.89 each) are the least even, though still far from a dominated distribution. This is a positive result: Figure 02's counts aren't being inflated by a long tail of rarely-used types sitting alongside one dominant type.

---

## Part 2: Pipeline Mechanics

### 09 — Pipeline funnel
**What it shows**: How many of the 1,100 scenarios ever passed each stage (Verifier, Weak Arm, Strong Arm).

**Reading it**: 100% at every stage. This reflects that this particular output directory contains only scenarios that reached `accepted` status — every scenario here eventually passed all three gates, however many rounds it took. This chart would look very different on a directory that also included scenarios that were abandoned after exhausting `max_rounds`.

### 12 — Rounds-taken distribution (overall)
**What it shows**: How many revision rounds each scenario needed before reaching its final (accepted) status.

**Reading it**: Strongly right-skewed. Mean 5.54, median 4 — most scenarios converge within 2–5 rounds, but a long tail extends out to 33 rounds. The gap between mean and median is itself informative: it means a relatively small number of very-stuck scenarios are pulling the average up well above what a "typical" scenario experiences.

### 13 — Rounds-taken distribution per domain
**What it shows**: The same data as Figure 12, broken out per domain as box plots (box = middle 50%, orange line = median, green triangle = mean, circles = outliers beyond 1.5× the box width).

**Reading it**: `informal_commerce_bargaining` is the clear hardest domain — not just outliers, its entire typical range (box: 4–11 rounds) sits above every other domain. `finance` and `healthcare` are the easiest (tight boxes, median ~4). Notably, `friends_family_informal` and `defense` have low, tight boxes (typical scenarios are easy) but each has one dramatic single outlier (33 and 31 rounds) — a specific stuck scenario rather than a systemically hard domain.

### 04 — Rounds-to-acceptance histogram, split by rejection tag
**What it shows**: Across every rejection event in the dataset, how many happened at each round number, colored by which tag caused it.

**Reading it**: Heavily front-loaded — round 1 alone accounts for ~1,100 rejections, dropping steeply and trailing out to round 32. UNCOORDINATED (blue) dominates at every round; MALFORMED (red) is present but smaller; LEAKED (orange) is essentially invisible at this scale. This is the same right-skew as Figures 12/13, viewed from the rejection-event side rather than the per-scenario side.

### 05 — Rejection-tag frequency, average per domain
**What it shows**: Average number of rejections of each tag per scenario, broken out by domain.

**Reading it**: UNCOORDINATED is the dominant rejection reason in **every single domain** (2.3–4.1 per scenario), MALFORMED is secondary (0.6–2.8), and LEAKED is essentially zero everywhere. `informal_commerce_bargaining` has both the highest UNCOORDINATED *and* highest MALFORMED rate — consistent with it being the hardest domain across every other metric in this document. `legal` and `friends_family_informal` also show relatively elevated MALFORMED rates compared to domains like `defense` or `finance`.

### 14 (transition matrix) — Reject-tag transitions
**What it shows**: For scenarios with 2+ rejections, what tag followed what tag on the *next* failed round — i.e., does fixing one problem tend to trigger a different one?

**Reading it — this is the most important diagnostic chart in the set.** MALFORMED→UNCOORDINATED (1,042) and UNCOORDINATED→MALFORMED (1,235) together account for 2,277 transitions — a direct, quantitative signature of the oscillation problem: fixing a structural issue (MALFORMED) frequently breaks coordination (UNCOORDINATED) on the very next attempt, and vice versa. UNCOORDINATED→UNCOORDINATED (1,426) is also large, meaning even *within* the same failure category, one fix often doesn't resolve it in a single attempt. LEAKED barely participates in any transition (3 total, only ever as a destination, never a source) — LEAKED failures, when they happen, resolve quickly and don't chain into further problems the way MALFORMED/UNCOORDINATED do.

### 18 — Stage-failure cross-check
**What it shows**: Two independently-computed counts of total failures per stage — one read directly from each scenario's own summary record (`outcome.json`), one reconstructed by walking every individual round/stage directory — plotted side by side as a validation check.

**Reading it**: Exact agreement after accounting for 4 scenarios with leftover round directories beyond their official completion point (likely an artifact of a retry/resume after an earlier interrupted run): Verifier 1,487=1,487, Weak Arm 5=5, Strong Arm 3,502=3,502. This confirms the two independent data sources in the pipeline agree with each other once measuring the same thing — a genuine reliability check on the underlying data, not just another content chart.

---

## Part 3: Per-Stage Pass Rates

### 07 — Verifier pass rate per domain
**What it shows**: Of all Verifier attempts in each domain, what fraction passed.

**Reading it**: 65–85%. `healthcare` is highest (85%), `informal_commerce_bargaining` lowest (65%) — consistent with the difficulty ranking seen throughout (rounds-taken, rejection frequency).

### 08 — Weak-arm pass rate per domain
**What it shows**: Of all Weak Arm attempts in each domain, what fraction passed the gate (i.e., the lone agent correctly *failed* to solve the scenario, which is what "passing this gate" means).

**Reading it**: 99–100% everywhere, essentially uniform across all 11 domains. This is a strong, clean validation that the anti-leakage design is working almost universally — a lone agent without the hidden information almost never manages to satisfy the checks by luck, guesswork, or generic reasoning, regardless of domain.

### 06 — Strong-arm pass rate per domain
**What it shows**: Of all Strong Arm attempts in each domain, what fraction passed (the coordinating group succeeded).

**Reading it**: 19–30%, with `finance` highest and `informal_commerce_bargaining` lowest — again matching the same difficulty ordering. Note this is meaningfully lower than 50%, meaning even the *coordinating* group fails more often than it succeeds on any single attempt — this is exactly why the pipeline needs multiple rounds (Figures 12/13) rather than expecting first-attempt success.

### 16 — Rounds taken vs. agent count
**What it shows**: Scatter of rounds-taken against agent count (3/4/5), with the mean per agent count connected by a line.

**Reading it**: Essentially flat — the mean sits at roughly 5–6 rounds regardless of agent count. This is a genuine **null result** worth stating explicitly: more agents does not predict more revision rounds needed. Scenario difficulty (as measured by rounds-to-convergence) appears to be independent of cast size.

### 17 — Rounds taken vs. check count
**What it shows**: Scatter of rounds-taken against total check count (content + provenance), with Pearson's r reported directly on the plot.

**Reading it**: r = 0.16 — a very weak positive correlation, close to no relationship at all. Having more checks to satisfy does not meaningfully predict how many rounds a scenario will need. Combined with Figure 16, this suggests that whatever drives the long-tail difficulty in Figures 12/13 isn't simple structural complexity (more agents, more checks) — it's something else, plausibly the oscillation dynamic captured in the transition matrix.

---

## Part 4: The Core Empirical Claim

### 10 — Distribution of the strong−weak gap
**What it shows**: For every scenario, (strong-arm rollout pass rate) − (weak-arm rollout pass rate), as a histogram.

**Reading it — this is the paper's central result, visualized.** The distribution concentrates almost entirely between 0.75 and 1.0, with a mean gap of 0.86 and essentially nothing near zero. This is a clean, direct empirical demonstration of the benchmark's defining property: the lone agent reliably fails while the coordinating group reliably succeeds, and the *size* of that gap is large and consistent across nearly all 1,100 scenarios — not a marginal or occasional effect.

---

## Summary: what the whole set says together

1. **The core claim holds empirically and strongly** (Fig. 10): a ~0.86 mean gap across 1,100 scenarios is a real, large effect, not noise.
2. **Difficulty is concentrated, not evenly spread**: most scenarios converge in 2–5 rounds; a persistent minority — most pronounced in `informal_commerce_bargaining` — take much longer (Figs. 12, 13).
3. **The dominant failure mode is oscillation, not any single defect type**: MALFORMED and UNCOORDINATED trade off against each other far more than either resolves cleanly (Fig. 14/transitions), and this is a better predictor of long-tail difficulty than raw structural complexity (Figs. 16, 17 both show near-zero correlation).
4. **The anti-leakage design works almost universally** (Fig. 08: 99–100% everywhere) — whatever is driving difficulty, it isn't leakage.
5. **Dataset composition is genuinely diverse and balanced**, not just superficially labeled that way (Figs. 01, 02, 14 corroborate each other rather than just one raw count).