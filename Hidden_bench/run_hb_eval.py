"""run_hb_eval.py - evaluate the trained checkpoint on HiddenBench, out of distribution.

    python data/prepare_hiddenbench.py     # once: builds data/hiddenbench_ood.json
    python run_hb_eval.py                  # vanilla vs co-evolved on those 65 scenarios

NOTHING IS TRAINED HERE. This runs `inf.py` - the same evaluator, the same two blind arms,
the same metrics, the same paired bootstrap - against a test set the run has never seen in
any phase. 1100D's `unseen` split measures held-out DOMAINS from the same generator; this
measures a held-out BENCHMARK from a different one, which is the stronger transfer claim.

WHY A RUNNER INSTEAD OF EDITING inf.py

inf.py is shared: it drops into train_450v2_big, train_550D, the config.py ablation
folders. Hard-coding a second test set into it would follow it everywhere. Two overrides
before main() is cleaner and leaves that file untouched.

THE TRAP THIS EXISTS TO AVOID

inf.py records a completed arm as OUT_DIR/<arm>_done.json, keyed to a fingerprint of the
CHECKPOINT'S ADAPTER WEIGHTS (inf.run_arm). Here we evaluate the SAME checkpoint against a
DIFFERENT test set, so that fingerprint matches the one written by the 1100D eval. Left in
results_inf/, this run would print "already complete for this checkpoint -> reusing" and
hand back the 1100D numbers as if they were HiddenBench numbers - no error, no warning.
OUT_DIR below is therefore separate, and separate per seed.

WHAT YOU GET, AND HOW TO READ IT

All 65 scenarios are tagged eval_group="unseen", so inf.py reports one group plus TOTAL
and skips the seen/unseen gap section (inf.group_uids drops empty groups). The headline
block, the paired bootstrap against vanilla, the per-domain tables and the insight-leak
guardrail all work unchanged.

    content_frac    IS HiddenBench accuracy - there is exactly one content check
    prov_frac       of the hidden facts, the share both revealed AND cited
    decisive_revealed / reveals    information pooling: the benchmark's actual bottleneck
    success         stricter than the benchmark: correct AND fully sourced

Read `literature_replication` (5 scenarios: Stasser & Stewart 1992, Baker 2010, Toma &
Butera 2009, Schulz-Hardt et al. 2012, Graetz et al. 1998) separately from the rest. Those
are published studies whose answers are plausibly in Qwen3's pretraining; if the VANILLA
arm already solves them they are measuring recall, not pooling.

POWER. inf.py runs ONE rollout per scenario per arm, so this is n=65 with sampling at
T=0.8 (agent) / T=0.3 (router) - the paired CI will be wide. Run it at several SEEDs and
report mean +/- sd across them, using one seed's paired bootstrap as the primary test.
Do NOT instead duplicate scenarios inside the JSON: inf.bootstrap_paired resamples uids
without clustering, so replicates of one scenario would be counted as independent and the
interval would come back too narrow.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


class CFG:
    # Built by data/prepare_hiddenbench.py. Re-runnable; the raw benchmark is vendored.
    TEST_PATH = os.path.join(_HERE, "data", "hiddenbench_ood.json")

    # Must NOT be results_inf/ - see "THE TRAP THIS EXISTS TO AVOID" above.
    OUT_ROOT = os.path.join(_HERE, "results_inf_hb")

    # Rollout sampling only; the scenario set itself is fixed. Change it and re-run to
    # collect another seed - each lands in its own directory, so nothing is overwritten
    # and no completion marker is shared.
    SEED = 42

    # None = the folder's checkpoint auto-detection (ckpt/latest here). A directory
    # holding router/ and agent/ adapter subdirectories forces a specific one - use it to
    # evaluate a per-round checkpoint (ckpt/round1_router, ...) instead of the final.
    CKPT_PATH = None


def settlement_shapes(out_dir, test_path, arms):
    """Post-hoc: did the settlements use the schema's nesting? Changes no score.

    1100D's settlement_schema carries 5-9 decision fields, so nesting them under
    "decisions" is unmistakable. HiddenBench's carries exactly ONE, and a model may well
    flatten {"decisions": {"answer": X}} to {"answer": X}. env.eval_check only ever sees
    `decisions`, so a flat settlement scores 0 on the single content check while still
    passing every provenance check - checks_frac reads ~0.80 instead of 0.00 and the
    failure looks like "nearly right" rather than "wrong shape".

    That is a measurement artefact, not a reasoning failure, and it is invisible in the
    aggregates. Rather than quietly teach the verifier to accept both shapes - which
    would also change what 1100D's numbers mean and make the two reports
    incommensurable - this counts them and prints what a lenient parser would have
    scored. Report the scored number; cite the ceiling as the format's cost.

    The same applies to the answer STRING. prepare_hiddenbench emits the correct answer
    exact, lower-cased and upper-cased, which covers the realistic slips, but not every
    one: "West city" and "West City." both miss. Those are counted here as near misses
    rather than silently scored as wrong.
    """
    def norm(x):
        return "".join(c for c in str(x).lower() if c.isalnum())

    try:
        with open(test_path, encoding="utf-8") as f:
            answer = {r["uid"]: r["correct_answer"] for r in json.load(f)}
    except (OSError, ValueError, KeyError):
        return

    rows = []
    for tag in arms:
        path = os.path.join(out_dir, "{}_episodes.jsonl".format(tag))
        if not os.path.isfile(path):
            continue
        shape = {"nested": 0, "flat": 0, "other": 0, "unsettled": 0}
        lenient = near = total = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ep = json.loads(line)
                except ValueError:
                    continue
                total += 1
                st = ep.get("settlement")
                if not isinstance(st, dict) or not st:
                    shape["unsettled"] += 1
                    continue
                dec = st.get("decisions")
                if isinstance(dec, dict) and "answer" in dec:
                    shape["nested"] += 1
                    got = dec["answer"]
                elif "answer" in st:
                    shape["flat"] += 1
                    got = st["answer"]
                else:
                    shape["other"] += 1
                    continue
                # What a shape- and format-tolerant parser would have credited.
                want = answer.get(ep.get("uid"))
                if norm(got) == norm(want):
                    lenient += 1
                    # The emitted check accepts exact / lower / upper only.
                    if got not in (want, str(want).lower(), str(want).upper()):
                        near += 1
        if total:
            rows.append((tag, total, shape, lenient / total, near))

    if not rows:
        return
    print("\n" + "=" * 92)
    print("SETTLEMENT SHAPE  (diagnostic - no score was changed)")
    print("=" * 92)
    print("  {:<12s}{:>9s}{:>8s}{:>7s}{:>7s}{:>11s}{:>11s}{:>12s}".format(
        "arm", "episodes", "nested", "flat", "other", "unsettled", "near-miss",
        "ceiling"))
    for tag, total, shape, lenient, near in rows:
        print("  {:<12s}{:>9d}{:>8d}{:>7d}{:>7d}{:>11d}{:>11d}{:>12.4f}".format(
            tag, total, shape["nested"], shape["flat"], shape["other"],
            shape["unsettled"], near, lenient))
    if any(s["flat"] or s["other"] or n for _, _, s, _, n in rows):
        print("\n  A flat / other settlement scores 0 on the content check however good "
              "the answer is;")
        print("  a near-miss is an answer that matches only after case and punctuation "
              "are stripped.")
        print("  `ceiling` is what a shape- and format-tolerant parser would reach. The "
              "gap between it")
        print("  and content_frac above is FORMAT, not reasoning - report content_frac, "
              "cite the gap.")


def main() -> None:
    if not os.path.isfile(CFG.TEST_PATH):
        raise SystemExit(
            "[FAIL] {} not found.\n"
            "       Build it first:  python data/prepare_hiddenbench.py".format(
                os.path.relpath(CFG.TEST_PATH, _HERE)))

    sys.path.insert(0, _HERE)
    import inf

    inf.CONFIG.TEST_PATH = CFG.TEST_PATH
    inf.CONFIG.OUT_DIR = os.path.join(CFG.OUT_ROOT, "seed{}".format(CFG.SEED))
    inf.CONFIG.SEED = CFG.SEED
    if CFG.CKPT_PATH:
        inf.CONFIG.CKPT_PATH = CFG.CKPT_PATH

    print("=" * 92)
    print("HIDDENBENCH  --  out-of-distribution evaluation (no training)")
    print("=" * 92)
    print("  test set  : {}".format(os.path.relpath(inf.CONFIG.TEST_PATH, _HERE)))
    print("  out dir   : {}".format(os.path.relpath(inf.CONFIG.OUT_DIR, _HERE)))
    print("  seed      : {}".format(inf.CONFIG.SEED))
    print("=" * 92)

    inf.main()
    settlement_shapes(inf.CONFIG.OUT_DIR, CFG.TEST_PATH,
                      [t for t, _, _ in inf.CONFIG.ARMS])


if __name__ == "__main__":
    main()
