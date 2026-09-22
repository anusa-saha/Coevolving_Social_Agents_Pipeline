"""eval.py - two arms on the held-out 90. Blind, always.

    python eval.py

No flags. It finds the checkpoints main.py wrote and runs:

    arm         router LoRA   agent LoRA   what it is
    vanilla         off           off      the untrained base system
    coevolved       ON            ON       the trained system

Two arms, not four. The router_only / agent_only decomposition that train_V1 ran is a
diagnostic, not a result: with n=90 and effect sizes near 0.01 it produced bootstrap
p-values of 0.61, 0.81 and 0.98, i.e. four numbers none of which could be distinguished
from zero. If you want the decomposition back later, main.py already writes per-round
checkpoints (router_roundK / agent_roundK) and the honest version of that experiment is
the cross-product matrix, not two extra arms here.

BOTH ARMS ARE BLIND. Neither ever sees the privileged profile - that exists only inside
training, and this is asserted at startup rather than trusted. LoRA's B matrix is
zero-initialised, so adapter-OFF is mathematically the untrained model: `vanilla` needs
no second model load, and both arms share one router and one agent in memory.

BOTH ARMS GET LAST CALL. env.ENV_CFG.FORCE_DM_* is a property of the environment, not of
the policy, so it applies to vanilla too. That is deliberate and it makes the comparison
harder, not easier: it hands the baseline the ~28% of episodes train_V1 threw away
(unsettled episodes scored 0.000 in 360/360 measured cases) and forces the trained system
to win on something other than a broken loop.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import time
import warnings

warnings.filterwarnings(
    "ignore", message=r".*torch\.cpu\.amp\.autocast.*", category=FutureWarning)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import env
import opd
import prompts
import rl

_HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONFIG
# ============================================================

class CONFIG:
    DATASET_PATH = os.path.join(_HERE, "dataset.json")
    SPLIT = "eval"                  # the 90 scenarios main.py never trained on
    TRAIN_SPLIT = 0.8               # must match main.py or the split is not held out
    N_SCENARIOS = 0                 # 0 = all 90
    SEED = 42                       # must match main.py

    CKPT_DIR = os.path.join(_HERE, "ckpt")
    ROUTER_ADAPTER = os.path.join(CKPT_DIR, "router")
    AGENT_ADAPTER = os.path.join(CKPT_DIR, "agent")

    # (tag, router_lora, agent_lora)
    ARMS = [
        ("vanilla", False, False),
        ("coevolved", True, True),
    ]

    N_BOOT = 2000                   # paired bootstrap resamples

    # Episodes in flight at once. Eval is pure rollout - no gradients, no optimiser
    # state - so it can run a wider batch than training. 24 held-out scenarios advance
    # in lockstep per generate() call.
    ROLLOUT_BATCH = 24

    DEVICE = "cuda:0"
    ROUTER_DEVICE = "cuda:1"        # must match main.py's placement (12 GiB card)
    AGENT_DEVICE = "cuda:0"         # 24 GiB card

    OUT_DIR = os.path.join(_HERE, "results_eval")
    VERBOSE = True
    RELEASE_EVERY = 5
    CSV_MAX_CHARS = 4000


# ============================================================
# HELPERS
# ============================================================

def apply_device(cfg, log):
    r_dev = cfg.ROUTER_DEVICE or cfg.DEVICE
    a_dev = cfg.AGENT_DEVICE or cfg.DEVICE
    opd.OPD_CFG.DEVICE = r_dev
    rl.RL_CFG.DEVICE = a_dev
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for name, d in (("router", r_dev), ("agent", a_dev)):
            if "cuda" in str(d):
                idx = int(str(d).split(":")[1]) if ":" in str(d) else 0
                if idx >= n:
                    raise RuntimeError(
                        "{} wants {} but only {} CUDA device(s) visible.".format(
                            name, d, n))
        log("  devices : router={}  agent={}  ({} CUDA device(s))".format(r_dev, a_dev, n))
    else:
        log("  devices : CUDA unavailable -> cpu")


class Tee:
    def __init__(self, path, append=False):
        self.f = open(path, "a" if append else "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _mean(vals):
    vals = [v for v in vals if v != "" and v is not None and v == v]
    return (sum(vals) / len(vals)) if vals else float("nan")


def _clip(v, n):
    s = "" if v is None else str(v)
    return s if len(s) <= n else s[:n] + "...[clipped]"


def _num(row, key):
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ============================================================
# ARMS
# ============================================================

def run_arm(scenarios, router, agent, tag, use_router_lora, use_agent_lora, log):
    """`vanilla` for either side = that model's adapter disabled for the whole episode."""
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    fs = open(os.path.join(CONFIG.OUT_DIR, "{}_scenarios.csv".format(tag)), "w",
              newline="", encoding="utf-8")
    ft = open(os.path.join(CONFIG.OUT_DIR, "{}_turns.csv".format(tag)), "w",
              newline="", encoding="utf-8")
    fj = open(os.path.join(CONFIG.OUT_DIR, "{}_episodes.jsonl".format(tag)), "w",
              encoding="utf-8")
    ws = csv.DictWriter(fs, fieldnames=env.SUM_COLS, quoting=csv.QUOTE_ALL,
                        extrasaction="ignore")
    wt = csv.DictWriter(ft, fieldnames=env.TURN_COLS, quoting=csv.QUOTE_ALL,
                        extrasaction="ignore")
    ws.writeheader()
    wt.writeheader()

    log("\n" + "=" * 78)
    log("ARM {}   router_lora={}  agent_lora={}".format(tag, use_router_lora,
                                                        use_agent_lora))
    log("=" * 78)

    summaries = []
    t0 = time.time()
    done = 0
    # Lockstep, exactly as training runs it - env.run_batch is the only episode loop in
    # the codebase, so eval cannot drift from what was trained against.
    for blk in range(0, len(scenarios), CONFIG.ROLLOUT_BATCH):
        block = scenarios[blk:blk + CONFIG.ROLLOUT_BATCH]
        r_ctx = contextlib.nullcontext() if use_router_lora else router.base_mode()
        a_ctx = contextlib.nullcontext() if use_agent_lora else agent.ref_mode()
        try:
            with r_ctx, a_ctx:
                outs = env.run_batch(
                    block, router, agent,
                    student_profile=prompts.PROFILES["student"],
                    teacher_profile=None,        # NO privileged context at eval
                    record_router=False, record_agent=False)
        except Exception as exc:                 # noqa: BLE001
            log("    [SKIP] uids={} {}: {}".format(
                [x.uid for x in block], type(exc).__name__, exc))
            done += len(block)
            continue

        for out in outs:
            done += 1
            sm = out["summary"]
            summaries.append(sm)
            ws.writerow({k: _clip(sm.get(k), CONFIG.CSV_MAX_CHARS)
                         for k in env.SUM_COLS})
            for r in out["rows"]:
                wt.writerow({k: _clip(r.get(k), CONFIG.CSV_MAX_CHARS)
                             for k in env.TURN_COLS})
            fj.write(json.dumps(out["episode"], ensure_ascii=False) + "\n")

            if CONFIG.VERBOSE:
                log("  [{}/{}] uid={}  checks {}/{} ({:.2f})  content {}/{}  "
                    "prov {}/{}  success={}  reveals={}  settled={}  "
                    "t_settle={}".format(
                        done, len(scenarios), sm["uid"], sm["checks_passed"],
                        sm["checks_total"], sm["checks_frac"], sm["content_passed"],
                        sm["content_total"], sm["prov_passed"], sm["prov_total"],
                        sm["success"], sm["reveals"], sm["settled"],
                        sm["turns_to_settle"] or "-"))
                log("        route: {}".format(sm["route_sequence"]))
        fs.flush(); ft.flush(); fj.flush()
        router.release()
        agent.release()

    fs.close(); ft.close(); fj.close()
    log("\n  arm {} done: n={}  {:.1f}s".format(tag, len(summaries), time.time() - t0))
    return summaries


def summarize(log, tag, summaries):
    n = len(summaries)
    log("\n" + "-" * 78)
    log("SUMMARY  {}   (n={})".format(tag, n))
    log("-" * 78)
    if not n:
        return
    m = lambda k: _mean([s.get(k) for s in summaries])  # noqa: E731
    st = [s for s in summaries if s.get("settled")]
    cp_ids = {id(s) for s in st
              if s.get("decisive_revealed") == s.get("decisive_total")}
    cp = [s for s in st if id(s) in cp_ids]
    inc = [s for s in st if id(s) not in cp_ids]

    log("  CHECKS")
    log("    content checks passed (mean)   : {:.3f} / {:.3f}".format(
        m("content_passed"), m("content_total")))
    log("    provenance checks passed (mean): {:.3f} / {:.3f}".format(
        m("prov_passed"), m("prov_total")))
    log("    BOTH: checks passed (mean)     : {:.3f} / {:.3f}".format(
        m("checks_passed"), m("checks_total")))
    log("    checks_frac (mean fraction)    : {:.4f}".format(m("checks_frac")))
    log("    success (ALL checks pass)      : {:.4f}".format(m("success")))
    log("  BEHAVIOUR")
    log("    reveals (mean)                 : {:.3f}".format(m("reveals")))
    log("    decisive facts revealed        : {:.3f} / {:.3f}".format(
        m("decisive_revealed"), m("decisive_total")))
    log("    settled rate                   : {:.4f}".format(m("settled")))
    log("    turns to settle (mean, settled): {:.2f}".format(m("turns_to_settle")))
    log("    turns used (mean)              : {:.2f}".format(m("turns_used")))
    log("  ROUTING")
    log("    routing precision              : {:.4f}".format(m("routing_precision")))
    log("    router invalid picks (mean)    : {:.3f}".format(m("route_invalid")))
    log("    forced hand-overs (last call)  : {:.3f}".format(m("forced_routes")))
    log("    free routing decisions (mean)  : {:.3f}".format(m("route_decisions")))
    log("  INSIGHT")
    log("    schema slots named             : {:.4f}".format(m("insight_slot_frac")))
    log("    PRIVATE leak = telepathy (max) : {:.4f}   <- GUARDRAIL".format(
        m("insight_private_leak_max")))
    # The decomposition that says WHICH bottleneck is binding. In train_V1 these were
    # 0.000 / 0.735 / 0.218 -- i.e. everything unsettled scored zero, and even a complete
    # record left a flat ~25% per-field transcription error.
    log("  BOTTLENECK DECOMPOSITION")
    log("    unsettled episodes             : n={:<3d} checks_frac={:.4f}".format(
        n - len(st), _mean([s["checks_frac"] for s in summaries if not s.get("settled")])))
    log("    settled, record INCOMPLETE     : n={:<3d} checks_frac={:.4f}".format(
        len(inc), _mean([s["checks_frac"] for s in inc])))
    log("    settled, record COMPLETE       : n={:<3d} checks_frac={:.4f}  success={:.4f}".format(
        len(cp), _mean([s["checks_frac"] for s in cp]),
        _mean([s["success"] for s in cp])))
    log("    (the last row is the ceiling routing alone can reach; beating it requires")
    log("     better settlement text, which is what the agent phase trains)")


# ============================================================
# COMPARISON
# ============================================================

def paired_bootstrap(base, arm, n_boot, seed=0):
    """Bootstrap over the PAIRED per-scenario differences. Both arms run the same 90
    scenarios, so pairing removes the between-scenario variance that dominates this task
    (84.2% of it, measured) - an unpaired test here would be almost pure noise."""
    rng = random.Random(seed)
    d = [a - b for a, b in zip(arm, base)]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = sum(d) / n
    means = []
    for _ in range(n_boot):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    # two-sided p: how often does a mean-centred resample reach |obs|?
    hits = sum(1 for m in means if abs(m - obs) >= abs(obs))
    return obs, lo, hi, hits / n_boot


def compare(log, tags, baseline):
    rows = {}
    for tag in tags:
        p = os.path.join(CONFIG.OUT_DIR, "{}_scenarios.csv".format(tag))
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rows[tag] = {int(r["uid"]): r for r in csv.DictReader(f)}
    if baseline not in rows or len(rows) < 2:
        log("\n[compare] need at least two completed arms")
        return

    uids = set(rows[baseline])
    for t in rows:
        uids &= set(rows[t])
    uids = sorted(uids)

    log("\n" + "=" * 92)
    log("PAIRED COMPARISON   baseline = {}   n paired = {}".format(baseline, len(uids)))
    log("=" * 92)
    other = [t for t in tags if t != baseline and t in rows]
    log("  {:<34s}{:>12s}{:>14s}".format("metric", baseline, other[0] if other else ""))
    log("  " + "-" * 62)
    for key, higher in env.METRICS:
        vals = {}
        for t in rows:
            v = [_num(rows[t][u], key) for u in uids]
            v = [x for x in v if x is not None]
            vals[t] = (sum(v) / len(v)) if v else None
        if vals.get(baseline) is None:
            continue
        line = "  {:<34s}{:>12.4f}".format(key, vals[baseline])
        for t in other:
            if vals.get(t) is None:
                line += "{:>14s}".format("-")
                continue
            better = (vals[t] > vals[baseline]) if higher else (vals[t] < vals[baseline])
            line += "{:>13.4f}{}".format(vals[t], "*" if better else " ")
        log(line)
    log("\n  * = better than baseline")

    log("\n" + "=" * 92)
    log("PAIRED BOOTSTRAP vs {}   ({} resamples, 95% CI)".format(baseline, CONFIG.N_BOOT))
    log("=" * 92)
    log("  {:<26s}{:>10s}{:>20s}{:>10s}".format("metric", "delta", "95% CI", "p"))
    log("  " + "-" * 66)
    for key, _higher in env.METRICS:
        b = [_num(rows[baseline][u], key) for u in uids]
        for t in other:
            a = [_num(rows[t][u], key) for u in uids]
            pairs = [(x, y) for x, y in zip(b, a) if x is not None and y is not None]
            if not pairs:
                continue
            obs, lo, hi, p = paired_bootstrap([x for x, _ in pairs],
                                              [y for _, y in pairs], CONFIG.N_BOOT)
            log("  {:<26s}{:>+10.4f}   [{:+.4f}, {:+.4f}]{:>10.4f}{}".format(
                key, obs, lo, hi, p, "  *" if p < 0.05 else ""))

    log("\n  * = p < 0.05. A CI that straddles zero means the arms are not distinguishable")
    log("    at n={}, whatever the point estimate says.".format(len(uids)))

    # ---- the guardrail verdict, stated rather than left to be noticed ----
    lk = "insight_private_leak_max"
    bl = _mean([_num(rows[baseline][u], lk) for u in uids])
    log("\n" + "=" * 92)
    log("GUARDRAIL   insight private leak (telepathy detector)")
    log("=" * 92)
    for t in other:
        al = _mean([_num(rows[t][u], lk) for u in uids])
        verdict = ("OK" if al <= bl * 1.5 + 0.01 else
                   "SUSPECT - the insight head may be fabricating hidden values")
        log("  {:<12s} {:.4f}  vs  {} {:.4f}   -> {}".format(t, al, baseline, bl, verdict))
    log("  A checks_frac gain that comes with a large leak rise is memorised telepathy,")
    log("  not routing. Reject that checkpoint rather than reporting it.")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    log = Tee(os.path.join(CONFIG.OUT_DIR, "eval.log"))

    random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)

    scenarios = env.load_split(CONFIG.DATASET_PATH, CONFIG.SPLIT, CONFIG.TRAIN_SPLIT,
                               CONFIG.SEED, CONFIG.N_SCENARIOS)

    log("=" * 78)
    log("EVALUATION   vanilla  vs  co-evolved      (held-out split, blind)")
    log("=" * 78)
    log("  scenarios : {} (eval split, seed={}, train_split={})".format(
        len(scenarios), CONFIG.SEED, CONFIG.TRAIN_SPLIT))
    log("  arms      : {}".format(", ".join(t for t, _, _ in CONFIG.ARMS)))
    log("  profile   : student (BLIND) for both arms - no privileged context at eval")
    log("  last call : complete={}  final_turn={}  (applies to BOTH arms)".format(
        env.ENV_CFG.FORCE_DM_WHEN_COMPLETE, env.ENV_CFG.FORCE_DM_ON_LAST_TURN))
    log("  batch     : {} episodes in lockstep per generate() call".format(
        CONFIG.ROLLOUT_BATCH))
    apply_device(CONFIG, log)

    r_ad = CONFIG.ROUTER_ADAPTER if os.path.isdir(CONFIG.ROUTER_ADAPTER) else None
    a_ad = CONFIG.AGENT_ADAPTER if os.path.isdir(CONFIG.AGENT_ADAPTER) else None
    if r_ad is None or a_ad is None:
        log("")
        log("  [FAIL] no trained checkpoint under {}".format(CONFIG.CKPT_DIR))
        log("         router: {}   agent: {}".format(
            r_ad or "MISSING", a_ad or "MISSING"))
        log("         run `python main.py` first.")
        log.close()
        return
    log("  router ad : {}".format(r_ad))
    log("  agent ad  : {}".format(a_ad))
    log("=" * 78)

    router = opd.RouterLM(opd.OPD_CFG, adapter_path=r_ad, train=False)
    agent = rl.AgentLM(rl.RL_CFG, adapter_path=a_ad, train=False)

    # The privileged profile must never reach eval. Asserted, not trusted.
    sc = scenarios[0]
    bud = {a: env.ENV_CFG.MAX_TURNS_PER_AGENT for a in sc.agent_ids}
    blind = prompts.PROFILES["student"].route_prompt(sc, "", [], set(), bud, 1, 12)
    hidden = [v.get("text", "") for v in sc.private_facts.values() if v.get("text")]
    if any(h and h in blind for h in hidden):
        log("\n  [FAIL] the eval profile is leaking private facts. Aborting.")
        log.close()
        return
    log("\n  [ok] eval profile is blind: 0/{} hidden facts visible".format(len(hidden)))

    done = []
    for tag, r_lora, a_lora in CONFIG.ARMS:
        sums = run_arm(scenarios, router, agent, tag, r_lora, a_lora, log)
        summarize(log, tag, sums)
        done.append(tag)

    compare(log, done, CONFIG.ARMS[0][0])

    log("\n" + "=" * 92)
    log("  per-scenario csv : {}/<arm>_scenarios.csv".format(CONFIG.OUT_DIR))
    log("  per-turn csv     : {}/<arm>_turns.csv".format(CONFIG.OUT_DIR))
    log("  transcripts      : {}/<arm>_episodes.jsonl".format(CONFIG.OUT_DIR))
    log("=" * 92)
    log.close()


if __name__ == "__main__":
    main()
