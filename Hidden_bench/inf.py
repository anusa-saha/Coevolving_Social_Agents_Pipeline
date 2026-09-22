"""inf.py - vanilla vs co-evolved on the held-out test set (seen / unseen / total).

    python inf.py

In THIS folder (train_1100_big) it reads data/1100_test.json, which carries eval_group
tags directly (2 domains held out entirely = "unseen", the other 9 domains' held-out 20%
= "seen"), and takes its GPU / VRAM cap / eval batch from gpu.py. run.py calls it after
main.py. If a test file ever lacks the tags, it falls back to deriving them from domain
membership in the matching *_train.json.

No flags. The same file also drops into any other training folder and adapts to it:

    folder style                        router module   checkpoint it evaluates
    flat trainer (train_1100_big,       opd.py          ckpt/latest/{router,agent}
                  train_450v2_big)
    flat trainer (train_550D, 1100D)    opd.py          ckpt/{router,agent}
    config.py ablation                  opd.py          runs/<RUN.NAME>/ckpt/{router,agent}
    config.py ablation, no insight      routerlm.py     runs/<RUN.NAME>/ckpt/{router,agent}

Ablation folders that import themselves as a package (`import abblations.<name>.env`)
are handled too, whatever the folder is actually called. Set CONFIG.CKPT_PATH to force a
specific checkpoint (a directory holding router/ and agent/).

    arm         router LoRA   agent LoRA   what it is
    vanilla         off           off      the untrained base system
    coevolved       ON            ON       the trained (final) checkpoint

data/1100_test.json (380 scenarios, none of them in data/1100_train.json - checked by
uid, by content hash and by domain+scenario_id in data/prepare_data.py):

    eval_group="seen"    180   the 9 trained domains' held-out 20%: 20 each
    eval_group="unseen"  200   2 domains no run trained on (informal_commerce_bargaining,
                               friends_family_informal), 100 each

data/550_test.json, where this file is used on 550D (190 scenarios):

    eval_group="seen"     90   450D's held-out 20%: 9 trained domains x 10
    eval_group="unseen"  100   the same 2 held-out domains, 50 each

Both arms are BLIND (student profile, no privileged context) - asserted at startup.

Report (results_inf/inf.log, or runs/<RUN.NAME>/inf/inf.log in a config.py folder,
plus CSVs):
    1. every arm, full metric block on SEEN, UNSEEN and TOTAL
    2. headline table: all metrics, both arms, all three groups side by side
    3. paired bootstrap, coevolved vs vanilla, per group
    4. generalisation gap (unseen - seen) per arm, and whether training widened it
    5. per-domain: overview table, then every metric per domain with paired delta / CI / p
    6. guardrail: insight private leak (skipped when the folder has no insight head)
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import inspect
import json
import os
import random
import re
import sys
import time
import types
import warnings

warnings.filterwarnings(
    "ignore", message=r".*torch\.cpu\.amp\.autocast.*", category=FutureWarning)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")    # match nvidia-smi (config.HW)

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONFIG
# ============================================================

class CONFIG:
    # None = the first that exists: 550_test.json next to this file, the folder config's
    # DATA.TEST_PATH (abblations arms), data/550_test.json, data/1100_test.json.
    TEST_PATH = None                 # 550D: 190 = 90 seen + 100 unseen | 1100D: 380 = 180 + 200
    # Only used when the test file has no eval_group tags: a test domain present in this
    # file is "seen". None = data/<same prefix>_train.json next to the test file.
    TRAIN_PATH = None
    N_SCENARIOS = 0                  # 0 = all; >0 = first N (smoke test only)
    SEED = 42                        # rollout sampling only; the set itself is fixed

    # ---- GPU placement --------------------------------------------------------
    # Where each model is loaded: "cuda:0", "cuda:1", "cpu". None = the folder's
    # config.py HW.*_DEVICE (None = its auto GPU choice), else gpu.py GPU, else "cuda:0".
    # A cuda:N that is not visible (e.g. a 1-GPU SLURM job, where the card is always
    # cuda:0) falls back to cuda:0 with a warning. The actual placement of every weight is
    # checked after loading and printed under DEVICES in the log; a mismatch aborts the run.
    ROUTER_DEVICE = None             # router base model + LoRA (name comes from the folder)
    AGENT_DEVICE = None              # agent base model + LoRA  (name comes from the folder)
    # Per-card VRAM cap in GiB for this process. None = the folder config's VRAM_GIB if it
    # has one (the abblations arms: 32), else config.HW.VRAM_GIB (Llama arms: 19), else
    # gpu.py's (train_550D: 34), else no cap.
    # A number here overrides them.
    VRAM_GIB = None

    # None = auto-detect (see module docstring). Otherwise a directory with router/ and
    # agent/ adapter subdirectories, absolute or relative to this folder.
    CKPT_PATH = None

    # (tag, router_lora, agent_lora). The first arm is the baseline for comparisons.
    ARMS = [
        ("vanilla", False, False),
        ("coevolved", True, True),
    ]

    # Episodes in lockstep per generate() call. None = the folder config's HW.EVAL_BATCH,
    # else gpu.py's EVAL_BATCH, else 24. env.run_batch halves and retries on OOM, so too high only costs time.
    ROLLOUT_BATCH = None
    N_BOOT = 2000                    # bootstrap resamples
    RESUME = True                    # skip an arm already completed for this checkpoint

    # None = runs/<RUN.NAME>/inf/ in a config.py folder, else results_inf/ here
    OUT_DIR = None
    VERBOSE = True
    CSV_MAX_CHARS = 4000


GROUPS = ("seen", "unseen")          # reported individually, then TOTAL
KEY_METRICS = ["checks_frac", "success", "settled", "decisive_revealed",
               "routing_precision", "turns_used"]   # for the gap / domain tables


# ============================================================
# PROJECT MODULES  (whatever this folder provides)
# ============================================================

def _package_prefix():
    """'abblations.train_expD_noInsights' if this folder's env.py imports itself as a
    package, else None (plain `import config` / `import opd` style)."""
    p = os.path.join(_HERE, "env.py")
    if not os.path.isfile(p):
        raise SystemExit("[FAIL] no env.py next to inf.py - put inf.py inside a "
                         "training folder.")
    with open(p, encoding="utf-8") as f:
        m = re.search(r"^import\s+([\w.]+)\.(?:config|prompts|rl)\s+as\s+\w+",
                      f.read(), re.M)
    return m.group(1) if m else None


def _load_modules():
    """Import env / prompts / rl / router module from THIS folder. For package-style
    folders the package name is aliased onto this directory, so the imports resolve here
    even if the folder was copied or renamed."""
    prefix = _package_prefix()
    if prefix is None:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        imp = importlib.import_module
    else:
        parts = prefix.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            pkg = types.ModuleType(name)
            pkg.__path__ = [_HERE] if i == len(parts) else []
            sys.modules[name] = pkg
        imp = lambda n: importlib.import_module(prefix + "." + n)  # noqa: E731

    mods = types.SimpleNamespace(prefix=prefix)
    mods.env = imp("env")
    mods.prompts = imp("prompts")
    mods.rl = imp("rl")
    if os.path.isfile(os.path.join(_HERE, "opd.py")):
        mods.router_mod = imp("opd")
    elif os.path.isfile(os.path.join(_HERE, "routerlm.py")):
        mods.router_mod = imp("routerlm")
    else:
        raise SystemExit("[FAIL] no opd.py or routerlm.py in {}".format(_HERE))
    mods.config = (imp("config") if os.path.isfile(os.path.join(_HERE, "config.py"))
                   else None)
    # GPU / VRAM helpers: gpu.py (train_550D), or a config.py that defines device() and
    # cap_vram() itself (the Llama arms)
    if mods.config is None and os.path.isfile(os.path.join(_HERE, "gpu.py")):
        mods.gpu = imp("gpu")
    elif mods.config is not None and hasattr(mods.config, "cap_vram"):
        mods.gpu = mods.config
    else:
        mods.gpu = None
    return mods


M = _load_modules()
env, prompts, rl = M.env, M.prompts, M.rl


def _resolve_config():
    """Fill every CONFIG value left as None from this folder: config.py (abblations arms,
    Llama arms: DATA / HW), else gpu.py (train_550D), else the defaults."""
    C, G = M.config, M.gpu
    hw = getattr(C, "HW", None)
    data = getattr(C, "DATA", None)
    if CONFIG.TEST_PATH is None:
        cands = [os.path.join(_HERE, "550_test.json"),
                 getattr(data, "TEST_PATH", None),
                 os.path.join(_HERE, "data", "550_test.json"),
                 os.path.join(_HERE, "data", "1100_test.json")]
        CONFIG.TEST_PATH = next((c for c in cands if c and os.path.isfile(c)), cands[0])
    if CONFIG.TRAIN_PATH is None:
        CONFIG.TRAIN_PATH = getattr(data, "TRAIN_PATH", None) or os.path.join(
            os.path.dirname(CONFIG.TEST_PATH),
            os.path.basename(CONFIG.TEST_PATH).replace("_test", "_train"))

    def default_dev():              # lazy: probes the GPUs only if a device is unset
        return G.device() if G is not None else "cuda:0"
    if CONFIG.ROUTER_DEVICE is None:
        CONFIG.ROUTER_DEVICE = getattr(hw, "ROUTER_DEVICE", None) or default_dev()
    if CONFIG.AGENT_DEVICE is None:
        CONFIG.AGENT_DEVICE = getattr(hw, "AGENT_DEVICE", None) or default_dev()
    if CONFIG.VRAM_GIB is None:
        CONFIG.VRAM_GIB = getattr(C, "VRAM_GIB", getattr(
            hw, "VRAM_GIB", getattr(G, "VRAM_GIB", None)))
    if CONFIG.ROLLOUT_BATCH is None:
        CONFIG.ROLLOUT_BATCH = getattr(hw, "EVAL_BATCH", getattr(G, "EVAL_BATCH", 24))
    if CONFIG.OUT_DIR is None:
        CONFIG.OUT_DIR = (C.path("inf") if C is not None and hasattr(C, "path")
                          else os.path.join(_HERE, "results_inf"))


_resolve_config()


def env_knob(name, default=None):
    """ENV setting from whichever place this folder keeps it."""
    if hasattr(env, "ENV_CFG") and hasattr(env.ENV_CFG, name):
        return getattr(env.ENV_CFG, name)
    if M.config is not None and hasattr(getattr(M.config, "ENV", None), name):
        return getattr(M.config.ENV, name)
    return default


def model_names():
    """(router, agent) base model ids as this folder configures them."""
    def pick(mod, attrs, group):
        for a in attrs:
            cfg = getattr(mod, a, None)
            if cfg is not None and hasattr(cfg, "MODEL"):
                return cfg.MODEL
        g = getattr(M.config, group, None) if M.config is not None else None
        return getattr(g, "MODEL", "?")
    return (pick(M.router_mod, ("OPD_CFG", "ROUTER_CFG"), "ROUTER"),
            pick(rl, ("RL_CFG",), "AGENT"))


def folder_style():
    return "{} / router={}{}".format(
        "config.py ablation" if M.config is not None else "flat trainer",
        M.router_mod.__name__.split(".")[-1],
        " / package={}".format(M.prefix) if M.prefix else "")


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def _adapter_dir(parent, name):
    for d in (os.path.join(parent, name), os.path.join(parent, name + ".old")):
        if os.path.isfile(os.path.join(d, "adapter_config.json")):
            return d
    return None


def find_checkpoint():
    """(router_dir, agent_dir, where) of the final trained checkpoint, or None."""
    cands = []
    if CONFIG.CKPT_PATH:
        cands.append(("CONFIG.CKPT_PATH", os.path.join(_HERE, CONFIG.CKPT_PATH)))
    else:
        ck = os.path.join(_HERE, "ckpt")
        # flat trainer with committed checkpoints: latest/ (state.json written last),
        # latest.old/ if a crash landed mid-swap
        for d in (os.path.join(ck, "latest"), os.path.join(ck, "latest.old")):
            if os.path.isfile(os.path.join(d, "state.json")):
                cands.append(("ckpt/latest (committed)", d))
        if M.config is not None and hasattr(M.config, "path"):
            cands.append(("config RUN.NAME={}".format(M.config.RUN.NAME),
                          M.config.path("ckpt")))
        cands.append(("ckpt/", ck))
        runs = os.path.join(_HERE, "runs")
        if os.path.isdir(runs):
            found = [os.path.join(runs, r, "ckpt") for r in os.listdir(runs)]
            found = [d for d in found if os.path.isdir(d)]
            found.sort(key=os.path.getmtime, reverse=True)
            cands += [("runs/{} (newest first)".format(
                os.path.basename(os.path.dirname(d))), d) for d in found]

    for where, d in cands:
        r, a = _adapter_dir(d, "router"), _adapter_dir(d, "agent")
        if r and a:
            return r, a, where
    return None, None, cands


def _fingerprint(r_ad, a_ad, use_r, use_a):
    def stamp(d):
        return sorted((n, os.path.getsize(os.path.join(d, n)))
                      for n in os.listdir(d) if n.startswith("adapter_model"))
    st = os.stat(CONFIG.TEST_PATH)
    return {"test": [os.path.basename(CONFIG.TEST_PATH), st.st_size, int(st.st_mtime)],
            "n": CONFIG.N_SCENARIOS,
            "router": [r_ad, stamp(r_ad)] if use_r else None,
            "agent": [a_ad, stamp(a_ad)] if use_a else None}


# ============================================================
# DEVICES
# ============================================================

def _check_device(d, log):
    d = str(d)
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("[FAIL] {} requested but CUDA is unavailable".format(d))
        idx = int(d.split(":")[1]) if ":" in d else 0
        n = torch.cuda.device_count()
        if idx >= n:
            log("  [warn] {} requested but only {} CUDA device(s) visible -> cuda:0"
                .format(d, n))
            d = "cuda:0"
    return d


def apply_devices(log):
    r_dev = _check_device(CONFIG.ROUTER_DEVICE, log)
    a_dev = _check_device(CONFIG.AGENT_DEVICE, log)
    # flat trainers read <module>_CFG.DEVICE; config.py ablations read HW.*_DEVICE
    for mod, dev in ((M.router_mod, r_dev), (rl, a_dev)):
        for attr in ("OPD_CFG", "ROUTER_CFG", "RL_CFG"):
            cfg = getattr(mod, attr, None)
            if cfg is not None and hasattr(cfg, "DEVICE"):
                cfg.DEVICE = dev
    if M.config is not None and hasattr(M.config, "HW"):
        M.config.HW.ROUTER_DEVICE = r_dev
        M.config.HW.AGENT_DEVICE = a_dev
    if M.config is not None and hasattr(M.config, "VRAM_GIB"):
        M.config.VRAM_GIB = CONFIG.VRAM_GIB      # the model constructors apply it
    elif CONFIG.VRAM_GIB and M.gpu is not None:
        for d in {r_dev, a_dev}:
            M.gpu.cap_vram(d, CONFIG.VRAM_GIB)
    elif CONFIG.VRAM_GIB:
        g = 1024.0 ** 3
        for d in {r_dev, a_dev}:
            if d.startswith("cuda"):
                idx = torch.device(d).index or 0
                total = torch.cuda.get_device_properties(idx).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, CONFIG.VRAM_GIB * g / total), idx)
    log("  vram cap  : {}".format(
        "{} GiB per card".format(CONFIG.VRAM_GIB) if CONFIG.VRAM_GIB else "none"))
    # anything that allocates on the bare "cuda" device lands on the router's card
    if r_dev.startswith("cuda"):
        torch.cuda.set_device(torch.device(r_dev))
    log("  devices   : router -> {}   agent -> {}   ({} CUDA device(s) visible)".format(
        r_dev, a_dev, torch.cuda.device_count() if torch.cuda.is_available() else 0))
    return r_dev, a_dev


def report_placement(log, router, agent, r_dev, a_dev):
    """Where every parameter actually lives. Aborts if it is not where CONFIG says."""
    log("\n  DEVICES (actual parameter placement)")
    ok = True
    for name, lm, want in (("router", router, r_dev), ("agent", agent, a_dev)):
        model = getattr(lm, "model", None)
        counts = {}
        if model is not None:
            for p in model.parameters():
                k = str(p.device)
                counts[k] = counts.get(k, 0) + p.numel()
        total = sum(counts.values()) or 1
        where = ", ".join("{} {:.1f}% ({:.2f}B)".format(k, 100.0 * v / total, v / 1e9)
                          for k, v in sorted(counts.items()))
        want_t = torch.device(want)
        good = all(torch.device(k).type == want_t.type
                   and (want_t.index is None or torch.device(k).index in (want_t.index, None)
                        if want_t.type == "cuda" else True)
                   for k in counts)
        ok &= good
        log("    {:<7s} wanted {:<7s} -> {}   {}".format(
            name, want, where or "?", "OK" if good else "MISMATCH"))
    if torch.cuda.is_available():
        g = 1024.0 ** 3
        for i in range(torch.cuda.device_count()):
            free, tot = torch.cuda.mem_get_info(i)
            log("    cuda:{}  {} | allocated by this process {:.1f} GiB | free {:.1f} of "
                "{:.1f} GiB".format(i, torch.cuda.get_device_name(i),
                                    torch.cuda.memory_allocated(i) / g, free / g, tot / g))
    if not ok:
        raise SystemExit("[FAIL] model weights are not on the configured device(s)")


# ============================================================
# DATA
# ============================================================

def _train_domains():
    """Domains of the training file, or None if it cannot be read."""
    p = CONFIG.TRAIN_PATH
    if not p or not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        return {s.get("domain", "") for s in json.load(f)}


def load_test(path):
    """uid = the file's own uid when it has one, else the row position - the same rule
    env.load_scenarios uses, so a uid means the SAME scenario here as it does in training.
    (1100_test.json carries uids from the pooled 1100 set, which are unique against
    1100_train.json; 550_test.json has none, so it falls back to position.)

    eval_group rides along as an attribute: the file's tag if it has one, else seen/unseen
    by the training file's domains. Returns (scenarios, n_invalid, how eval_group was set).
    """
    if not os.path.isfile(path):
        raise SystemExit("[FAIL] {} not found - put the test json in data/".format(path))
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    tagged = all("eval_group" in s for s in raw)
    doms = None if tagged else _train_domains()
    if tagged:
        how = "from the file's eval_group tags"
    elif doms is not None:
        how = "derived: domain in {} -> seen, else unseen".format(
            os.path.basename(CONFIG.TRAIN_PATH))
    else:
        how = "UNKNOWN (no eval_group tags and no {})".format(CONFIG.TRAIN_PATH)
    known = env.Scenario.__dataclass_fields__.keys()
    scen = []
    for i, s in enumerate(raw):
        obj = env.Scenario(**{k: v for k, v in s.items() if k in known})
        if not isinstance(s.get("uid"), int):
            obj.uid = i
        if tagged:
            obj.eval_group = s["eval_group"]
        elif doms is not None:
            obj.eval_group = "seen" if s.get("domain", "") in doms else "unseen"
        else:
            obj.eval_group = "unknown"
        scen.append(obj)
    # Every per-scenario CSV this script writes is keyed by uid, so a duplicate would
    # silently overwrite a row and quietly shrink the analysis instead of failing.
    uids = [s.uid for s in scen]
    if len(set(uids)) != len(uids):
        raise SystemExit("[FAIL] {}: duplicate uids".format(path))
    valid = [s for s in scen if env._valid_scenario(s)]
    n_bad = len(scen) - len(valid)
    if CONFIG.N_SCENARIOS:
        valid = valid[:CONFIG.N_SCENARIOS]
    return valid, n_bad, how


def assert_blind(scenarios, log):
    """The student prompt must not contain any private fact text."""
    prof = prompts.PROFILES["student"]
    params = list(inspect.signature(prof.route_prompt).parameters)
    t_max = env_knob("T_MAX", 12)
    per_agent = env_knob("MAX_TURNS_PER_AGENT", 3)
    leaks = checked = 0
    for sc in scenarios:
        bud = {a: per_agent for a in sc.agent_ids}
        args = [sc, "", [], set(), bud, 1, t_max]
        if "insight" not in params:
            args.pop(1)
        blind = prof.route_prompt(*args)
        for v in sc.private_facts.values():
            h = v.get("text", "")
            if h:
                checked += 1
                leaks += h in blind
    if leaks:
        raise SystemExit("[FAIL] eval profile leaks {}/{} private facts".format(leaks, checked))
    log("  [ok] eval profile is blind: 0/{} private facts visible across {} scenarios"
        .format(checked, len(scenarios)))


# ============================================================
# HELPERS
# ============================================================

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _num(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return float(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        return {"true": 1.0, "false": 0.0}.get(s)
    return None if x != x else x


def _mean(vals):
    vals = [x for x in (_num(v) for v in vals) if x is not None]
    return (sum(vals) / len(vals)) if vals else float("nan")


def _f(x, w=10, p=4, sign=False):
    if x is None or x != x:
        return "{:>{w}s}".format("-", w=w)
    return "{:>{s}{w}.{p}f}".format(x, w=w, p=p, s="+" if sign else "")


def _clip(v, n):
    s = "" if v is None else str(v)
    return s if len(s) <= n else s[:n] + "...[clipped]"


def metric_list():
    """env.METRICS de-duplicated; metrics this folder never produces are dropped later."""
    seen, out = set(), []
    for k, hi in env.METRICS:
        if k not in seen:
            seen.add(k)
            out.append((k, hi))
    return out


def bootstrap_paired(d, n_boot, seed=0):
    rng = random.Random(seed)
    n = len(d)
    if not n:
        return (float("nan"),) * 4
    obs = sum(d) / n
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot), n_boot - 1)]
    p = sum(1 for m in means if abs(m - obs) >= abs(obs)) / n_boot
    return obs, lo, hi, p


def bootstrap_unpaired(a, b, n_boot, seed=0):
    """mean(b) - mean(a) for two independent samples (seen vs unseen are different
    scenarios, so they cannot be paired)."""
    rng = random.Random(seed)
    if not a or not b:
        return (float("nan"),) * 4
    obs = sum(b) / len(b) - sum(a) / len(a)
    diffs = []
    for _ in range(n_boot):
        ma = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
        mb = sum(b[rng.randrange(len(b))] for _ in b) / len(b)
        diffs.append(mb - ma)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[min(int(0.975 * n_boot), n_boot - 1)]
    p = sum(1 for d in diffs if abs(d - obs) >= abs(obs)) / n_boot
    return obs, lo, hi, p


# ============================================================
# ROLLOUTS
# ============================================================

def run_arm(scenarios, router, agent, tag, use_r, use_a, fp, log):
    out = CONFIG.OUT_DIR
    p_sum = os.path.join(out, "{}_scenarios.csv".format(tag))
    p_done = os.path.join(out, "{}_done.json".format(tag))
    if CONFIG.RESUME and os.path.isfile(p_done) and os.path.isfile(p_sum):
        with open(p_done, encoding="utf-8") as f:
            if json.load(f) == fp:
                log("\n  arm {}: already complete for this checkpoint -> reusing {}"
                    .format(tag, p_sum))
                return
    if os.path.isfile(p_done):
        os.remove(p_done)

    sum_fields = list(env.SUM_COLS) + ["eval_group"]
    fs = open(p_sum, "w", newline="", encoding="utf-8")
    ft = open(os.path.join(out, "{}_turns.csv".format(tag)), "w", newline="",
              encoding="utf-8")
    fj = open(os.path.join(out, "{}_episodes.jsonl".format(tag)), "w", encoding="utf-8")
    ws = csv.DictWriter(fs, fieldnames=sum_fields, quoting=csv.QUOTE_ALL,
                        extrasaction="ignore")
    wt = csv.DictWriter(ft, fieldnames=env.TURN_COLS, quoting=csv.QUOTE_ALL,
                        extrasaction="ignore")
    ws.writeheader()
    wt.writeheader()

    log("\n" + "=" * 92)
    log("ARM {}   router_lora={}  agent_lora={}".format(tag, use_r, use_a))
    log("=" * 92)

    rb_params = inspect.signature(env.run_batch).parameters
    kw = {"student_profile": prompts.PROFILES["student"],
          "record_router": False, "record_agent": False}
    if "teacher_profile" in rb_params:
        kw["teacher_profile"] = None             # NO privileged context at eval

    group_of = {s.uid: s.eval_group for s in scenarios}
    t0, done, skipped = time.time(), 0, 0
    for blk in range(0, len(scenarios), CONFIG.ROLLOUT_BATCH):
        block = scenarios[blk:blk + CONFIG.ROLLOUT_BATCH]
        r_ctx = contextlib.nullcontext() if use_r else router.base_mode()
        a_ctx = contextlib.nullcontext() if use_a else agent.ref_mode()
        try:
            with r_ctx, a_ctx:
                outs = env.run_batch(block, router, agent, **kw)
        except Exception as exc:                 # noqa: BLE001
            log("    [SKIP] uids={} {}: {}".format([x.uid for x in block],
                                                  type(exc).__name__, exc))
            done += len(block)
            skipped += len(block)
            continue

        for o in outs:
            done += 1
            sm = o["summary"]
            sm["eval_group"] = group_of.get(sm["uid"], "unknown")
            ws.writerow({k: _clip(sm.get(k), CONFIG.CSV_MAX_CHARS) for k in sum_fields})
            for r in o["rows"]:
                wt.writerow({k: _clip(r.get(k), CONFIG.CSV_MAX_CHARS)
                             for k in env.TURN_COLS})
            fj.write(json.dumps(o["episode"], ensure_ascii=False) + "\n")
            if CONFIG.VERBOSE:
                log("  [{}/{}] uid={} ({}, {})  checks {}/{} ({:.2f})  success={}  "
                    "reveals={}  settled={}  t_settle={}".format(
                        done, len(scenarios), sm["uid"], sm["eval_group"],
                        sm.get("domain", ""), sm.get("checks_passed"),
                        sm.get("checks_total"), _num(sm.get("checks_frac")) or 0.0,
                        sm.get("success"), sm.get("reveals"), sm.get("settled"),
                        sm.get("turns_to_settle") or "-"))
                if sm.get("route_sequence") is not None:
                    log("        route: {}".format(sm.get("route_sequence")))
        fs.flush(); ft.flush(); fj.flush()
        router.release()
        agent.release()

    fs.close(); ft.close(); fj.close()
    log("\n  arm {} done: {} episodes, {} skipped, {:.1f}s".format(
        tag, done - skipped, skipped, time.time() - t0))
    if not skipped:                               # only a clean arm is reusable
        with open(p_done, "w", encoding="utf-8") as f:
            json.dump(fp, f)


def load_rows(tag):
    p = os.path.join(CONFIG.OUT_DIR, "{}_scenarios.csv".format(tag))
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return {int(r["uid"]): r for r in csv.DictReader(f)}


# ============================================================
# ANALYSIS
# ============================================================

def summarize(log, title, rows):
    n = len(rows)
    log("\n" + "-" * 92)
    log("SUMMARY  {}   (n={})".format(title, n))
    log("-" * 92)
    if not n:
        return
    m = lambda k: _mean([r.get(k) for r in rows])  # noqa: E731
    st = [r for r in rows if _num(r.get("settled"))]
    cp = [r for r in st if _num(r.get("decisive_revealed")) == _num(r.get("decisive_total"))]
    cp_ids = {id(r) for r in cp}
    inc = [r for r in st if id(r) not in cp_ids]
    uns = [r for r in rows if not _num(r.get("settled"))]
    L = lambda label, val: log("    {:<34s}: {}".format(label, val))  # noqa: E731

    log("  CHECKS")
    L("content checks passed (mean)", "{:.3f} / {:.3f}".format(
        m("content_passed"), m("content_total")))
    L("provenance checks passed (mean)", "{:.3f} / {:.3f}".format(
        m("prov_passed"), m("prov_total")))
    L("BOTH: checks passed (mean)", "{:.3f} / {:.3f}".format(
        m("checks_passed"), m("checks_total")))
    L("checks_frac (mean fraction)", "{:.4f}".format(m("checks_frac")))
    L("success (ALL checks pass)", "{:.4f}".format(m("success")))
    log("  BEHAVIOUR")
    L("reveals (mean)", "{:.3f}".format(m("reveals")))
    L("decisive facts revealed", "{:.3f} / {:.3f}".format(
        m("decisive_revealed"), m("decisive_total")))
    L("decisive reveal fraction", "{:.4f}".format(_mean(
        [(_num(r.get("decisive_revealed")) or 0) / _num(r.get("decisive_total"))
         for r in rows if _num(r.get("decisive_total"))])))
    L("settled rate", "{:.4f}".format(m("settled")))
    L("turns to settle (mean, settled)", "{:.2f}".format(m("turns_to_settle")))
    L("turns used (mean)", "{:.2f}".format(m("turns_used")))
    log("  ROUTING")
    L("routing precision", "{:.4f}".format(m("routing_precision")))
    L("routing hits / opportunities", "{:.3f} / {:.3f}".format(
        m("routing_hits"), m("routing_opportunities")))
    L("router invalid picks (mean)", "{:.3f}".format(m("route_invalid")))
    L("forced hand-overs (last call)", "{:.3f}".format(m("forced_routes")))
    L("free routing decisions (mean)", "{:.3f}".format(m("route_decisions")))
    if any(_num(r.get("insight_slot_frac")) is not None for r in rows):
        log("  INSIGHT")
        L("schema slots named", "{:.4f}".format(m("insight_slot_frac")))
        L("PRIVATE leak = telepathy (max)", "{:.4f}   <- GUARDRAIL".format(
            m("insight_private_leak_max")))
    log("  REWARD (as training would score it)")
    L("train_reward (mean)", "{:.4f}".format(m("train_reward")))
    L("route_reward (mean)", "{:.4f}".format(m("route_reward")))
    L("wall time per episode (s)", "{:.2f}".format(m("wall_time")))
    log("  BOTTLENECK DECOMPOSITION")
    log("    unsettled episodes             : n={:<3d} checks_frac={:.4f}".format(
        len(uns), _mean([r.get("checks_frac") for r in uns])))
    log("    settled, record INCOMPLETE     : n={:<3d} checks_frac={:.4f}".format(
        len(inc), _mean([r.get("checks_frac") for r in inc])))
    log("    settled, record COMPLETE       : n={:<3d} checks_frac={:.4f}  success={:.4f}"
        .format(len(cp), _mean([r.get("checks_frac") for r in cp]),
                _mean([r.get("success") for r in cp])))


def group_uids(rows_any, uids):
    out = {g: [u for u in uids if rows_any[u].get("eval_group") == g] for g in GROUPS}
    out = {g: v for g, v in out.items() if v}
    out["total"] = list(uids)
    return out


def headline(log, rows, tags, groups, metrics, writer):
    log("\n" + "=" * 92)
    log("HEADLINE   mean per arm, per group   (* = better than {})".format(tags[0]))
    log("=" * 92)
    cols = [(g, t) for g in groups for t in tags]
    log("  {:<26s}".format("metric") + "".join(
        "{:>13s}".format((t[:5] + "/" + g[:6])) for g, t in cols))
    log("  " + "-" * (26 + 13 * len(cols)))
    for key, higher in metrics:
        line = "  {:<26s}".format(key)
        for g, t in cols:
            v = _mean([rows[t][u].get(key) for u in groups[g]])
            b = _mean([rows[tags[0]][u].get(key) for u in groups[g]])
            star = ""
            if t != tags[0] and v == v and b == b and v != b:
                star = "*" if (v > b) == higher else " "
            line += _f(v, 12) + (star or " ")
            writer.writerow({"arm": t, "group": g, "n": len(groups[g]),
                             "metric": key, "mean": "" if v != v else round(v, 6)})
        log(line)
    log("  n per group: " + ", ".join("{}={}".format(g, len(u)) for g, u in groups.items()))


def paired(log, rows, tags, groups, metrics, writer):
    base = tags[0]
    for other in tags[1:]:
        for g, uids in groups.items():
            log("\n" + "-" * 92)
            log("PAIRED BOOTSTRAP [{} n={}]   {} - {}   ({} resamples, 95% CI)".format(
                g.upper(), len(uids), other, base, CONFIG.N_BOOT))
            log("-" * 92)
            log("  {:<26s}{:>10s}{:>10s}{:>10s}{:>22s}{:>8s}".format(
                "metric", base[:10], other[:10], "delta", "95% CI", "p"))
            for key, higher in metrics:
                pr = [(_num(rows[base][u].get(key)), _num(rows[other][u].get(key)))
                      for u in uids]
                pr = [(x, y) for x, y in pr if x is not None and y is not None]
                if not pr:
                    continue
                obs, lo, hi, p = bootstrap_paired([y - x for x, y in pr], CONFIG.N_BOOT)
                mb = sum(x for x, _ in pr) / len(pr)
                mo = sum(y for _, y in pr) / len(pr)
                verdict = ""
                if p < 0.05 and obs != 0:
                    verdict = "  better" if (obs > 0) == higher else "  WORSE"
                log("  {:<26s}{}{}{}   [{:+.4f}, {:+.4f}]{:>8.4f}{}".format(
                    key, _f(mb), _f(mo), _f(obs, sign=True), lo, hi, p, verdict))
                writer.writerow({"group": g, "n": len(pr), "metric": key,
                                 "baseline": base, "arm": other,
                                 "baseline_mean": round(mb, 6), "arm_mean": round(mo, 6),
                                 "delta": round(obs, 6), "ci_lo": round(lo, 6),
                                 "ci_hi": round(hi, 6), "p": round(p, 4),
                                 "higher_is_better": higher})
            log("  better/WORSE = p < 0.05 in that direction. A CI straddling zero means "
                "no detectable difference at this n.")


def generalisation_gap(log, rows, tags, groups, writer):
    if "seen" not in groups or "unseen" not in groups:
        return
    log("\n" + "=" * 92)
    log("GENERALISATION GAP   unseen - seen   (unpaired bootstrap: different scenarios)")
    log("=" * 92)
    log("  {:<22s}{:<12s}{:>9s}{:>9s}{:>10s}{:>22s}{:>8s}".format(
        "metric", "arm", "seen", "unseen", "gap", "95% CI", "p"))
    for key in KEY_METRICS:
        gaps = {}
        for t in tags:
            a = [x for x in (_num(rows[t][u].get(key)) for u in groups["seen"]) if x is not None]
            b = [x for x in (_num(rows[t][u].get(key)) for u in groups["unseen"]) if x is not None]
            if not a or not b:
                continue
            obs, lo, hi, p = bootstrap_unpaired(a, b, CONFIG.N_BOOT)
            gaps[t] = obs
            log("  {:<22s}{:<12s}{}{}{}   [{:+.4f}, {:+.4f}]{:>8.4f}".format(
                key, t, _f(sum(a) / len(a), 9), _f(sum(b) / len(b), 9),
                _f(obs, 10, sign=True), lo, hi, p))
            writer.writerow({"metric": key, "arm": t, "seen": round(sum(a) / len(a), 6),
                             "unseen": round(sum(b) / len(b), 6), "gap": round(obs, 6),
                             "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "p": round(p, 4)})
        base = tags[0]
        for t in tags[1:]:
            if t in gaps and base in gaps:
                log("  {:<22s}{:<12s}{:>38s}{}".format(
                    "", "", "gap change ({} - {}):".format(t, base),
                    _f(gaps[t] - gaps[base], 10, sign=True)))
    log("  gain transfer: for each arm pair, compare the paired SEEN delta with the paired")
    log("  UNSEEN delta above - a gain that holds on unseen domains is not domain memorisation.")


# Counts behind the ratios in env.METRICS, reported next to them per domain.
DOMAIN_EXTRA = [("content_total", True), ("prov_total", True), ("checks_total", True),
                ("decisive_total", True), ("routing_hits", True),
                ("routing_opportunities", True), ("forced_routes", False),
                ("route_decisions", True), ("route_reward", True), ("wall_time", False)]
OVERVIEW_COLS = ["checks_frac", "success", "settled", "turns_used", "routing_precision"]


def domain_table(log, rows, tags, uids, metrics, writer):
    """Per (group, domain): a compact overview, then EVERY metric for every arm with the
    paired delta vs the baseline arm, its 95% CI and p. Per-domain n is small (550D seen
    10, unseen 50; 1100D 20), so those CIs are wide - read them as descriptive."""
    base = tags[0]
    dom = {}
    for u in uids:
        r = rows[base][u]
        dom.setdefault((r.get("eval_group", ""), r.get("domain", "")), []).append(u)
    have = {k for k, _ in metrics}
    full = list(metrics) + [(k, h) for k, h in DOMAIN_EXTRA if k not in have and any(
        _num(rows[base][u].get(k)) is not None for u in uids)]
    arrow = " -> ".join(t[:9] for t in tags)

    # ---- overview ----
    log("\n" + "=" * 92)
    log("PER-DOMAIN OVERVIEW   (each cell: {})".format(arrow))
    log("=" * 92)
    log("  {:<8s}{:<30s}{:>4s}".format("group", "domain", "n") +
        "".join("{:>19s}".format(c[:17]) for c in OVERVIEW_COLS))
    for (g, d) in sorted(dom):
        us = dom[(g, d)]
        line = "  {:<8s}{:<30s}{:>4d}".format(g, d[:29], len(us))
        for c in OVERVIEW_COLS:
            vals = [_mean([rows[t][u].get(c) for u in us]) for t in tags]
            p = 2 if c == "turns_used" else 3
            line += "{:>19s}".format(" -> ".join(
                "-" if v != v else "{:.{p}f}".format(v, p=p) for v in vals))
        log(line)

    # ---- full detail, one block per domain ----
    log("\n" + "=" * 92)
    log("PER-DOMAIN DETAIL   every metric, paired {} - {}   ({} resamples, 95% CI)".format(
        " / ".join(tags[1:]), base, CONFIG.N_BOOT))
    log("=" * 92)
    for (g, d) in sorted(dom):
        us = dom[(g, d)]
        log("\n  [{}] {}   n={}".format(g.upper(), d, len(us)))
        log("  {:<26s}".format("metric") + "".join("{:>12s}".format(t[:11]) for t in tags)
            + "".join("{:>10s}{:>20s}{:>8s}".format("delta", "95% CI", "p")
                      for _ in tags[1:]))
        for key, higher in full:
            vals = {t: [_num(rows[t][u].get(key)) for u in us] for t in tags}
            means = {t: _mean(v) for t, v in vals.items()}
            if all(m != m for m in means.values()):
                continue
            line = "  {:<26s}".format(key) + "".join(_f(means[t], 12) for t in tags)
            for t in tags[1:]:
                pr = [(x, y) for x, y in zip(vals[base], vals[t])
                      if x is not None and y is not None]
                if pr:
                    obs, lo, hi, pv = bootstrap_paired([y - x for x, y in pr], CONFIG.N_BOOT)
                    mark = ""
                    if pv < 0.05 and obs != 0:
                        mark = " +" if (obs > 0) == higher else " -"
                    line += "{}   [{:+.3f}, {:+.3f}]{:>8.3f}{}".format(
                        _f(obs, 10, sign=True), lo, hi, pv, mark)
                else:
                    obs = lo = hi = pv = float("nan")
                    line += "{:>38s}".format("-")
                writer.writerow({
                    "group": g, "domain": d, "n": len(us), "metric": key,
                    "higher_is_better": higher, "baseline": base, "arm": t,
                    "baseline_mean": "" if means[base] != means[base] else round(means[base], 6),
                    "arm_mean": "" if means[t] != means[t] else round(means[t], 6),
                    "delta": "" if obs != obs else round(obs, 6),
                    "ci_lo": "" if lo != lo else round(lo, 6),
                    "ci_hi": "" if hi != hi else round(hi, 6),
                    "p": "" if pv != pv else round(pv, 4)})
            log(line)
    log("\n  + / - = p < 0.05, better / worse than {}. Per-domain n is small (550D seen: 10,"
        .format(base))
    log("  1100D: 20): read those deltas as descriptive; only a large, consistent effect is")
    log("  detectable there.")


def guardrail(log, rows, tags, groups):
    lk = "insight_private_leak_max"
    if all(_num(rows[tags[0]][u].get(lk)) is None for u in groups["total"]):
        log("\n  GUARDRAIL: no insight head in this folder - telepathy check not applicable")
        return
    log("\n" + "=" * 92)
    log("GUARDRAIL   insight private leak (telepathy detector)")
    log("=" * 92)
    base = tags[0]
    for g, uids in groups.items():
        bl = _mean([rows[base][u].get(lk) for u in uids])
        for t in tags[1:]:
            al = _mean([rows[t][u].get(lk) for u in uids])
            verdict = ("OK" if al <= bl * 1.5 + 0.01 else
                       "SUSPECT - the insight head may be fabricating hidden values")
            log("  {:<8s}{:<12s} {:.4f}  vs  {} {:.4f}   -> {}".format(
                g, t, al, base, bl, verdict))


def _csv(name, fields):
    f = open(os.path.join(CONFIG.OUT_DIR, name), "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def analyse(log, tags):
    rows = {t: load_rows(t) for t in tags}
    missing = [t for t, r in rows.items() if r is None]
    if missing:
        log("\n[analysis] missing results for arm(s): {}".format(missing))
        return
    uids = set(rows[tags[0]])
    for t in tags:
        uids &= set(rows[t])
    uids = sorted(uids)
    if not uids:
        log("\n[analysis] no scenario completed in every arm")
        return
    dropped = {t: len(rows[t]) - len(uids) for t in tags if len(rows[t]) != len(uids)}
    if dropped:
        log("\n  [warn] analysis restricted to {} scenarios completed by every arm "
            "(extra rows ignored: {})".format(len(uids), dropped))

    groups = group_uids(rows[tags[0]], uids)
    metrics = [(k, h) for k, h in metric_list()
               if any(_num(rows[tags[0]][u].get(k)) is not None for u in uids)]

    for t in tags:
        log("\n" + "#" * 92)
        log("ARM {}   -- seen / unseen / total --".format(t))
        log("#" * 92)
        for g, us in groups.items():
            summarize(log, "{} / {}".format(t, g.upper()), [rows[t][u] for u in us])

    fh, wh = _csv("metrics_by_group.csv", ["arm", "group", "n", "metric", "mean"])
    headline(log, rows, tags, groups, metrics, wh)
    fh.close()

    if len(tags) > 1:
        fp, wp = _csv("paired_comparison.csv",
                      ["group", "n", "metric", "baseline", "arm", "baseline_mean",
                       "arm_mean", "delta", "ci_lo", "ci_hi", "p", "higher_is_better"])
        paired(log, rows, tags, groups, metrics, wp)
        fp.close()

    fg, wg = _csv("generalisation_gap.csv",
                  ["metric", "arm", "seen", "unseen", "gap", "ci_lo", "ci_hi", "p"])
    generalisation_gap(log, rows, tags, groups, wg)
    fg.close()

    fd, wd = _csv("metrics_by_domain.csv",
                  ["group", "domain", "n", "metric", "higher_is_better", "baseline", "arm",
                   "baseline_mean", "arm_mean", "delta", "ci_lo", "ci_hi", "p"])
    domain_table(log, rows, tags, uids, metrics, wd)
    fd.close()

    if len(tags) > 1:
        guardrail(log, rows, tags, groups)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    log = Tee(os.path.join(CONFIG.OUT_DIR, "inf.log"))
    random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)

    scenarios, n_bad, how = load_test(CONFIG.TEST_PATH)
    counts = {g: sum(1 for s in scenarios if s.eval_group == g) for g in GROUPS}

    log("=" * 92)
    log("INFERENCE EVAL   vanilla vs co-evolved   (held-out set, seen/unseen, blind)")
    log("=" * 92)
    log("  folder    : {}".format(_HERE))
    log("  style     : {}".format(folder_style()))
    log("  models    : router={}   agent={}".format(*model_names()))
    log("  test set  : {}  -> {} scenarios ({})  {}".format(
        CONFIG.TEST_PATH, len(scenarios),
        ", ".join("{} {}".format(v, k) for k, v in counts.items()),
        "[{} invalid dropped]".format(n_bad) if n_bad else ""))
    log("  groups    : {}".format(how))
    log("  seen      : trained domains, unseen episodes: {}".format(
        ", ".join(sorted({s.domain for s in scenarios if s.eval_group == "seen"})) or "-"))
    log("  unseen    : domains never trained on: {}".format(
        ", ".join(sorted({s.domain for s in scenarios if s.eval_group == "unseen"})) or "-"))
    log("  arms      : {}".format(", ".join(t for t, _, _ in CONFIG.ARMS)))
    log("  last call : complete={}  final_turn={}  (applies to every arm)".format(
        env_knob("FORCE_DM_WHEN_COMPLETE"), env_knob("FORCE_DM_ON_LAST_TURN")))
    log("  batch     : {} episodes per generate() call".format(CONFIG.ROLLOUT_BATCH))
    r_dev, a_dev = apply_devices(log)

    r_ad, a_ad, where = find_checkpoint()
    if r_ad is None:
        log("\n  [FAIL] no trained checkpoint (router/ + agent/ with adapter_config.json). "
            "Looked in:")
        for w, d in where:
            log("           {:<32s} {}".format(w, d))
        log("         set CONFIG.CKPT_PATH to point at one.")
        log.close()
        sys.exit(1)
    log("  checkpoint: {}".format(where))
    log("    router  : {}".format(r_ad))
    log("    agent   : {}".format(a_ad))

    assert_blind(scenarios, log)
    log("=" * 92)

    router = M.router_mod.RouterLM(adapter_path=r_ad, train=False)
    agent = rl.AgentLM(adapter_path=a_ad, train=False)
    report_placement(log, router, agent, r_dev, a_dev)

    for tag, use_r, use_a in CONFIG.ARMS:
        run_arm(scenarios, router, agent, tag, use_r, use_a,
                _fingerprint(r_ad, a_ad, use_r, use_a), log)

    analyse(log, [t for t, _, _ in CONFIG.ARMS])

    out = CONFIG.OUT_DIR
    log("\n" + "=" * 92)
    log("  log                : {}".format(os.path.join(out, "inf.log")))
    log("  headline table     : {}".format(os.path.join(out, "metrics_by_group.csv")))
    log("  paired comparison  : {}".format(os.path.join(out, "paired_comparison.csv")))
    log("  generalisation gap : {}".format(os.path.join(out, "generalisation_gap.csv")))
    log("  per-domain         : {}".format(os.path.join(out, "metrics_by_domain.csv")))
    log("  per-scenario       : {}/<arm>_scenarios.csv  (eval_group column)".format(out))
    log("  per-turn / episodes: {}/<arm>_turns.csv, <arm>_episodes.jsonl".format(out))
    log("=" * 92)
    log.close()


if __name__ == "__main__":
    main()
