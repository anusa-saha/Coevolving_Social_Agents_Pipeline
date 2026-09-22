"""main.py - alternating co-evolution. Run this; it does the whole training.

    python main.py

No flags, no arguments, no environment variables. Every knob is CONFIG below, or the
module configs it overwrites at startup.


WHAT THIS TRAINS, AND WITH WHAT

    router.next_agent   REINFORCE + GRPO baseline      the only real decision in the
                                                       system: `settled | DM-routed` was
                                                       1.000 with no exceptions in V1
    router.insight      on-policy distillation         a description, not an action, so
                        from a privileged teacher      a privileged teacher can teach it
    agent  text         REINFORCE + GRPO baseline      the settlement and what gets said;
                                                       0.735 content pass even with a
                                                       COMPLETE record is a generation
                                                       problem, not an information one

The agents' ACTION keyword is not trained by anything, because it is not a decision: 61%
of V1's turns offered exactly one legal action, and of the 1,000 that offered `reveal` the
agent took it 1,000 times.


THE ASYMMETRIC SPLIT, AFTER TTPO (arXiv:2608.27448)

Within each group of G rollouts the two branches take different halves:

    A_i >  0   ->  insight distillation  (dense, token-weighted, privileged teacher)
    A_i <= 0   ->  route / text RL       (sparse, token-masked)

TTPO's motivation is pseudo-label noise, which we do not have - env.TerminalVerifier gives
an exact reward. Ours is stronger: a privileged teacher is a fair target for a rollout that
already did well, because the record it is looking at is one this student produced, and a
bad target for a failed rollout, where "what would someone who knew every secret have done"
is exactly the signal that collapsed onlyOPD (reveal rate 0.832 -> 0.134, success -> 0.000).

Token-level selection is taken wholesale, because it fixes two failures we can point at:
distillation weighting stops the insight KL grinding away at already-mastered JSON skeleton
(check_one drove its KL 0.076 -> 0.055 over ~90 steps while checks fell 0.351 -> 0.135),
and negative-sample masking stops one bad settlement penalising its own correct
punctuation. See ttpo.py.

The distillation branch also moved to FORWARD KL, per TTPO's positive-branch ablation and
our own coverage number: check_one's insight named 0.984 of the schema slots but covered
only 0.191 of the facts actually on the record, which is the reverse-KL mode-collapse
signature. opd.OPD_CFG.KL_DIRECTION = "reverse" restores the old behaviour exactly.


WHY ALTERNATING RATHER THAN SIMULTANEOUS

train_V1 updated both models against the same rollouts every step, so each was a moving
target for the other and neither faced a stationary problem. Its own decomposition came
out sub-additive (router alone +0.0127, agents alone -0.0045, co-evolved +0.0017, i.e.
interaction -0.0066: they interfered).

Alternating best-response fixes that and costs NOTHING extra, because the rounds
PARTITION the 720 training scenarios - four rounds of 180 rather than one pass of 720. Each
phase is a stationary MDP for the learner that is moving, and per-round checkpoints let
you evaluate the cross-product (router_i vs agent_j) afterwards, which is the standard
evidence that co-evolution is real: the diagonal should beat the off-diagonal.


COST, AND WHERE IT WENT

    720 scenarios  x  GROUP_SIZE rollouts  =  2880 episodes at the default G=4.
    (train_450v2_big: 360 x 4 = 1440 - same method, twice the data.)

Group methods cost G times a single-rollout pass; that is the price of a baseline that
actually cancels scenario difficulty (84.2% of reward variance, measured).

The first V2 run measured 152 seconds PER EPISODE and projected 61 hours. That was not
the group size - it was batch-1 decoding. A turn issues three generate() calls (route,
action, insight); at batch 1 autoregressive decode is entirely memory-bandwidth bound,
so the weights are streamed once per token to serve a single sequence. Running episodes
in LOCKSTEP - all of them take their route step together, then their action step
together - turns N x 21 sequential calls into 21 batched ones for almost the same wall
time. env.run_batch is now the only episode loop in the codebase, and it is verified
bit-identical to the sequential path at every chunk size.

Three more things pay the group cost back: episodes are far shorter now that LAST CALL
stops meetings idling to the turn cap (V1 averaged 8.9 turns with turns_to_all_decisive
at ~4.5), and V2 takes no rollout-time logprob passes where V1 took two per agent turn.
On this build the router and agent share one card, capped at 60 GiB (see gpu.py);
train_V2_grpo's two-card placement is a hardware fact of that run, not a requirement.

Levers, in the order worth pulling:
    SCENARIOS_PER_STEP   raise it while nvidia-smi shows headroom - the real lockstep
                         batch is SCENARIOS_PER_STEP * GROUP_SIZE, and ROLLOUT_BATCH can
                         never push it past that (see CONFIG). Changes the optimiser
                         step schedule (fewer, bigger steps); rescale ROUTER_WARMUP to
                         hold the same warmup FRACTION when you change this.
    ROLLOUT_BATCH        a pure cap on top of the above; raise it only alongside
                         SCENARIOS_PER_STEP, never past SCENARIOS_PER_STEP * GROUP_SIZE.
    GROUP_SIZE           3 still gives a usable leave-one-out baseline; below 3 it does
                         not. This is part of the algorithm, not a throughput knob.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
import warnings

# torch.utils.checkpoint calls the deprecated torch.cpu.amp.autocast internally. It is
# torch's own call, not ours, there is nothing to fix on this side, and it prints on
# every checkpointed backward. Silence exactly that one message and nothing else.
warnings.filterwarnings(
    "ignore", message=r".*torch\.cpu\.amp\.autocast.*", category=FutureWarning)

# Must be set BEFORE torch initialises CUDA. The V1 run died of allocator fragmentation
# between rollout (many small KV-cache blocks) and update (a few large activation blocks);
# expandable segments is the cheapest fix for exactly that shape.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Also before torch: without this, torch orders devices FASTEST_FIRST, which need not match
# nvidia-smi's PCI order - so gpu.GPU = "cuda:1" could select a different physical card here
# than in inf.py, which pins the same thing. On a shared box that is the difference between
# the idle card and someone else's job.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

import env
import gpu
import opd
import prompts
import resume
import rl
import ttpo

_HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONFIG
# ============================================================

class CONFIG:
    # ---- data ------------------------------------------------------------
    # 1100D, pre-split with 2 domains HELD OUT ENTIRELY (the 550D/1100D methodology, not
    # 450D's plain stratified split). The split lives in these files, not a runtime
    # shuffle; each scenario carries its index in the pooled 1100-scenario set as `uid`,
    # so ids are unique and traceable across the two files. See data/prepare_data.py,
    # which regenerates them from ICLR/1100D/ and hard-fails on any train/test overlap.
    #
    #   1100_train.json  720 = 9 trained domains x 80
    #   1100_test.json   380 = 180 eval_group="seen"   (those 9 domains' held-out 20%)
    #                        + 200 eval_group="unseen" (friends_family_informal and
    #                          informal_commerce_bargaining: 100 each, never trained on)
    #
    # The unseen half is the point of this split: a gain that holds there is not domain
    # memorisation. inf.py reports seen / unseen / total separately.
    TRAIN_PATH = os.path.join(_HERE, "data", "1100_train.json")  # 720 = 9 x 80
    TEST_PATH = os.path.join(_HERE, "data", "1100_test.json")    # 380, held out
    N_SCENARIOS = 0                 # 0 = the whole train split (720)
    SEED = 42

    # ---- schedule --------------------------------------------------------
    # One epoch over 720 scenarios, PARTITIONED across the phases below. Four phases ->
    # 180 distinct scenarios each; the router is updated in two of them and the agents in
    # the other two. Total rollouts = 720 * GROUP_SIZE regardless of how they are split.
    PHASES = ["router", "agent", "router", "agent"]
    GROUP_SIZE = 4                  # rollouts per scenario (mirrors rl.RL_CFG) - UNCHANGED,
                                    # this is part of the algorithm (the GRPO group), not a
                                    # throughput knob.
    # groups accumulated per optimiser step, i.e. the REAL number of scenarios run in
    # lockstep per generate() call is SCENARIOS_PER_STEP * GROUP_SIZE (see ROLLOUT_BATCH
    # below - it can only ever be a throughput cap ON TOP of this, never raise it past it).
    # 8, unchanged from train_450v2_big: at 2, main.py never ran more than 8 episodes
    # concurrently regardless of ROLLOUT_BATCH, so the card sat mostly idle during training
    # (eval, which has no such cap, used it fine). GROUP_SIZE, the reward, the estimator
    # and every learning rate are untouched. The phase is bigger here only because the
    # dataset is: 180 scenarios per phase instead of 90.
    SCENARIOS_PER_STEP = 8          # -> 22 steps per phase (44 router + 44 agent opt steps)

    # ---- throughput ------------------------------------------------------
    # Episodes in flight at once. Every one of them takes its route step together, then
    # its action step together, so a turn costs ONE batched generate() instead of N.
    # Autoregressive decode is memory-bandwidth bound at these model sizes: the weights
    # are streamed once per token however many sequences ride along, so this is very
    # close to free throughput up to the point where the KV cache stops fitting.
    #
    # This is a pure THROUGHPUT cap - env.run_batch is verified bit-identical to the
    # sequential path at every chunk size - but the loop below takes
    # min(ROLLOUT_BATCH // GROUP_SIZE, SCENARIOS_PER_STEP) scenarios per lockstep call, so
    # it can never exceed SCENARIOS_PER_STEP * GROUP_SIZE (=32) in practice; set equal to
    # that product, not above it, so there is no dead headroom implied by a bigger number.
    # Drop it on an allocator error; if the card still shows headroom at 32, the lever to
    # pull next is SCENARIOS_PER_STEP again, not this.
    ROLLOUT_BATCH = 32
    CHUNK_RETRIES = 1                # extra attempts for a rollout chunk before it's skipped
    MAX_CONSEC_FAILS = 6             # consecutive skipped chunks -> abort (bad CUDA state)

    # ---- optimisation ----------------------------------------------------
    # ONE optimiser for the router, because both its heads share ONE LoRA. Two optimisers
    # over the same parameters would each carry stale Adam moments for the other's steps.
    # 2e-5 is check/OPD_check_one's validated value (kl/tok 0.11 -> 0.05, and the only
    # setting on record that produced a real held-out gain).
    ROUTER_LR = 2e-5
    AGENT_LR = 1e-5                 # SEED's 1e-6 assumes a far larger batch; on a LoRA
                                    # at this batch size it would barely move the weights
    # 20 was ~22% of the 90 router optimiser steps at SCENARIOS_PER_STEP=2 on 450D (45
    # steps/phase x 2 router phases). Held at that SAME warmup FRACTION here: 720 training
    # scenarios at SCENARIOS_PER_STEP=8 give ~44 router steps (22/phase x 2), and 22% of
    # 44 is 10. train_450v2_big's 5 was the same fraction of its own 24 steps.
    ROUTER_WARMUP = 10
    GRAD_CLIP = 1.0

    # Relative weight of the insight KL against the route RL loss inside the router's
    # combined backward. Both land near 0.1-0.5 in practice, so 1.0 is a fair start;
    # raise it if kl_insight stops falling, lower it if routing stops improving.
    INSIGHT_COEF = 1.0

    # ---- TTPO token-level selection (arXiv:2608.27448) --------------------
    # Mask NEGATIVE-advantage spans down to the tokens that actually carried the error:
    # confident (low-entropy) yet unlikely (high-surprisal). Without it, one negative
    # advantage is spread evenly over ~191 settlement tokens, most of them JSON
    # boilderplate the model emits at logp ~= 0, and the punctuation is punished as hard
    # as the wrong field value. Positive spans are never masked - see ttpo.py.
    # A group is "low spread" when its reward standard deviation is below this. GRPO
    # divides by that std, so these are exactly the groups where it manufactures a large
    # advantage from noise. Reported as low_std_frac in train_steps.csv; it is a
    # diagnostic threshold only and never enters the loss.
    LOW_STD_THRESH = 0.05

    TTPO_MASK = True
    TTPO_KEEP = 0.5                 # fraction of tokens the mask keeps (paper: top-50%)

    # ---- profiles --------------------------------------------------------
    STUDENT_PROFILE = "student"     # blind
    TEACHER_PROFILE = "teacher"     # privileged - sees every private fact

    # ---- hardware --------------------------------------------------------
    # One card, both models on it, capped at gpu.VRAM_GIB (60). Router (Qwen3-4B, ~8 GiB
    # bf16) + agent (Qwen3-8B, ~16.4 GiB bf16) weights are ~24.4 GiB together, leaving
    # ~34.6 GiB for KV cache and update activations at ROLLOUT_BATCH=32. Those are
    # arithmetic, not measurements - see gpu.py for the full budget and the per-token KV
    # figure it rests on. Watch `peak_gib` in train_steps.csv for the first few steps and
    # drop ROLLOUT_BATCH (or SCENARIOS_PER_STEP) if it is not leaving a comfortable margin;
    # env.run_batch's OOM-halving and _update_with_retry mean a too-large value degrades to
    # a slower step rather than crashing, but that is a safety net, not a substitute for
    # checking. Everything hardware lives in gpu.py so training and inf.py agree.
    # Point ROUTER_DEVICE/AGENT_DEVICE at separate cards only if the box actually has two;
    # sharing one avoids cross-model fragmentation when there is only one to share.
    DEVICE = gpu.device()
    ROUTER_DEVICE = None            # None = DEVICE. Qwen3-4B,  ~8 GiB weights
    AGENT_DEVICE = None             # None = DEVICE. Qwen3-8B, ~16.4 GiB weights
    VRAM_GIB = gpu.VRAM_GIB         # hard allocator cap per card for this process

    # ---- io --------------------------------------------------------------
    OUT_DIR = os.path.join(_HERE, "results_train")
    CKPT_DIR = os.path.join(_HERE, "ckpt")
    RESUME = True
    PREFLIGHT_ABORT = True          # stop before a long run if the preflight fails
    RELEASE_EVERY = 5               # groups between torch.cuda.empty_cache()
    CSV_MAX_CHARS = 4000
    LOG_EPISODES = True             # write episodes.jsonl (full transcripts)


# ============================================================
# DEVICE
# ============================================================

def apply_device(cfg, log):
    """Push the runner's GPU choice into the module configs, then report it."""
    r_dev = cfg.ROUTER_DEVICE or cfg.DEVICE
    a_dev = cfg.AGENT_DEVICE or cfg.DEVICE
    opd.OPD_CFG.DEVICE = r_dev
    rl.RL_CFG.DEVICE = a_dev
    rl.RL_CFG.GROUP_SIZE = cfg.GROUP_SIZE
    opd.OPD_CFG.GRAD_CHECKPOINT = gpu.GRAD_CHECKPOINT
    rl.RL_CFG.GRAD_CHECKPOINT = gpu.GRAD_CHECKPOINT
    # Before the models are built: the cap is what makes a runaway batch raise OOM (which
    # env.run_batch answers by halving) instead of taking the whole card down with it.
    for d in {str(r_dev), str(a_dev)}:
        gpu.cap_vram(d, cfg.VRAM_GIB)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for name, d in (("router", r_dev), ("agent", a_dev)):
            if "cuda" in str(d):
                idx = int(str(d).split(":")[1]) if ":" in str(d) else 0
                if idx >= n:
                    raise RuntimeError(
                        "{} wants {} but only {} CUDA device(s) visible. Check "
                        "CONFIG.DEVICE / CUDA_VISIBLE_DEVICES.".format(name, d, n))
        log("  devices    : router={}  agent={}   ({} CUDA device(s) visible){}".format(
            r_dev, a_dev, n, "" if r_dev == a_dev else "   [SPLIT ACROSS GPUs]"))
    else:
        log("  devices    : CUDA unavailable -> both models will fall back to cpu")


# ============================================================
# LOGGING
# ============================================================

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


# ============================================================
# UPDATES
# ============================================================

def _peak_mem():
    """True high-water mark of ALLOCATED memory, per device, in GiB.

    This is the number to trust, not nvidia-smi. nvidia-smi reports memory RESERVED by
    PyTorch's caching allocator, which by design grows to the largest batch x longest
    sequence seen and is never handed back to the driver unless empty_cache() is called.
    A reserved figure that creeps up and then flattens is the allocator warming up, not
    a leak. If THIS number flattens while nvidia-smi still climbs, there is nothing wrong.
    """
    out = {}
    try:
        if not torch.cuda.is_available():
            return out
        for d in {opd.OPD_CFG.DEVICE, rl.RL_CFG.DEVICE}:
            if "cuda" in str(d):
                out[str(d)] = torch.cuda.max_memory_allocated(d) / (1024.0 ** 3)
    except Exception:                                           # noqa: BLE001
        pass
    return out


def _raw_gnorm(params):
    """Total L2 gradient norm WITHOUT clipping or otherwise touching the gradients.

    Used to read the route head's contribution before the insight head's backward is
    added, because they share one LoRA and therefore one clip. On the first V2 run |g|
    was 0.99 / 1.99 / 6.90 while route_logp sat at -0.000 - the norm was almost entirely
    distillation, and a single combined number could not say so.
    """
    tot = 0.0
    for prm in params:
        if prm.grad is not None:
            tot += float(prm.grad.detach().float().pow(2).sum())
    return tot ** 0.5


def _gnorm(params, clip):
    """clip_grad_norm_ returns the total PRE-clip norm, so the run's most important
    diagnostic costs nothing. Defensive float(): the return type has moved between torch
    versions and a 60-hour run must not die converting a log line."""
    try:
        return float(torch.nn.utils.clip_grad_norm_(params, clip))
    except (TypeError, ValueError):
        return float("nan")


def router_update(router, route_turns, opd_steps, optim, sched, params, cfg=CONFIG):
    """ONE optimiser step for the router, carrying BOTH heads' gradients.

      route head    -A * mean_t log pi(next_agent tokens)      [reward]
      insight head  per-token reverse KL vs the privileged teacher   [distillation]

    They share one LoRA, so they share one backward and one step. Backward is taken per
    item and the graph freed immediately - accumulating a batch of 152k-vocab graphs is
    what made the previous run OOM.
    """
    optim.zero_grad(set_to_none=True)
    router.model.train()

    # ---- head 1: the routing decision, from reward ----
    # TTPO's negative-sample mask applies only where the advantage is negative. A route
    # span is 1-2 tokens, so the mask is a no-op there by construction (see
    # ttpo.confident_error_mask) - it is threaded through anyway so the two heads share
    # one code path and the agent side cannot drift from it.
    scored = [t for t in route_turns if abs(t.get("advantage", 0.0)) > 1e-9]
    rl_loss, n_route, n_route_tok, lp_acc = 0.0, 0, 0, 0.0
    for tr in scored:
        neg = tr["advantage"] < 0
        out = router.span_logprobs(tr["system"], tr["user"], tr["generation"],
                                   tr["span"], with_grad=True,
                                   want_entropy=neg and cfg.TTPO_MASK)
        if out is None:
            continue
        lp, ent = out if isinstance(out, tuple) else (out, None)
        if lp.numel() == 0:
            continue
        mask = (ttpo.confident_error_mask(lp.detach(), ent, cfg.TTPO_KEEP)
                if (neg and cfg.TTPO_MASK and ent is not None) else None)
        loss = rl.reinforce_loss(lp, tr["advantage"], mask)
        (loss / len(scored)).backward()
        tr["n_tokens"] = int(lp.numel())
        tr["logp_mean"] = float(lp.detach().mean())
        tr["masked_frac"] = float(mask.mean()) if mask is not None else 1.0
        rl_loss += float(loss.detach())
        lp_acc += tr["logp_mean"]
        n_route_tok += int(lp.numel())
        n_route += 1
        del loss, lp

    g_route = _raw_gnorm(params)

    # ---- head 2: the insight, distilled from the privileged teacher ----
    # POSITIVE_ONLY (TTPO 3.2): only rollouts that beat their group mean are distilled.
    # main() tags every recorded router call with the advantage of the rollout it came
    # from, so this filter is a comparison rather than another forward pass.
    steps = [s for s in opd_steps if opd.keep_step(s, opd.OPD_CFG)]
    if opd.OPD_CFG.POSITIVE_ONLY:
        steps = [s for s in steps if s.get("_advantage", 0.0) > 0]
    random.shuffle(steps)
    kl = {"route": 0.0, "insight": 0.0}
    tok = {"route": 0, "insight": 0}
    n_opd, w_acc = 0, 0.0
    for rec in steps:
        loss_sum, ntok = opd.step_loss(router, rec, opd.OPD_CFG)
        if loss_sum is None:
            continue
        (loss_sum / max(ntok, 1) / len(steps) * cfg.INSIGHT_COEF).backward()
        k = rec["kind"]
        kl[k] = kl.get(k, 0.0) + float(loss_sum.detach())
        tok[k] = tok.get(k, 0) + ntok
        w_acc += rec.get("_mean_token_weight", 1.0)
        n_opd += 1
        del loss_sum

    # THE diagnostic. clip_grad_norm_ returns the total pre-clip norm, so this is free -
    # and it is the only number here that actually says whether the weights moved.
    #
    # `route_loss` cannot say that, and never could. Group advantages sum to exactly zero
    # under BOTH estimators - GRPO divides by a per-group constant, which preserves it -
    # within a group, and the four route spans of one group carry near-identical log pi
    # (they all pick some agent id at temperature 0.3), so
    #     sum_i -A_i * logp_i  ~=  -logp_bar * sum_i A_i  =  0.
    # The scalar collapses; the GRADIENT does not, because each rollout picked a
    # DIFFERENT id and so contributes a different direction in parameter space. A loss of
    # +0.0001 next to a healthy grad_norm is exactly what a working group estimator looks
    # like. Judge the run on grad_norm and logp_mean, never on route_loss.
    gnorm = float("nan")
    if n_route or n_opd:
        gnorm = _gnorm(params, cfg.GRAD_CLIP)
        optim.step()
        sched.step()
    optim.zero_grad(set_to_none=True)

    per = lambda k: (kl[k] / tok[k]) if tok[k] else float("nan")  # noqa: E731
    return {
        "route_loss": (rl_loss / n_route) if n_route else float("nan"),
        "route_scored": n_route, "route_tok": n_route_tok,
        "route_logp": (lp_acc / n_route) if n_route else float("nan"),
        "grad_norm": gnorm, "g_route": g_route,
        "g_insight": max(0.0, gnorm - g_route) if gnorm == gnorm else float("nan"),
        "kl_insight": per("insight"), "tok_insight": tok["insight"],
        "opd_steps": n_opd, "opd_pool": len(opd_steps),
        "tok_weight": (w_acc / n_opd) if n_opd else float("nan"),
    }


def agent_update(agent, turns, optim, params, cfg=CONFIG):
    """ONE optimiser step for the agents.

        L = - (1/N) * sum_spans  A_rollout * mean_t log pi(text tokens)

    No ratio, no clip, no reference forward: with a single inner pass theta == theta_old,
    so the importance ratio is identically 1. train_V1 ran a PPO clip anyway and measured
    clip_frac 0.001-0.003 across the whole run - it never fired once.
    """
    optim.zero_grad(set_to_none=True)
    agent.model.train()

    scored = [t for t in turns if abs(t.get("advantage", 0.0)) > 1e-9 and t.get("span")]
    if not scored:
        return {"agent_loss": float("nan"), "agent_scored": 0, "agent_tok": 0,
                "agent_neg": 0, "grad_norm": float("nan"),
                "masked_frac": float("nan"), "logp_mean": float("nan")}

    tot, n_tok, used, lp_acc, mk_acc, n_neg = 0.0, 0, 0, 0.0, 0.0, 0
    for tr in scored:
        neg = tr["advantage"] < 0
        want_h = neg and cfg.TTPO_MASK
        out = agent.token_logprobs(tr["rendered_prompt"], tr["generation"],
                                   span=tr["span"], with_grad=True,
                                   want_entropy=want_h)
        if out is None:
            continue
        lp, ent = out if isinstance(out, tuple) else (out, None)
        if lp.numel() == 0:
            continue
        # TTPO 3.4: the mask exists to stop a failed rollout punishing its own
        # locally-correct tokens. In a group-relative objective there are no
        # positive-advantage gradients on those same positions to cancel the damage,
        # which is exactly why the paper masks negatives and leaves positives whole.
        mask = (ttpo.confident_error_mask(lp.detach(), ent, cfg.TTPO_KEEP)
                if (want_h and ent is not None) else None)
        loss = rl.reinforce_loss(lp, tr["advantage"], mask)
        (loss / len(scored)).backward()
        tr["n_tokens"] = int(lp.numel())
        tr["logp_mean"] = float(lp.detach().mean())
        tr["masked_frac"] = float(mask.mean()) if mask is not None else 1.0
        if mask is not None:
            mk_acc += tr["masked_frac"]
            n_neg += 1
        lp_acc += tr["logp_mean"]
        tot += float(loss.detach())
        n_tok += int(lp.numel())
        used += 1
        del loss, lp

    gnorm = float("nan")
    if used:
        gnorm = _gnorm(params, cfg.GRAD_CLIP)
        optim.step()
    optim.zero_grad(set_to_none=True)

    return {"agent_loss": (tot / used) if used else float("nan"),
            "agent_scored": used, "agent_tok": n_tok,
            "agent_neg": n_neg, "grad_norm": gnorm,
            "masked_frac": (mk_acc / n_neg) if n_neg else float("nan"),
            "logp_mean": (lp_acc / used) if used else float("nan")}


# ============================================================
# CSV
# ============================================================

STEP_COLS = ["round", "phase", "gstep", "scenarios", "episodes", "degenerate_frac",
             "group_std", "low_std_frac", "adv_saturated",
             "route_loss", "route_scored", "route_tok",
             "route_logp", "grad_norm", "g_route", "g_insight",
             "kl_insight", "tok_insight", "opd_steps", "opd_pool", "tok_weight",
             "agent_loss", "agent_scored", "agent_tok", "agent_neg", "masked_frac",
             "logp_mean",
             "reward_mean", "reward_spread", "checks_frac", "success",
             "settled", "content_frac", "prov_frac", "reveals",
             "routing_precision", "route_invalid", "forced_routes",
             "turns_used", "leak_max", "slot_frac", "peak_gib", "secs"]

ROUND_COLS = ["round", "phase", "scenarios", "episodes"] + [
    "checks_frac", "success", "settled", "content_frac", "prov_frac", "reveals",
    "routing_precision", "route_invalid", "turns_used", "degenerate_frac",
    "leak_max", "slot_frac", "reward_mean", "hours"]


def _agg(summaries):
    """The per-episode numbers a step or a round reports."""
    m = lambda k: _mean([s.get(k) for s in summaries])  # noqa: E731
    ct = _mean([s.get("content_total") for s in summaries]) or 1.0
    pt = _mean([s.get("prov_total") for s in summaries]) or 1.0
    return {
        "checks_frac": round(m("checks_frac"), 4),
        "success": round(m("success"), 4),
        "settled": round(m("settled"), 4),
        "content_frac": round(m("content_passed") / ct, 4) if ct else "",
        "prov_frac": round(m("prov_passed") / pt, 4) if pt else "",
        "reveals": round(m("reveals"), 3),
        "routing_precision": round(m("routing_precision"), 4),
        "route_invalid": round(m("route_invalid"), 3),
        "forced_routes": round(m("forced_routes"), 3),
        "turns_used": round(m("turns_used"), 2),
        "leak_max": round(m("insight_private_leak_max"), 4),
        "slot_frac": round(m("insight_slot_frac"), 4),
        "reward_mean": round(m("train_reward"), 4),
    }


# ============================================================
# CHECKPOINTS
# ============================================================

def _fingerprint(n_train):
    """What a checkpoint must agree with before it is resumed. Resuming under a different
    model, data file or schedule would silently splice two experiments into one."""
    return {"router_model": opd.OPD_CFG.MODEL, "agent_model": rl.RL_CFG.MODEL,
            "train_file": os.path.basename(CONFIG.TRAIN_PATH), "n_train": n_train,
            "phases": list(CONFIG.PHASES), "group_size": CONFIG.GROUP_SIZE,
            "scenarios_per_step": CONFIG.SCENARIOS_PER_STEP, "seed": CONFIG.SEED,
            "router_lora_r": opd.OPD_CFG.LORA_R, "agent_lora_r": rl.RL_CFG.LORA_R,
            "estimator": rl.RL_CFG.ESTIMATOR}


def _rng_state():
    st = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st, log):
    try:
        random.setstate(st["python"])
        torch.set_rng_state(st["torch"])
        if st.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as exc:                                    # noqa: BLE001
        log("  [resume] RNG state not restored ({}: {}) - sampling from here on differs "
            "from an uninterrupted run".format(type(exc).__name__, exc))


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint(router, agent, r_optim, a_optim, r_sched, state):
    """Both adapters, both AdamW states, the LR schedule and every RNG, committed as ONE
    unit. state.json is written last and is what marks the directory complete."""
    final = os.path.join(CONFIG.CKPT_DIR, "latest")
    tmp = final + ".tmp"
    resume.fresh_dir(tmp)
    router.save(os.path.join(tmp, "router"))
    agent.save(os.path.join(tmp, "agent"))
    torch.save({"r_optim": r_optim.state_dict(), "a_optim": a_optim.state_dict(),
                "r_sched": r_sched.state_dict(), "rng": _rng_state()},
               os.path.join(tmp, "trainer.pt"))
    resume.write_json_atomic(os.path.join(tmp, "state.json"), state)
    resume.fsync_tree(tmp)
    resume.commit_dir(tmp, final)


def save_round(lm, name):
    final = os.path.join(CONFIG.CKPT_DIR, name)
    tmp = final + ".tmp"
    resume.fresh_dir(tmp)
    lm.save(tmp)
    resume.fsync_tree(tmp)
    resume.commit_dir(tmp, final)


def _fatal(log, msg):
    log("")
    log("[FATAL] " + msg)
    log.close()
    sys.exit(resume.EXIT_FATAL)


def _update_with_retry(fn, router, agent, log, *args):
    """An OOM inside the backward leaves nothing half-applied - the optimiser only steps at
    the very end and zero_grad runs first - so one retry on a freed cache is safe."""
    try:
        return fn(*args)
    except env._OOM as exc:
        log("    [update OOM] {} - freeing cache, retrying once".format(str(exc)[:160]))
        env._empty_cache(router, agent)
        return fn(*args)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
    os.makedirs(CONFIG.CKPT_DIR, exist_ok=True)

    # The last COMMITTED checkpoint, never a half-written one (see resume.commit_dir).
    ck = (resume.committed_dir(os.path.join(CONFIG.CKPT_DIR, "latest"))
          if CONFIG.RESUME else None)
    saved = resume.read_json(os.path.join(ck, "state.json")) if ck else None
    resuming = saved is not None

    log = Tee(os.path.join(CONFIG.OUT_DIR, "train.log"), append=resuming)

    try:
        scenarios = env.load_scenarios(CONFIG.TRAIN_PATH, CONFIG.N_SCENARIOS)
        held_out = env.load_scenarios(CONFIG.TEST_PATH)
    except (OSError, ValueError, TypeError) as exc:
        _fatal(log, "cannot load the data: {}: {}".format(type(exc).__name__, exc))
    overlap = ({(s.domain, s.scenario_id) for s in scenarios}
               & {(s.domain, s.scenario_id) for s in held_out})
    if overlap:
        _fatal(log, "{} held-out scenarios are also in the training file, e.g. {}".format(
            len(overlap), sorted(overlap)[:3]))

    fp = _fingerprint(len(scenarios))
    if resuming:
        if saved.get("fingerprint") != fp:
            _fatal(log, "checkpoint {} belongs to a different experiment.\n"
                        "  checkpoint: {}\n  config    : {}\n"
                        "  Restore the config, or move ckpt/ and results_train/ aside to "
                        "start fresh.".format(ck, saved.get("fingerprint"), fp))
        if saved.get("done"):
            log("Training already complete (gstep={}). Checkpoint: {}".format(
                saved["gstep"], ck))
            log.close()
            return
    state = saved or {"round": 0, "idx": 0, "gstep": 0}

    random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)

    student = prompts.PROFILES[CONFIG.STUDENT_PROFILE]
    teacher = prompts.PROFILES[CONFIG.TEACHER_PROFILE]

    # ---- the phase partition: 4 contiguous slices of a fixed shuffle ----
    order = list(range(len(scenarios)))
    random.Random(CONFIG.SEED).shuffle(order)
    n_ph = len(CONFIG.PHASES)
    per = len(order) // n_ph
    slices = [[scenarios[i] for i in order[k * per:(k + 1) * per]] for k in range(n_ph)]
    slices[-1] += [scenarios[i] for i in order[n_ph * per:]]     # remainder to the last

    total_eps = len(scenarios) * CONFIG.GROUP_SIZE

    log("=" * 78)
    log("CO-EVOLVING TRAINING   V2-GRPO   alternating best-response")
    log("=" * 78)
    if resuming:
        log("  RESUMING   : round {}/{} at scenario {}, gstep {}   <- {}".format(
            state["round"] + 1, len(CONFIG.PHASES), state["idx"], state["gstep"], ck))
    log("  router     : {}  LoRA r={}  lr={}".format(
        opd.OPD_CFG.MODEL, opd.OPD_CFG.LORA_R, CONFIG.ROUTER_LR))
    log("               next_agent <- REINFORCE/GRPO      insight <- OPD (privileged)")
    log("  agent      : {}  LoRA r={}  lr={}".format(
        rl.RL_CFG.MODEL, rl.RL_CFG.LORA_R, CONFIG.AGENT_LR))
    log("               text (content / settlement) <- REINFORCE/GRPO")
    log("               action keyword <- NOTHING (61% of turns have one legal action)")
    log("  data       : {} train ({})   {} held out ({}, eval only)   seed={}".format(
        len(scenarios), os.path.basename(CONFIG.TRAIN_PATH), len(held_out),
        os.path.basename(CONFIG.TEST_PATH), CONFIG.SEED))
    log("  phases     : {}  ({} scenarios each)".format(" -> ".join(CONFIG.PHASES), per))
    log("  group size : {}   -> {} episodes total".format(CONFIG.GROUP_SIZE, total_eps))
    log("  rollout    : {} episodes in lockstep per generate() call".format(
        CONFIG.ROLLOUT_BATCH))
    log("  opt step   : every {} scenarios ({} rollouts), {} steps per phase".format(
        CONFIG.SCENARIOS_PER_STEP, CONFIG.SCENARIOS_PER_STEP * CONFIG.GROUP_SIZE,
        max(1, per // CONFIG.SCENARIOS_PER_STEP)))
    log("  last call  : complete={}  final_turn={}   (V1 lost 104/360 episodes here)".format(
        env.ENV_CFG.FORCE_DM_WHEN_COMPLETE, env.ENV_CFG.FORCE_DM_ON_LAST_TURN))
    log("  shaping    : route_hit_bonus={}".format(env.ENV_CFG.ROUTE_HIT_BONUS))
    log("  reward     : R_train in [{:.2f}, {:.2f}]  shift={} (the group is the baseline)"
        .format(*rl.REWARD.bounds(), rl.REWARD.return_shift))
    log("  estimator  : {}   A_i = (r_i - mean_j r_j) / (std_j + {})   clip={}".format(
        rl.RL_CFG.ESTIMATOR.upper(), rl.RL_CFG.ADV_EPS, rl.RL_CFG.ADV_CLIP))
    log("  ttpo       : kl={}  token_weighting={}  positive_only={}  "
        "neg_mask={} keep={}".format(
            opd.OPD_CFG.KL_DIRECTION, opd.OPD_CFG.TOKEN_WEIGHTING,
            opd.OPD_CFG.POSITIVE_ONLY, CONFIG.TTPO_MASK, CONFIG.TTPO_KEEP))
    log("  checkpoint : every optimiser step -> {}  (adapters + AdamW + LR + RNG)".format(
        os.path.join(CONFIG.CKPT_DIR, "latest")))
    apply_device(CONFIG, log)
    log("=" * 78)

    router = opd.RouterLM(opd.OPD_CFG, train=True,
                          adapter_path=os.path.join(ck, "router") if resuming else None)
    agent = rl.AgentLM(rl.RL_CFG, train=True,
                       adapter_path=os.path.join(ck, "agent") if resuming else None)

    # Fresh start: a failing preflight stops the run before it costs anything. On resume
    # the adapters are already trained, so preflight is reported but never blocks recovery.
    if CONFIG.PREFLIGHT_ABORT and not resuming:
        if not env.preflight(router, agent, scenarios, log, True):
            _fatal(log, "preflight failed - see above. Nothing was trained.")
    else:
        env.preflight(router, agent, scenarios, log, False)

    r_params = [p for p in router.model.parameters() if p.requires_grad]
    a_params = [p for p in agent.model.parameters() if p.requires_grad]
    r_optim = torch.optim.AdamW(r_params, lr=CONFIG.ROUTER_LR)
    a_optim = torch.optim.AdamW(a_params, lr=CONFIG.AGENT_LR)
    r_sched = torch.optim.lr_scheduler.LambdaLR(
        r_optim, lambda s: min(1.0, (s + 1) / max(1, CONFIG.ROUTER_WARMUP)))
    if resuming:
        blob = _torch_load(os.path.join(ck, "trainer.pt"))
        r_optim.load_state_dict(blob["r_optim"])
        a_optim.load_state_dict(blob["a_optim"])
        r_sched.load_state_dict(blob["r_sched"])
        _set_rng_state(blob["rng"], log)
        del blob
        log("  [resume] adapters, AdamW moments, LR schedule and RNG state restored")

    # ---- csv sinks ----
    # On resume every file is cut back to its size at the checkpoint: rows written after
    # it belong to rollouts that are about to be redone and must not be counted twice.
    names = {"step": ("train_steps.csv", STEP_COLS),
             "round": ("train_rounds.csv", ROUND_COLS),
             "turn": ("train_turns.csv", env.TURN_COLS),
             "span": ("train_rl_spans.csv", env.RL_TURN_COLS),
             "ep": ("episodes.jsonl", None)}
    files, writers = {}, {}
    for key, (fname, cols) in names.items():
        path = os.path.join(CONFIG.OUT_DIR, fname)
        if resuming and key in saved.get("offsets", {}):
            resume.truncate(path, saved["offsets"][key])
        keep = resuming and os.path.exists(path) and os.path.getsize(path) > 0
        files[key] = open(path, "a" if keep else "w", encoding="utf-8",
                          newline="" if cols else None)
        if cols:
            writers[key] = csv.DictWriter(files[key], fieldnames=cols,
                                          quoting=csv.QUOTE_ALL, extrasaction="ignore")
            if not keep:
                writers[key].writeheader()
    f_step, f_round, f_turn, f_span, f_ep = (files[k] for k in names)
    w_step, w_round, w_turn, w_span = (writers[k] for k in ("step", "round", "turn", "span"))

    t_start = time.time()
    elapsed0 = float(state.get("elapsed", 0.0))
    gstep = state["gstep"]
    n_groups_done = 0
    consec_fail = 0
    skipped = list(state.get("skipped", []))

    def commit(rnd_next, idx, acc, done=False):
        save_checkpoint(router, agent, r_optim, a_optim, r_sched, {
            "round": rnd_next, "idx": idx, "gstep": gstep, "done": done,
            "fingerprint": fp, "offsets": resume.sizes(files), "acc": acc,
            "elapsed": elapsed0 + time.time() - t_start, "skipped": skipped})

    # Scenarios per lockstep rollout call. One optimiser step's worth, unless
    # ROLLOUT_BATCH allows more than that in flight at once.
    per_call = max(1, CONFIG.ROLLOUT_BATCH // CONFIG.GROUP_SIZE)
    per_call = min(per_call, CONFIG.SCENARIOS_PER_STEP)

    for rnd in range(state["round"], len(CONFIG.PHASES)):
        phase = CONFIG.PHASES[rnd]
        batch = slices[rnd]
        start_i = state["idx"] if rnd == state["round"] else 0
        # The round's running aggregates, restored so train_rounds.csv covers the WHOLE
        # round and not just the part after a crash.
        acc = (state.get("acc") if rnd == state["round"] else None) or {}

        log("")
        log("-" * 78)
        log("ROUND {}/{}   phase={}   {} scenarios x {} rollouts{}".format(
            rnd + 1, len(CONFIG.PHASES), phase.upper(), len(batch), CONFIG.GROUP_SIZE,
            "   (resuming at {})".format(start_i) if start_i else ""))
        log("  training : {}".format(
            "router.next_agent (GRPO) + router.insight (OPD)" if phase == "router"
            else "agent text spans (GRPO)"))
        log("  frozen   : {}".format("agents" if phase == "router" else "router"))
        log("-" * 78)

        r_t0 = time.time() - float(acc.get("secs", 0.0))
        pending_route, pending_agent, pending_opd, pending_sum = [], [], [], []
        round_sums = acc.get("sums", [])
        round_degen, round_groups = acc.get("degen", 0), acc.get("groups", 0)
        round_stds = acc.get("stds", [])
        round_lowstd, round_nondegen = acc.get("lowstd", 0), acc.get("nondegen", 0)
        round_satur, round_advs = acc.get("satur", 0), acc.get("advs", 0)
        s_t0 = time.time()

        i = start_i - 1
        for blk in range(start_i, len(batch), per_call):
            chunk_scens = batch[blk:blk + per_call]
            # ONE lockstep rollout for every scenario in the block x GROUP_SIZE rollouts
            # each. This is the throughput change: len(chunk)*G episodes advance together,
            # so a turn costs one batched generate() per role instead of one per episode.
            groups = None
            for attempt in range(1 + CONFIG.CHUNK_RETRIES):
                try:
                    groups = env.run_groups(
                        chunk_scens, router, agent, CONFIG.GROUP_SIZE,
                        record_router=(phase == "router"),
                        chunk=CONFIG.ROLLOUT_BATCH,
                        student_profile=student,
                        teacher_profile=teacher if phase == "router" else None,
                        record_agent=True)
                    break
                except Exception as exc:                   # noqa: BLE001
                    log("    [chunk error {}/{}] uids={} {}: {}".format(
                        attempt + 1, 1 + CONFIG.CHUNK_RETRIES,
                        [s_.uid for s_ in chunk_scens], type(exc).__name__, str(exc)[:300]))
                    env._empty_cache(router, agent)
            if groups is None:
                consec_fail += 1
                if consec_fail >= CONFIG.MAX_CONSEC_FAILS:
                    raise RuntimeError(
                        "{} rollout chunks failed in a row - the CUDA context is probably "
                        "unusable. Exiting so run.py restarts from the last checkpoint."
                        .format(consec_fail))
                log("    [SKIP] uids={}".format([s_.uid for s_ in chunk_scens]))
                skipped.extend(s_.uid for s_ in chunk_scens)
                i = blk + len(chunk_scens) - 1
                continue
            consec_fail = 0

            for group in groups:
                i += 1
                # ---- GRPO over the group (train_V2_grpo) ----
                key = "route_reward" if phase == "router" else "train_reward"
                rewards = [g["summary"][key] for g in group]
                advs = rl.group_advantages(rewards, clip=rl.RL_CFG.ADV_CLIP)
                round_degen += int(rl.degenerate(rewards))
                round_groups += 1

                # GRPO-specific diagnostics. The estimator divides by the group std, so
                # a group with a small spread produces a large advantage from very little
                # information. These two counters make that visible per step instead of
                # leaving it to be inferred from a training curve that has already
                # diverged.
                _sd = rl.group_std(rewards)
                if not rl.degenerate(rewards):
                    round_stds.append(_sd)
                    if _sd < CONFIG.LOW_STD_THRESH:
                        round_lowstd += 1
                    round_nondegen += 1
                round_satur += sum(1 for a in advs
                                   if abs(a) >= rl.RL_CFG.ADV_CLIP - 1e-9)
                round_advs += len(advs)

                for g, a in zip(group, advs):
                    # TTPO 3.2: tag every router call with the advantage of the rollout
                    # that produced it, so router_update distils only the positives.
                    for d in g["router_steps"]:
                        d["_advantage"] = a
                    sm = g["summary"]
                    round_sums.append(sm)
                    pending_sum.append(sm)
                    for tr in g["route_turns"]:
                        tr["advantage"] = a
                        tr["group_mean"] = sum(rewards) / len(rewards)
                        tr["who"] = "router"
                        pending_route.append(tr)
                    for tr in g["agent_turns"]:
                        tr["advantage"] = a
                        tr["group_mean"] = sum(rewards) / len(rewards)
                        tr["who"] = "agent"
                        pending_agent.append(tr)
                    for r in g["rows"]:
                        w_turn.writerow({k: _clip(r.get(k), CONFIG.CSV_MAX_CHARS)
                                         for k in env.TURN_COLS})
                    if CONFIG.LOG_EPISODES:
                        f_ep.write(json.dumps(g["episode"], ensure_ascii=False) + "\n")
                    pending_opd.extend(g["router_steps"])
                n_groups_done += 1

                # ---- optimiser step ----
                if not ((i + 1) % CONFIG.SCENARIOS_PER_STEP == 0
                        or i == len(batch) - 1):
                    continue
                gstep += 1
                if phase == "router":
                    stats = _update_with_retry(router_update, router, agent, log,
                                               router, pending_route, pending_opd,
                                               r_optim, r_sched, r_params)
                    scored = pending_route
                else:
                    stats = _update_with_retry(agent_update, router, agent, log,
                                               agent, pending_agent, a_optim, a_params)
                    scored = pending_agent

                for tr in scored:
                    row = env.rl_turn_row(tr)
                    row.update({"round": rnd, "phase": phase, "gstep": gstep})
                    w_span.writerow(row)

                agg = _agg(pending_sum)
                rs = [s["train_reward"] for s in pending_sum]
                row = {"round": rnd, "phase": phase, "gstep": gstep,
                       "scenarios": CONFIG.SCENARIOS_PER_STEP,
                       "episodes": len(pending_sum),
                       "degenerate_frac": round(round_degen / max(round_groups, 1), 4),
                       "group_std": round(_mean(round_stds), 4) if round_stds else "",
                       "low_std_frac": round(round_lowstd / max(round_nondegen, 1), 4),
                       "adv_saturated": round(round_satur / max(round_advs, 1), 4),
                       "reward_mean": round(_mean(rs), 4),
                       "reward_spread": round(max(rs) - min(rs), 4) if rs else "",
                       "secs": round(time.time() - s_t0, 1)}
                pk = _peak_mem()
                row["peak_gib"] = round(max(pk.values()), 2) if pk else ""
                row.update(agg)
                row.update(stats)
                w_step.writerow(row)
                f_step.flush(); f_turn.flush(); f_span.flush(); f_ep.flush()

                head = ("|g|={:.3f} (rt {:.3f}) logp={:.3f} n={} | kl_ins={:.4f} "
                        "opd={}/{} w={:.2f}".format(
                            stats.get("grad_norm", float("nan")),
                            stats.get("g_route", float("nan")),
                            stats.get("route_logp", float("nan")),
                            stats.get("route_scored", 0),
                            stats.get("kl_insight", float("nan")),
                            stats.get("opd_steps", 0), stats.get("opd_pool", 0),
                            stats.get("tok_weight", float("nan")))
                        if phase == "router" else
                        "|g|={:.4f} logp={:.3f} n={} (neg {}) mask={:.2f} tok={}".format(
                            stats.get("grad_norm", float("nan")),
                            stats.get("logp_mean", float("nan")),
                            stats.get("agent_scored", 0), stats.get("agent_neg", 0),
                            stats.get("masked_frac", float("nan")),
                            stats.get("agent_tok", 0)))
                log("[r{} {} {}/{}] {} | R={:+.3f} spread={:.3f} degen={:.2f} | "
                    "checks={:.3f} success={:.2f} settled={:.2f} rprec={:.3f} "
                    "turns={:.1f} leak={:.3f} | {}{:.0f}s".format(
                        rnd + 1, phase[:3], i + 1, len(batch), head,
                        row["reward_mean"], row["reward_spread"] or 0.0,
                        row["degenerate_frac"], agg["checks_frac"], agg["success"],
                        agg["settled"], agg["routing_precision"] or 0.0,
                        agg["turns_used"], agg["leak_max"] or 0.0,
                        ("%.1fG " % row["peak_gib"]) if row["peak_gib"] != "" else "",
                        row["secs"]))

                # After the very first step, project the whole run. On a rented GPU the
                # number you want before committing is hours, not steps.
                if gstep == 1:
                    per_ep = row["secs"] / max(len(pending_sum), 1)
                    proj = per_ep * len(scenarios) * CONFIG.GROUP_SIZE / 3600.0
                    log("  [projection] {:.1f}s/episode -> {} episodes ~= {:.1f}h total."
                        .format(per_ep, len(scenarios) * CONFIG.GROUP_SIZE, proj))
                    log("               GROUP_SIZE is the lever: it scales linearly, and "
                        "3 still gives a usable leave-one-out baseline.")

                pending_route, pending_agent, pending_opd, pending_sum = [], [], [], []
                s_t0 = time.time()

                commit(rnd, i + 1, {
                    "sums": round_sums, "degen": round_degen, "groups": round_groups,
                    "stds": round_stds, "lowstd": round_lowstd,
                    "nondegen": round_nondegen, "satur": round_satur, "advs": round_advs,
                    "secs": time.time() - r_t0})

            if CONFIG.RELEASE_EVERY and n_groups_done % CONFIG.RELEASE_EVERY == 0:
                router.release()
                agent.release()

        # ---- end of round: a checkpoint you can put in a cross-product matrix ----
        save_round(router, "router_round{}".format(rnd + 1))
        save_round(agent, "agent_round{}".format(rnd + 1))
        agg = _agg(round_sums)
        rrow = {"round": rnd, "phase": phase, "scenarios": len(batch),
                "episodes": len(round_sums),
                "degenerate_frac": round(round_degen / max(round_groups, 1), 4),
                "hours": round((time.time() - r_t0) / 3600.0, 3)}
        rrow.update(agg)
        w_round.writerow(rrow)
        f_round.flush()
        log("-- round {} ({}) done in {:.2f}h -> router_round{} / agent_round{} --".format(
            rnd + 1, phase, rrow["hours"], rnd + 1, rnd + 1))
        pk = _peak_mem()
        if pk:
            log("   peak allocated: {}   (nvidia-smi shows RESERVED, which is higher by "
                "design and is not a leak)".format(
                    "  ".join("{}={:.2f}GiB".format(k, v) for k, v in sorted(pk.items()))))
        log("   checks={:.4f} success={:.4f} settled={:.4f} rprec={:.4f} "
            "turns={:.2f} degen={:.2f} leak={:.4f}".format(
                agg["checks_frac"], agg["success"], agg["settled"],
                agg["routing_precision"] or 0.0, agg["turns_used"],
                rrow["degenerate_frac"], agg["leak_max"] or 0.0))
        commit(rnd + 1, 0, None, done=(rnd + 1 == len(CONFIG.PHASES)))

    for f in files.values():
        f.close()

    log("")
    log("DONE in {:.2f}h   gstep={}".format((elapsed0 + time.time() - t_start) / 3600.0,
                                            gstep))
    if skipped:
        log("  skipped    : {} scenario(s) failed twice and were not trained on: {}".format(
            len(skipped), skipped))
    log("  checkpoints : {}".format(CONFIG.CKPT_DIR))
    log("      final   : latest/router, latest/agent     <- inf.py uses these")
    log("      per-round: router_roundK / agent_roundK  <- for the cross-product matrix")
    log("  curves      : {}/train_steps.csv, train_rounds.csv".format(CONFIG.OUT_DIR))
    log("  per turn    : {}/train_turns.csv".format(CONFIG.OUT_DIR))
    log("  scored spans: {}/train_rl_spans.csv".format(CONFIG.OUT_DIR))
    log("  transcripts : {}/episodes.jsonl".format(CONFIG.OUT_DIR))
    log("")
    log("Now run:  python inf.py")
    log("")
    log("What to watch, and what it means if it does not move:")
    log("  |g| (rt X)                       total grad norm, with the ROUTE head's share")
    log("                                   in brackets. Both heads share one LoRA and")
    log("                                   one clip, so the total alone cannot say which")
    log("                                   one is moving - on the first V2 run |g| read")
    log("                                   0.99-6.90 while the route head contributed")
    log("                                   nothing. Watch (rt X), not |g|.")
    log("                                   route_loss / agent_loss are ~0 BY")
    log("                                   CONSTRUCTION: group advantages sum to zero")
    log("                                   inside a group, so the scalar cancels while")
    log("                                   the gradient does not. Never judge on loss.")
    log("  logp (route)                     the router's confidence on the agent-id")
    log("                                   token. Healthy: -0.2 to -1.5. At ~0 the")
    log("                                   decision is being made somewhere the credit")
    log("                                   span does not cover - which is exactly what")
    log("                                   the removed `reason` field was doing.")
    log("  n_tokens in train_rl_spans.csv   settlement spans ~150-250, text ~15-40.")
    log("                                   Collapsing to ~6 means the credit span")
    log("                                   slipped back onto the JSON keyword.")
    log("  logp_mean                        ~-0.03 on settlements. If it is ~0 the span")
    log("                                   has no entropy and there is no gradient.")
    log("  degenerate_frac                  above ~0.3 -> raise GROUP_SIZE.")
    log("  group_std                        mean reward spread inside a group. GRPO")
    log("                                   DIVIDES by this, so a small value inflates")
    log("                                   every advantage in that group.")
    log("  low_std_frac                     share of non-degenerate groups with std <")
    log("                                   {}. These are where GRPO manufactures a".format(
        CONFIG.LOW_STD_THRESH))
    log("                                   large advantage out of sampling noise. If")
    log("                                   this is high AND adv_saturated is high, the")
    log("                                   std division is driving training, not the")
    log("                                   reward - which is the failure this run tests.")
    log("  adv_saturated                    share of advantages pinned at +/-ADV_CLIP.")
    log("                                   Near 0 under RLOO. A large value here is the")
    log("                                   clearest single symptom of the division.")
    log("  settled                          should sit near 1.00 from step one; LAST")
    log("                                   CALL guarantees it. If not, something in")
    log("                                   env.ENV_CFG.FORCE_DM_* is off.")
    log("  leak_max                         GUARDRAIL, not an objective. If it climbs")
    log("                                   well above vanilla's, the insight head is")
    log("                                   fabricating hidden values and any checks")
    log("                                   gain is telepathy. Reject that checkpoint.")
    log("  tok_weight (router phase)        mean TTPO distillation weight. Starts around")
    log("                                   0.3-0.6 and should FALL as the student masters")
    log("                                   the slot table. Stuck at ~1.0 means the")
    log("                                   weighting is not discriminating.")
    log("  masked_frac (agent phase)        should sit near TTPO_KEEP (0.5). Far above it")
    log("                                   means the spans are too short to mask.")
    log("  peak_gib                         TRUE high-water mark of ALLOCATED memory.")
    log("                                   nvidia-smi reports RESERVED, which grows to")
    log("                                   the widest batch seen and is never returned")
    log("                                   to the driver - creeping there is normal. If")
    log("                                   peak_gib flattens, nothing is leaking.")
    log("  opd_steps / opd_pool             the positive-only fraction. Near 0 means no")
    log("                                   rollout beat its group mean, i.e. the groups")
    log("                                   are degenerate and GROUP_SIZE is too small.")
    log.close()


if __name__ == "__main__":
    main()
