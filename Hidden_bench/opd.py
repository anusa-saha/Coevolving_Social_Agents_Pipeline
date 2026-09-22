"""opd.py - the ROUTER. Two heads, two completely different learning signals.

    router.next_agent  <- REINFORCE with an RLOO baseline   (an ACTION)
    router.insight     <- on-policy distillation, privileged (a DESCRIPTION)

WHY THE SPLIT. train_V1 and check/OPD_check_one both distilled BOTH heads from the same
privileged teacher. Measured on the held-out 90, that produced:

    insight_slot_frac       0.291 -> 0.984     the insight head learned a lot
    insight_coverage        0.053 -> 0.191
    elicitation_rate        0.407 -> 0.647     and it changed downstream behaviour
    routing_precision       0.642 -> 0.599     the route head got WORSE
    success                 0.011 -> 0.000

Same model, same loss, same run, opposite outcomes. The reason is structural, not a
hyperparameter:

  * The INSIGHT is a description of the observable record - "this slot is settled, this
    one is UNKNOWN, here is what would settle it". The teacher's privilege makes it
    better at judging what is MISSING, and its prompt forbids it from stating hidden
    content, so the only thing it can transmit is "know what you do not know". That is a
    real, leak-free skill and it transfers.

  * The ROUTE is an action. A teacher that already knows every secret has no reason to
    call on anyone to find anything out, so its action distribution is the wrong target.
    Copying it teaches the student to skip the investigation. train_V1's onlyOPD run is
    the clean demonstration: reveal rate 0.832 -> 0.134, episodes 4.80 -> 2.30 turns,
    success -> 0.000.

So: DISTIL WHAT THE TEACHER KNOWS, REINFORCE WHAT THE TEACHER CANNOT DEMONSTRATE.

A second, mechanical reason to keep the route head out of the KL: the route completion is
~33 tokens of which `next_agent` is ~2, so ~95% of the routing gradient was spent matching
the teacher's prose style in the `reason` field. That is why train_V1's kl_route fell 61%
while `settled` went DOWN.

The insight KL is unchanged from check/OPD_check_one, which is where the +0.136
checks_frac came from: student samples, teacher scores the student's OWN tokens, per-token
reverse KL with discount 0.

    L = sum_steps sum_t sum_v  pi_s(v | P_student, y<t) [ log pi_s(v) - log pi_t(v | P_teacher, y<t) ]

Student = base + LoRA, blind. Teacher = the SAME base, adapter OFF, privileged prompt. The
teacher is stationary, costs no extra VRAM, and is not a stronger model - it has exactly
the student's capability and strictly more information.

UNGATED, deliberately. SEED gates each token by sigmoid(beta * (logp_teacher -
logp_student)); this file uses plain reverse KL over every token, which is the version
that produced checks_frac 0.118 -> 0.254 on the held-out 90.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

import env
import prompts
import rl
import ttpo


# ============================================================
# CONFIG
# ============================================================

class OPD_CFG:
    # ---- model ----------------------------------------------------------
    # Qwen3, not Qwen3.5, on purpose. Qwen3.5 is a HYBRID stack - three Gated-DeltaNet
    # (linear_attention) layers per one full-attention layer - which needs transformers
    # >= 5.2, wants causal_conv1d + fla or it falls back to slow ops, and carries a
    # 248k vocab that makes the full-vocab reverse KL ~1.6x heavier. Qwen3 is plain
    # dense: transformers >= 4.51, standard q/k/v/o + gate/up/down on EVERY layer, and
    # the same 151,936 vocab the check_one baseline was measured on.
    MODEL = "/scratch/rohank__iitp/Qwen3-4B"         # ~8 GiB bf16
    DEVICE = "cuda:0"               # DEFAULT ONLY - main.py / eval.py CONFIG owns this
                                    # and overwrites it before the model is built.
    DTYPE = "bfloat16"

    # Qwen3 THINKS BY DEFAULT: every generation opens with <think>...</think>. With
    # ROUTE_MAX_TOKENS below that block would eat the whole budget and every route would
    # parse as invalid. It must also be identical on the student and teacher passes or
    # the two prompts stop being byte-comparable. Leave this False - env.preflight
    # verifies it took effect before any long run starts.
    ENABLE_THINKING = False

    # ---- LoRA (the only trainable parameters) ---------------------------
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    # "auto" = discover every nn.Linear in the loaded model. On a dense Qwen2.5/Qwen3
    # this yields exactly {q,k,v,o,gate,up,down}_proj - the check_one recipe. On a hybrid
    # stack (Qwen3.5: 3 Gated-DeltaNet layers per 1 attention layer) an explicit q/k/v/o
    # list would reach only ~25% of layers and say nothing. See rl.discover_lora_targets.
    LORA_TARGETS = "auto"
    # OFF on the 80 GiB card: identical gradients without the forward recompute. If an
    # update ever OOMs, main._update_with_retry turns it back on for the rest of the run.
    GRAD_CHECKPOINT = False

    # ---- generation -----------------------------------------------------
    # 0.3, not 0.0, and not V1's 0.3-for-show: the route head is now trained by RLOO, and
    # RLOO needs the G rollouts of a group to actually differ. A deterministic router
    # would make every group degenerate and the whole route objective a no-op. 0.3 keeps
    # routing decisive while leaving real spread across a group of 4.
    TEMPERATURE = 0.3
    TOP_P = 0.95
    TOP_K = 20                      # Qwen3 recommendation for non-thinking mode
    MAX_SEQ_LEN = 4096
    # The route output is now {"next_agent": "Ax"} - about 10 tokens. 24 leaves margin
    # for a stray space or a repeated key without letting a rambling sequence hold up the
    # whole batch: a batched generate() runs until its LONGEST member stops, so a loose
    # cap is paid by every sequence in the batch, not just the offender.
    ROUTE_MAX_TOKENS = 24
    INSIGHT_MAX_TOKENS = 320

    # ---- distillation ---------------------------------------------------
    # ROUTE IS OFF. See the module docstring: a teacher holding every secret has no
    # reason to elicit anything, so its routing distribution is the wrong target -
    # measured, distilling it moved routing_precision 0.642 -> 0.599 while the insight
    # head moved slot_frac 0.291 -> 0.984 in the same run. The route head learns from
    # reward instead (main.py::router_rl_update).
    TRAIN_ROUTE = False
    TRAIN_INSIGHT = True

    # ---- route RL (the other head) --------------------------------------
    ROUTE_LR = 1e-5                 # separate from the insight LR: one head is chasing a
                                    # KL to a fixed teacher, the other a noisy reward
    INSIGHT_LR = 2e-5               # check_one's validated value (kl/tok 0.11 -> 0.05)
    KL_TEMP = 1.0
    KL_CHUNK = 64                   # sequence positions per KL chunk (vocab is ~152k wide)
    KL_CLIP = 20.0                  # per-token KL ceiling
    MAX_TRAIN_LEN = 3072            # skip a step longer than this (prompt + completion)

    # ---- TTPO (arXiv:2608.27448) ----------------------------------------
    # "forward" = KL(teacher || student), mass-covering, TTPO's positive branch.
    # "reverse" = KL(student || teacher), mode-seeking, what check/OPD_check_one ran.
    # See _distil_loss for why forward is the default here.
    KL_DIRECTION = "forward"
    # TTPO eq. 4: weight each token by Soft-OR(student entropy, teacher divergence), so
    # already-mastered positions stop consuming the gradient.
    TOKEN_WEIGHTING = True
    # Distil only the rollouts that scored ABOVE their group mean (TTPO 3.2). A privileged
    # teacher is a fair target for a record the student itself produced well; it is the
    # wrong target for a failed rollout, where "what would someone who knew every secret
    # have done" is precisely the signal that collapsed onlyOPD.
    POSITIVE_ONLY = True

    # ---- SEED's confidence gate: OFF. Ablation hook only. ----------------
    USE_GATE = False
    GATE_BETA = 5.0                 # SEED beta_opd


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ============================================================
# ROUTER LM
# ============================================================

class RouterLM:
    """One base + one LoRA. Student = adapter on. Teacher = adapter off + privileged prompt."""

    def __init__(self, cfg=OPD_CFG, adapter_path=None, train=True):
        self.cfg = cfg
        dev = cfg.DEVICE
        if "cuda" in str(dev) and not torch.cuda.is_available():
            print("[router] CUDA unavailable -> cpu", flush=True)
            dev = "cpu"
        self.device = dev
        self.calls = 0

        print("[router] loading {} -> {}".format(cfg.MODEL, dev), flush=True)
        self.tok = AutoTokenizer.from_pretrained(cfg.MODEL, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"   # chat_ids_batch flips to left per call
        dtype = _DTYPES[cfg.DTYPE] if dev != "cpu" else torch.float32
        base = rl.load_base_lm(cfg.MODEL, dtype, dev, "router")
        targets = rl.resolve_lora_targets(base, cfg.LORA_TARGETS, "router")

        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=train)
            print("[router] adapter loaded from {}".format(adapter_path), flush=True)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
                bias="none", task_type="CAUSAL_LM",
                target_modules=targets))

        if train and cfg.GRAD_CHECKPOINT and "cuda" in str(dev):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

        # Gradient checkpointing and a KV cache are mutually exclusive, and transformers
        # says so with a warning on every training forward. We never want a cache in the
        # update pass, so declare it once here rather than being told. generate() passes
        # use_cache=True explicitly and that kwarg wins, so ROLLOUT IS STILL CACHED -
        # which matters: an uncached decode re-runs the whole prompt per token and would
        # make a 2.5k-token route call quadratic.
        try:
            self.model.config.use_cache = False
            if hasattr(self.model, "base_model"):
                self.model.base_model.model.config.use_cache = False
        except AttributeError:
            pass

        self.model.train() if train else self.model.eval()
        n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[router] trainable params: {:,}  (thinking={})".format(
            n_tr, cfg.ENABLE_THINKING), flush=True)

    # ---------- adapter control ----------

    @contextlib.contextmanager
    def base_mode(self):
        """Adapter OFF: the raw base model.

        Two uses, same mechanism:
          + a PRIVILEGED prompt -> the teacher (training). Stationary by construction,
            since nothing inside this context is trainable.
          + a BLIND prompt      -> the vanilla arm (eval). LoRA's B is zero-initialised,
            so adapter-off IS mathematically the untrained model - no second load needed.
        """
        with self.model.disable_adapter():
            yield

    # training reads better as teacher_mode; eval reads better as base_mode
    teacher_mode = base_mode

    # ---------- tokenisation ----------

    def prompt_ids(self, system, user):
        """Chat template with thinking forced off. Applied identically to the student and
        teacher prompts - if these two diverged in templating, the per-token KL would be
        comparing different token grids."""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.cfg.ENABLE_THINKING)
        except TypeError:
            # tokenizer predates the enable_thinking kwarg (e.g. Qwen2.5)
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.cfg.MAX_SEQ_LEN)
        return enc["input_ids"][0]

    # ---------- sampling (the student policy) ----------

    def chat_ids(self, system, user, max_new_tokens):
        """Returns (text, completion_ids). The ids are what the teacher will score."""
        ids = self.prompt_ids(system, user).unsqueeze(0).to(self.device)
        self.calls += 1
        was_training = self.model.training
        self.model.eval()               # no LoRA dropout while sampling
        try:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=self.cfg.TEMPERATURE > 0,
                    temperature=max(self.cfg.TEMPERATURE, 1e-4),
                    top_p=self.cfg.TOP_P, top_k=self.cfg.TOP_K,
                    use_cache=True, pad_token_id=self.tok.pad_token_id)
        finally:
            self.model.train(was_training)
        new = out[0][ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), new.detach().cpu()

    def chat_ids_batch(self, system, users, max_new_tokens):
        """N prompts -> N (text, completion_ids). One batched decode. See
        rl.AgentLM.generate_batch for why this is the whole speedup.

        The per-sequence completion ids are trimmed at the first pad/eos, because OPD
        scores exactly the tokens the student produced and a trailing run of PAD would be
        distilled as if it were content.
        """
        msgs = [[{"role": "system", "content": system},
                 {"role": "user", "content": u}] for u in users]
        texts = []
        for m in msgs:
            try:
                texts.append(self.tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.cfg.ENABLE_THINKING))
            except TypeError:
                texts.append(self.tok.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True))

        prev_side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.cfg.MAX_SEQ_LEN,
                           add_special_tokens=False).to(self.device)
        finally:
            self.tok.padding_side = prev_side
        self.calls += len(users)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=self.cfg.TEMPERATURE > 0,
                    temperature=max(self.cfg.TEMPERATURE, 1e-4),
                    top_p=self.cfg.TOP_P, top_k=self.cfg.TOP_K,
                    use_cache=True, pad_token_id=self.tok.pad_token_id)
        finally:
            self.model.train(was_training)

        n_prompt = enc["input_ids"].shape[1]
        eos, pad = self.tok.eos_token_id, self.tok.pad_token_id
        res = []
        for row in out:
            new_ids = row[n_prompt:]
            keep = new_ids.shape[0]
            for j in range(new_ids.shape[0]):
                v = int(new_ids[j])
                if v == eos or (pad is not None and v == pad):
                    keep = j
                    break
            trimmed = new_ids[:keep]
            res.append((self.tok.decode(trimmed, skip_special_tokens=True),
                        trimmed.detach().cpu()))
        return res

    def ask_json_batch(self, users, max_tokens, key):
        """Batched ask_json. Returns N tuples of (value_at_key, raw, ids, reason).

        `reason` is retained in the tuple for call-site compatibility and is now always
        "" for routes - the field was removed from the output. See PROMPT_CFG.
        """
        out = []
        for raw, ids in self.chat_ids_batch(prompts.SYS_ROUTER, users, max_tokens):
            obj = env.extract_json(raw)
            if isinstance(obj, dict):
                out.append((obj.get(key), raw, ids, str(obj.get("reason", ""))[:400]))
            else:
                out.append((None, raw, ids, ""))
        return out

    def ask_json(self, user, max_tokens, key):
        """One router call. Returns (value_at_key, raw_text, completion_ids, reason)."""
        raw, ids = self.chat_ids(prompts.SYS_ROUTER, user, max_tokens)
        obj = env.extract_json(raw)
        if isinstance(obj, dict):
            return obj.get(key), raw, ids, str(obj.get("reason", ""))[:400]
        return None, raw, ids, ""

    # ---------- span scoring (the route head's RL objective) ----------

    def span_token_range(self, continuation: str, span):
        """Char span -> token index range [a, b) over the tokenised continuation."""
        if span is None:
            return None
        s_ch, e_ch = span
        if e_ch <= s_ch:
            return None
        a = len(self.tok(continuation[:s_ch], add_special_tokens=False).input_ids)
        b = len(self.tok(continuation[:e_ch], add_special_tokens=False).input_ids)
        return (a, max(b, a + 1))

    def span_logprobs(self, system: str, user: str, continuation: str, span,
                      with_grad: bool = True, want_entropy: bool = False):
        """Per-token log pi over `continuation[span]`, under the STUDENT policy.

        Used for exactly one thing: the `next_agent` value of a route call. Two tokens,
        typically. That is the entire routing decision, and scoring anything else was
        train_V1's mistake - `reason` is 30-odd tokens of prose that the router does not
        get graded on.

        The prompt is re-rendered through prompt_ids(), the same path sampling used, so
        there is no templating drift between the policy that generated and the policy
        that scores.
        """
        rng = self.span_token_range(continuation, span)
        if rng is None:
            return None
        p_ids = self.prompt_ids(system, user).unsqueeze(0).to(self.device)
        c_ids = self.tok(continuation, return_tensors="pt",
                         add_special_tokens=False).input_ids
        n_cont = c_ids.shape[1]
        if n_cont == 0:
            return None
        a, b = max(0, rng[0]), min(n_cont, rng[1])
        if b <= a:
            return None

        full = torch.cat([p_ids, c_ids.to(self.device)], dim=1)
        if full.shape[1] > self.cfg.MAX_SEQ_LEN:
            return None
        am = torch.ones_like(full)

        grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
        with grad_ctx:
            try:
                out = self.model(input_ids=full, attention_mask=am,
                                 logits_to_keep=n_cont + 1)
                logits = out.logits[:, :-1, :]
            except TypeError:
                out = self.model(input_ids=full, attention_mask=am)
                logits = out.logits[:, -(n_cont + 1):-1, :]
            if logits.shape[1] != n_cont:
                logits = logits[:, -n_cont:, :]
            lg = logits[:, a:b, :].float()
            tgt = c_ids[:, a:b].to(self.device)
            lp = F.log_softmax(lg, dim=-1)
            out = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)
            if want_entropy:
                with torch.no_grad():
                    ent = -(lp.exp() * lp).sum(-1).squeeze(0)
                return out, ent
            return out

    # ---------- io ----------

    def release(self):
        # empty_cache() frees the CURRENT device's cached blocks. Without the device
        # context this is a silent no-op whenever the model does not live on cuda:0,
        # and the rollout/update fragmentation fix quietly stops working.
        if "cuda" in str(self.device):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ============================================================
# LOSS
# ============================================================

def _completion_logits(model, input_ids, n_keep):
    """Logits over the completion span only. Projecting all ~3k prompt positions to a
    152k vocab is what blew up memory in the previous run."""
    am = torch.ones_like(input_ids)
    try:
        out = model(input_ids=input_ids, attention_mask=am, logits_to_keep=n_keep + 1)
        return out.logits[:, :-1, :]
    except TypeError:
        pass
    try:
        out = model(input_ids=input_ids, attention_mask=am, num_logits_to_keep=n_keep + 1)
        return out.logits[:, :-1, :]
    except TypeError:
        out = model(input_ids=input_ids, attention_mask=am)
        return out.logits[:, -(n_keep + 1):-1, :]


def _distil_loss(s_logits, t_logits, cfg):
    """Per-token distillation loss between the student and the privileged teacher.

    Returns (summed_loss, mean_weight). Chunked over the sequence: a full [1, L, 152k]
    float32 softmax at L=320 is ~195MB per tensor and we need four live at once.

    DIRECTION (cfg.KL_DIRECTION)

      "forward"  KL(teacher || student)   mass-covering. TTPO's positive branch, and its
                 ablation puts it clearly ahead on the samples that already did well
                 (46.7 vs 43.9 for all-FKL, 37.2 for the reversed assignment).
      "reverse"  KL(student || teacher)   mode-seeking. What check/OPD_check_one ran.

    Forward is the default here because our own numbers show the reverse-KL failure
    signature: check_one's insight_coverage_final reached only 0.191 - the trained insight
    named the schema slots almost perfectly (slot_frac 0.984) but covered under a fifth of
    the facts actually on the record. That is mode collapse onto the skeleton of a slot
    table. A mass-covering objective is the direct fix, and the insight is exactly the
    kind of output that wants coverage: a table that must mention every UNKNOWN slot, not
    the single most probable sentence. Set "reverse" to reproduce check_one exactly.

    TOKEN WEIGHTING (cfg.TOKEN_WEIGHTING)

    TTPO eq. 4. Down-weights positions where the student is already confident AND already
    agrees - the JSON skeleton, the slot names, the punctuation. check_one spent ~90
    optimiser steps and roughly fourteen hours driving its KL from 0.076 to 0.055 on
    exactly those tokens while its task metric went nowhere; this is what stops that.
    """
    total = s_logits.new_zeros(())
    w_sum, n_tok = 0.0, 0
    n = s_logits.shape[1]
    fwd = (cfg.KL_DIRECTION == "forward")

    for i in range(0, n, cfg.KL_CHUNK):
        s = s_logits[:, i:i + cfg.KL_CHUNK, :].float() / cfg.KL_TEMP
        t = t_logits[:, i:i + cfg.KL_CHUNK, :].float() / cfg.KL_TEMP
        ls = F.log_softmax(s, dim=-1)
        lt = F.log_softmax(t, dim=-1)

        if fwd:
            # KL(t || s): the teacher's mass is the reference, so every token the teacher
            # would emit has to be covered by the student.
            kl = (lt.exp() * (lt - ls)).sum(-1)
        else:
            kl = (ls.exp() * (ls - lt)).sum(-1)

        if cfg.TOKEN_WEIGHTING:
            with torch.no_grad():
                h = -(ls.exp() * ls).sum(-1)            # student entropy H(t)
                w = ttpo.distil_weights(h.squeeze(0), kl.squeeze(0).detach())
            kl = kl * w.unsqueeze(0)
            w_sum += float(w.sum())
        else:
            w_sum += float(kl.shape[-1])

        if cfg.USE_GATE:
            # SEED's confidence gate, OFF by default. Trust the teacher on tokens it
            # would have made MORE likely; ignore tokens it suppresses. Superseded by
            # TOKEN_WEIGHTING, which selects on the student's state rather than the
            # teacher's preference - kept as an ablation hook.
            with torch.no_grad():
                gate = torch.sigmoid(cfg.GATE_BETA * (lt - ls).sum(-1))
            kl = kl * gate

        if cfg.KL_CLIP:
            kl = kl.clamp(max=cfg.KL_CLIP)
        total = total + kl.sum()
        n_tok += kl.shape[-1]

    return total, (w_sum / n_tok if n_tok else 0.0)


# kept for callers that still name the old function
_reverse_kl = lambda s, t, c: _distil_loss(s, t, c)[0]  # noqa: E731


def step_loss(lm: RouterLM, rec: dict, cfg=OPD_CFG):
    """One recorded router call -> (summed KL over its tokens, token count).

    The student and teacher prompts differ in LENGTH (the teacher carries the privileged
    block), so the two forward passes are aligned from the RIGHT: the last L positions
    are the completion in both cases.
    """
    comp = rec["completion_ids"]
    if comp is None or len(comp) == 0 or rec["teacher_prompt"] is None:
        return None, 0
    comp = comp.to(lm.device)
    L = comp.shape[0]

    s_ids = lm.prompt_ids(prompts.SYS_ROUTER, rec["student_prompt"]).to(lm.device)
    t_ids = lm.prompt_ids(prompts.SYS_ROUTER, rec["teacher_prompt"]).to(lm.device)
    s_in = torch.cat([s_ids, comp]).unsqueeze(0)
    t_in = torch.cat([t_ids, comp]).unsqueeze(0)
    if s_in.shape[1] > cfg.MAX_TRAIN_LEN or t_in.shape[1] > cfg.MAX_TRAIN_LEN:
        return None, 0

    with torch.no_grad(), lm.teacher_mode():
        t_logits = _completion_logits(lm.model, t_in, L)
    s_logits = _completion_logits(lm.model, s_in, L)
    # If logits_to_keep was silently ignored these are full-sequence and misaligned.
    if s_logits.shape[1] != L or t_logits.shape[1] != L:
        s_logits = s_logits[:, -L:, :]
        t_logits = t_logits[:, -L:, :]

    loss, mean_w = _distil_loss(s_logits, t_logits, cfg)
    rec["_mean_token_weight"] = mean_w
    return loss, L


def keep_step(rec: dict, cfg=OPD_CFG) -> bool:
    """Cheap gates. Nothing here costs an extra generation."""
    if rec["completion_ids"] is None or len(rec["completion_ids"]) == 0:
        return False
    if rec["teacher_prompt"] is None:
        return False
    # A route with one legal choice carries no routing decision to distil.
    if rec["kind"] == "route" and len(rec["meta"].get("available", [])) < 2:
        return False
    if rec["kind"] == "insight" and not cfg.TRAIN_INSIGHT:
        return False
    if rec["kind"] == "route" and not cfg.TRAIN_ROUTE:
        return False
    return True
