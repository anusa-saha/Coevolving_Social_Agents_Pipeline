"""rl.py - the agent policy, the reward, and the RLOO estimator.

WHAT CHANGED FROM train_V1

  1. CREDIT SPAN.  V1 scored the JSON action keyword and threw the text away. Measured
     over its 3,205 scored turns that keyword span had sequence probability 0.9994 - the
     gradient of log pi through it is ~0, which is why rl_loss sat at -0.0002 and
     clip_frac at 0.001 for seventeen hours. V2 scores the text (env.text_span_chars).

  2. BASELINE.  V1 used REINFORCE++ global batch normalisation over four DIFFERENT
     scenarios. Measured on the held-out 90, 84.2% of reward variance is BETWEEN
     scenarios (difficulty) and only 15.8% WITHIN (policy quality), so ~84% of every
     advantage it computed was noise about which scenario got drawn. V2 uses a
     leave-one-out baseline over G rollouts of the SAME scenario, where difficulty
     cancels exactly.

  3. NO std DIVISION.  Vanilla GRPO divides by the group standard deviation. In V1's
     eval 33% of scenarios showed near-zero spread across arms; dividing a small
     numerator by a small denominator manufactures a large advantage out of nothing.
     RLOO's leave-one-out mean is unbiased and needs no such division.

  4. NO TURN-CREDIT WEIGHTS.  V1 multiplied the reward by 0.25 for non-settle turns and
     1.0 for settle turns BEFORE centring. Because the reward is essentially never
     negative (measured minimum -0.02, mean 0.45), centring then split the batch by turn
     TYPE rather than by outcome: corr(advantage, is_settle) = 0.68 against
     corr(advantage, reward) = 0.29. Even inside episodes scoring above 0.9 every
     `reveal` turn received advantage -0.52 - seventeen hours spent instructing the
     policy to make its correct reveals less likely. V2: one reward, one advantage per
     rollout, applied to every span that rollout produced.

  5. NO RATIO, NO CLIP, NO REFERENCE FORWARD.  With one inner pass theta == theta_old, so
     the importance ratio is identically 1 and the clip is dead weight - V1 measured
     exactly that. Dropping it also removes two rollout-time forward passes per turn.

        L = - (1/N) * sum_spans  A_rollout * mean_t log pi(y_t | y_<t, s)
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# 1. CONFIG
# ============================================================

class RL_CFG:
    """Every agent-side knob. Nothing here is read from the command line."""

    # ---- model ----
    MODEL = "/scratch/rohank__iitp/Qwen3-8B"         # ~16.4 GiB bf16
    DEVICE = "cuda:0"               # DEFAULT ONLY - main.py / eval.py CONFIG owns this
    DTYPE = "bfloat16"
    ENABLE_THINKING = False

    # ---- LoRA ----
    # ONE adapter shared by every agent. The agents differ by PROMPT (their private facts
    # and role), not by weights: a per-agent adapter would need N times the memory and
    # would learn N times slower from the same trajectories.
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.0              # 0.0 REQUIRED: the policy that SCORES a span must be
                                    # the one that GENERATED it.
    LORA_TARGETS = "auto"           # discover every nn.Linear; portable across Qwen2.5 /
                                    # Qwen3 / Qwen3.5 hybrid stacks
    GRAD_CHECKPOINT = False         # see opd.OPD_CFG.GRAD_CHECKPOINT

    # ---- generation ----
    # Temperature matters more in V2 than in V1, because the object being sampled is now
    # the TEXT rather than the action keyword. A settlement at T=0.8 has measured
    # sequence probability ~0.0023, i.e. the G rollouts of a group genuinely differ.
    TEMPERATURE = 0.8
    TOP_P = 0.95
    TOP_K = 20
    MAX_SEQ_LEN = 4096
    AGENT_MAX_TOKENS = 160
    SETTLE_MAX_TOKENS = 400

    # ---- RLOO ----
    GROUP_SIZE = 4                  # rollouts per scenario. THE cost knob: wall time
                                    # scales linearly with it, and 4 is the smallest
                                    # value that gives a usable leave-one-out baseline.
    # ---- estimator (train_V2_grpo) ----
    # "grpo" = group-relative, mean over ALL G members (the sample is in its own
    #          baseline) and divided by the group standard deviation.
    # "rloo" = leave-one-out mean, no division. train_V2's default.
    # This is the ONLY intended difference between train_V2 and train_V2_grpo.
    ESTIMATOR = "grpo"

    # Denominator floor for GRPO. The canonical implementation divides by
    # (std + ADV_EPS). This value is load-bearing and must be reported: too small and a
    # group whose rewards differ by 0.001 produces a saturated +/-ADV_CLIP advantage out
    # of noise; too large and the division stops mattering, which quietly turns GRPO back
    # into RLOO-with-a-rescale and makes the ablation say nothing.
    ADV_EPS = 1e-4

    ADV_CLIP = 5.0                  # stops one pathological rollout owning a step
    LOGP_CHUNK = 64                 # sequence positions per log_softmax chunk

    # ---- episode ----
    T_MAX = 12
    MAX_TURNS_PER_AGENT = 3
    SEED_INSIGHT = True


@dataclass
class RewardConfig:
    """The reward composition, unchanged from train_V1 - it was never the problem.

        R_train =   w_S * 1[success]                    success_reward
                  + w_C * (content checks passed/all)   content_progress
                  + w_P * (prov.   checks passed/all)   provenance_progress
                  + w_D * (decisive facts revealed/all) reveal_progress
                  - min(|l_inv| * n_invalid, |cap|)     invalid_action_penalty / _cap
                  - |l_free| * n_unnecessary_free       free_action_penalty
                  - |l_turn| * max(0, T - T_min)        turn_penalty
                  - |l_to|   * 1[no settlement]         timeout_penalty

    Invariants (do not break these when retuning):

      * DOMINANCE. w_S > w_C + w_P + w_D, so farming partial credit can never beat
        finishing. Best possible FAILURE = 0.85; success = 1.85.
      * FRACTIONS, NOT COUNTS, for progress. A 13-check scenario is ~3x harder than a
        4-check one; rewarding raw counts would let scenario SIZE dominate the gradient.
      * CONTENT/PROVENANCE CREDIT IS GATED ON A SETTLEMENT EXISTING, which enforces
        settled-but-wrong > never-settled.
      * DENSE, NOT BINARY. Optimise checks_frac; REPORT success. Measured success is
        0.11, so a binary reward would leave ~89% of RLOO groups with identical members,
        zero spread and no gradient - G rollouts spent to teach nothing.
    """
    success_reward: float = 1.00
    content_progress: float = 0.50
    provenance_progress: float = 0.20
    reveal_progress: float = 0.15

    invalid_action_penalty: float = -0.10
    invalid_penalty_cap: float = -0.30
    free_action_penalty: float = -0.02
    turn_penalty: float = -0.01
    timeout_penalty: float = -0.10
    failed_settle_penalty: float = 0.0

    # 0.0, and it must stay 0.0: the leave-one-out group mean IS the baseline. V1 carried
    # a static shift AND batch normalisation, centring every advantage twice.
    return_shift: float = 0.0

    reveal_bonus: float = 0.0
    over_disclosure_penalty: float = 0.0
    free_penalty_only_when_reveal_available: bool = True

    def bounds(self) -> Tuple[float, float]:
        hi = (self.success_reward + self.content_progress
              + self.provenance_progress + self.reveal_progress)
        lo = (self.invalid_penalty_cap + self.timeout_penalty
              + 12.0 * self.turn_penalty + 12.0 * self.free_action_penalty)
        return lo, hi


REWARD = RewardConfig()


# ============================================================
# 2. MODEL PLUMBING
# ============================================================

def discover_lora_targets(model, exclude=("lm_head", "embed", "visual", "vision",
                                          "patch_embed", "merger")):
    """Every nn.Linear basename actually present in the model.

    Hardcoding ["q_proj","k_proj","v_proj","o_proj"] silently UNDER-ATTACHES on a hybrid
    stack. Qwen3.5 interleaves three Gated-DeltaNet (`linear_attention`) layers for every
    one full-attention layer, and the DeltaNet blocks have no q_proj/k_proj/v_proj/o_proj
    whatsoever - so that list would reach roughly a quarter of the layers, raise nothing,
    and quietly train with a quarter of the intended capacity.

    On a plain Qwen2.5/Qwen3 dense model this returns exactly
    {q,k,v,o,gate,up,down}_proj, i.e. the check_one recipe, unchanged.

    Returns (sorted_names, {name: how_many_layers_have_it}).
    """
    import torch.nn as nn
    counts = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(x in name for x in exclude):
            continue
        base = name.rsplit(".", 1)[-1]
        counts[base] = counts.get(base, 0) + 1
    return sorted(counts), counts


def resolve_lora_targets(model, cfg_targets, tag="lm"):
    """cfg_targets == "auto"  -> discover from the model (recommended, portable).
    An explicit list -> used as given, but any name absent from the model is reported;
    that warning is the difference between a 25%-attached adapter and a real one."""
    found, counts = discover_lora_targets(model)
    if cfg_targets == "auto" or cfg_targets == ["auto"]:
        print("[{}] LoRA targets (auto): {}".format(tag, found), flush=True)
        print("[{}]   layers per target: {}".format(
            tag, {k: counts[k] for k in found}), flush=True)
        return found
    want = list(cfg_targets)
    missing = [w for w in want if w not in counts]
    if missing:
        print("[{}] WARNING: LoRA targets not present in this model: {}".format(tag, missing),
              flush=True)
        print("[{}]          model actually has: {}".format(tag, found), flush=True)
        print("[{}]          set LORA_TARGETS = \"auto\" unless you mean to skip them.".format(tag),
              flush=True)
    hit = [w for w in want if w in counts]
    if not hit:
        raise RuntimeError(
            "[{}] none of LORA_TARGETS={} exist in {}. Set LORA_TARGETS = \"auto\".".format(
                tag, want, found))
    # COVERAGE, not name-existence, is the number that matters. On a hybrid stack
    # q/k/v/o_proj all exist - just only on the 1-in-4 full-attention layers - so a
    # name check reports "4 of 4 requested" while the adapter reaches ~16% of the model.
    cov = sum(counts[h] for h in hit)
    tot = sum(counts.values())
    print("[{}] LoRA targets: {}  ({} of {} requested, {}/{} linear modules = {:.0%})".format(
        tag, hit, len(hit), len(want), cov, tot, cov / tot if tot else 0), flush=True)
    if tot and cov / tot < 0.5:
        print("[{}] WARNING: this adapter reaches only {:.0%} of the model's linear "
              "layers.".format(tag, cov / tot), flush=True)
        print("[{}]          Typical of a hybrid attention stack. Set LORA_TARGETS = "
              "\"auto\".".format(tag), flush=True)
    return hit


def load_base_lm(model_id, dtype, device, tag="lm"):
    """Load the base model in the requested dtype - and VERIFY that it actually is.

    transformers renamed `torch_dtype` to `dtype` in v5, but from_pretrained SWALLOWS
    unknown keyword arguments instead of raising, so passing the wrong spelling does not
    fail loudly - it quietly loads float32. A 4B model in fp32 is ~16 GiB instead of ~8,
    and with two models resident on a 24 GiB card that is the whole difference between
    running and OOM. So: try the old name first (accepted by 4.x AND still by 5.x), then
    assert the dtype we actually got.
    """
    from transformers import AutoModelForCausalLM
    import transformers
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True)
        got = next(model.parameters()).dtype
        if got != dtype:
            print("[{}] WARNING: asked for {} but the model loaded as {} - casting. "
                  "(a dtype kwarg was ignored by transformers {})".format(
                      tag, dtype, got, transformers.__version__), flush=True)
            model = model.to(dtype)
        model = model.to(device)
        n = sum(p.numel() for p in model.parameters())
        print("[{}] {} loaded: {:,} params in {} -> {}  (transformers {})".format(
            tag, model_id, n, next(model.parameters()).dtype, device,
            transformers.__version__), flush=True)
        return model
    except (ValueError, KeyError) as exc:
        raise RuntimeError("\n".join([
            "[{}] cannot load {!r} with transformers {}.".format(
                tag, model_id, transformers.__version__),
            "  {}".format(exc),
            "",
            "  Qwen3 (model_type 'qwen3') needs transformers >= 4.51.",
            "      pip install -U 'transformers>=4.51,<5'",
            "  PIN BELOW 5: transformers v5 is a major break against older peft/torch,",
            "  and a bare 'pip install -U transformers' will pull it.",
            "",
            "  If you meant Qwen3.5 instead: it is a HYBRID stack (model_type",
            "  'qwen3_5'), needs transformers >= 5.2, and wants causal_conv1d + fla or",
            "  its Gated-DeltaNet path silently falls back to slow, memory-hungry ops.",
            "      pip install -U 'transformers>=5.2' causal-conv1d fla",
            "  Keep LORA_TARGETS = 'auto' there: an explicit q/k/v/o list reaches only",
            "  the 1-in-4 full-attention layers.",
        ])) from exc


# ============================================================
# 2. AGENT LM
# ============================================================

class AgentLM:
    """Base + ONE shared LoRA. Adapter on = the policy pi_theta.

    Adapter off (ref_mode) is the frozen base model. V2 does not use it during training -
    there is no KL anchor, because with a single inner pass and a leave-one-out baseline
    there is no reference term in the objective. eval.py uses it for the `vanilla` arm:
    LoRA's B matrix is zero-initialised, so adapter-off IS mathematically the untrained
    model and the baseline needs no second model load."""

    def __init__(self, cfg=RL_CFG, adapter_path=None, train=True):
        import torch
        from transformers import AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model

        self.torch = torch
        self.cfg = cfg
        dev = cfg.DEVICE
        if "cuda" in str(dev) and not torch.cuda.is_available():
            print("[agent] CUDA unavailable -> cpu", flush=True)
            dev = "cpu"
        self.device = dev
        self.calls = 0

        print("[agent] loading {} -> {}".format(cfg.MODEL, dev), flush=True)
        self.tok = AutoTokenizer.from_pretrained(cfg.MODEL, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Scoring wants right padding (it never pads at all); generation wants left.
        # generate_batch flips it per call and restores it, so this is only the default.
        self.tok.padding_side = "right"
        dtype = getattr(torch, cfg.DTYPE) if dev != "cpu" else torch.float32
        base = load_base_lm(cfg.MODEL, dtype, dev, "agent")
        for p in base.parameters():
            p.requires_grad = False
        targets = resolve_lora_targets(base, cfg.LORA_TARGETS, "agent")

        if adapter_path:
            print("[agent] adapter loaded from {}".format(adapter_path), flush=True)
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=train)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
                bias="none", task_type="CAUSAL_LM",
                target_modules=targets))

        if train and cfg.GRAD_CHECKPOINT and "cuda" in str(dev):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

        # See the note in opd.RouterLM: declare the training-forward behaviour instead of
        # being warned about it. generate()/generate_batch() pass use_cache=True
        # explicitly, so the rollout keeps its KV cache.
        try:
            self.model.config.use_cache = False
            if hasattr(self.model, "base_model"):
                self.model.base_model.model.config.use_cache = False
        except AttributeError:
            pass

        self.model.train() if train else self.model.eval()
        if cfg.LORA_DROPOUT:
            print("[agent] WARNING: LORA_DROPOUT={} means the scoring pass draws a "
                  "different dropout mask than generation did; the importance ratio will "
                  "not match the sampled trajectory".format(cfg.LORA_DROPOUT), flush=True)
        n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[agent] trainable params: {:,}  (thinking={})".format(
            n_tr, cfg.ENABLE_THINKING), flush=True)

    # ---------- adapter control ----------

    @contextlib.contextmanager
    def ref_mode(self):
        """Adapter OFF = the frozen reference policy."""
        with self.model.disable_adapter():
            yield

    # ---------- prompting ----------

    def render_prompt(self, user: str, system: str) -> str:
        """Applied ONCE. The rendered string is stored and reused verbatim in the update,
        so there is no re-templating drift between sampling and scoring."""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.cfg.ENABLE_THINKING)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    # ---------- sampling ----------

    def generate(self, user: str, system: str, max_new_tokens: int) -> Tuple[str, str]:
        torch = self.torch
        rendered = self.render_prompt(user, system)
        enc = self.tok(rendered, return_tensors="pt", truncation=True,
                       max_length=self.cfg.MAX_SEQ_LEN).to(self.device)
        self.calls += 1
        # eval() for rollout is a THROUGHPUT fix, not a semantic one: HF refuses a KV
        # cache when a train()-mode model has gradient checkpointing on, making every
        # token re-run a full forward. With LORA_DROPOUT=0.0 eval and train are
        # numerically identical, so this does not reintroduce a sampled-vs-scored mismatch.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=self.cfg.TEMPERATURE, top_p=self.cfg.TOP_P,
                    top_k=self.cfg.TOP_K, use_cache=True,
                    pad_token_id=self.tok.pad_token_id)
        finally:
            self.model.train(was_training)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return rendered, text

    def generate_batch(self, users, system: str, max_new_tokens: int):
        """N prompts -> N (rendered_prompt, completion_text) pairs, in ONE forward stack.

        This is where the wall time went. Autoregressive decode at batch 1 is entirely
        memory-bandwidth bound: the weights are streamed once per token regardless of how
        many sequences ride along, so a batch of 8 costs barely more than a batch of 1.
        The V1-shaped loop issued three batch-1 generate() calls per turn and measured
        152s per episode; the same work batched across a group is roughly 4x cheaper.

        LEFT padding, always. A decoder-only model continues from the last position, so
        right-padding would have every short prompt generating from a run of PAD tokens.
        """
        torch = self.torch
        rendered = [self.render_prompt(u, system) for u in users]
        prev_side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            enc = self.tok(rendered, return_tensors="pt", padding=True, truncation=True,
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
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=self.cfg.TEMPERATURE, top_p=self.cfg.TOP_P,
                    top_k=self.cfg.TOP_K, use_cache=True,
                    pad_token_id=self.tok.pad_token_id)
        finally:
            self.model.train(was_training)

        n_prompt = enc["input_ids"].shape[1]
        texts = [self.tok.decode(row[n_prompt:], skip_special_tokens=True)
                 for row in out]
        return list(zip(rendered, texts))

    # ---------- per-token log-probabilities ----------

    def span_token_range(self, continuation: str, span):
        """Char span -> token index range [a, b) over the tokenised continuation.

        Boundaries are resolved by tokenising the prefixes, which is exact whenever the
        span starts and ends on a token boundary and off by at most one token when a BPE
        merge straddles it. That is acceptable here: the spans are JSON string values, so
        both boundaries sit next to a quote character.
        """
        if span is None:
            return None
        s_ch, e_ch = span
        if e_ch <= s_ch:
            return None
        a = len(self.tok(continuation[:s_ch], add_special_tokens=False).input_ids)
        b = len(self.tok(continuation[:e_ch], add_special_tokens=False).input_ids)
        return (a, max(b, a + 1))

    def token_logprobs(self, rendered_prompt: str, continuation: str, span=None,
                       with_grad: bool = False, use_ref: bool = False,
                       want_entropy: bool = False):
        """Per-token log pi over the CREDIT SPAN of `continuation`.

        Returns a 1-D tensor of length n_scored (carrying grad if with_grad), or None if
        the span cannot be scored.

        span=None       -> score every generated token
        span=(a, b)     -> score only the tokens covering continuation[a:b]
        use_ref=True    -> score under the adapter-OFF reference policy
        want_entropy    -> also return per-token H(t), which ttpo.confident_error_mask
                           needs. It comes off the log-softmax we already computed, so
                           it costs one extra reduction and no extra forward pass.
                           Returns (logprobs, entropy) instead of logprobs.

        Only the span is softmaxed. train_V1 scored a prefix; V2 scores an interior
        window, which matters because the settlement object starts ~35 characters into
        the generation and the `content` string starts ~30 in.
        """
        torch = self.torch
        import torch.nn.functional as F

        # add_special_tokens=False: `rendered_prompt` already carries the chat template's
        # special tokens AS TEXT, so letting the tokeniser add its own would shift every
        # position by one and silently score the wrong tokens.
        p_ids = self.tok(rendered_prompt, return_tensors="pt",
                         add_special_tokens=False).input_ids
        c_ids = self.tok(continuation, return_tensors="pt",
                         add_special_tokens=False).input_ids
        n_cont = c_ids.shape[1]
        if n_cont == 0:
            return None

        rng = self.span_token_range(continuation, span) if span else (0, n_cont)
        if rng is None:
            return None
        a, b = max(0, rng[0]), min(n_cont, rng[1])
        if b <= a:
            return None

        full = torch.cat([p_ids, c_ids], dim=1).to(self.device)
        if full.shape[1] > self.cfg.MAX_SEQ_LEN:
            return None
        am = torch.ones_like(full)

        ctx = self.ref_mode() if use_ref else contextlib.nullcontext()
        grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
        with ctx, grad_ctx:
            try:
                out = self.model(input_ids=full, attention_mask=am,
                                 logits_to_keep=n_cont + 1)
                logits = out.logits[:, :-1, :]
            except TypeError:
                out = self.model(input_ids=full, attention_mask=am)
                logits = out.logits[:, -(n_cont + 1):-1, :]
            if logits.shape[1] != n_cont:
                logits = logits[:, -n_cont:, :]
            logits = logits[:, a:b, :]
            tgt = c_ids[:, a:b].to(self.device)

            # chunked: a [1, 400, 152k] float32 log_softmax is ~243MB, and with grad the
            # graph holds it for the whole backward.
            parts, ents = [], []
            n_score = b - a
            for i in range(0, n_score, self.cfg.LOGP_CHUNK):
                lg = logits[:, i:i + self.cfg.LOGP_CHUNK, :].float()
                lp = F.log_softmax(lg, dim=-1)
                parts.append(lp.gather(-1, tgt[:, i:i + self.cfg.LOGP_CHUNK]
                                       .unsqueeze(-1)).squeeze(-1).squeeze(0))
                if want_entropy:
                    with torch.no_grad():
                        ents.append(-(lp.exp() * lp).sum(-1).squeeze(0))
            out_lp = torch.cat(parts, dim=0)
            if want_entropy:
                return out_lp, torch.cat(ents, dim=0)
            return out_lp

    # ---------- io ----------

    def release(self):
        # device-scoped: see the note in opd.RouterLM.release
        if "cuda" in str(self.device):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ============================================================
# 3. REWARD
# ============================================================

@dataclass
class EpisodeOutcome:
    settled: bool = False
    content_results: Dict[str, bool] = field(default_factory=dict)
    provenance_results: Dict[str, bool] = field(default_factory=dict)
    invalid_action_count: int = 0
    valid_reveal_count: int = 0
    unnecessary_free_count: int = 0
    timed_out: bool = False
    episode_length: int = 0
    decisive_facts_total: int = 0
    decisive_facts_revealed: int = 0
    non_decisive_reveal_count: int = 0

    def content_fraction(self) -> float:
        if not self.settled or not self.content_results:
            return 0.0
        return sum(1 for v in self.content_results.values() if v) / len(self.content_results)

    def provenance_fraction(self) -> float:
        if not self.settled or not self.provenance_results:
            return 0.0
        return sum(1 for v in self.provenance_results.values() if v) / len(self.provenance_results)

    def reveal_fraction(self) -> float:
        if self.decisive_facts_total <= 0:
            return 0.0
        return min(1.0, self.decisive_facts_revealed / self.decisive_facts_total)

    def min_turns(self) -> int:
        """The shortest correct episode is one reveal per decisive fact, then the settle.
        Turns up to T_min are FREE - only stalling past it is penalised, which is what
        stops the turn penalty from ever encouraging a premature settle."""
        return max(1, self.decisive_facts_total + 1)

    def excess_turns(self) -> int:
        return max(0, self.episode_length - self.min_turns())


def compute_official_reward(o: EpisodeOutcome) -> int:
    """THE BENCHMARK METRIC. 1 iff a settlement exists and EVERY content check and EVERY
    provenance check passes. No shaping, ever. This is what the paper reports."""
    if not o.settled:
        return 0
    if not o.content_results and not o.provenance_results:
        return 0
    content_ok = all(o.content_results.values()) if o.content_results else True
    prov_ok = all(o.provenance_results.values()) if o.provenance_results else True
    return int(content_ok and prov_ok)


def compute_training_reward(o: EpisodeOutcome, cfg: RewardConfig = REWARD,
                            official: Optional[int] = None):
    """The shaped scalar used ONLY for optimisation. Returns (reward, breakdown).

    Why it is dense: content checks are 4-13 exact string equalities per scenario (mean
    6.9) and ALL must pass, so an untrained policy scores 0 on essentially every episode.
    With a binary R the whole batch gets the same advantage, batch normalisation divides
    by ~0, and nothing is learned.
    """
    if official is None:
        official = compute_official_reward(o)
    c, p, d = o.content_fraction(), o.provenance_fraction(), o.reveal_fraction()

    # capped: one chaotic 12-turn episode must not own the batch's advantage scale
    invalid_raw = cfg.invalid_action_penalty * o.invalid_action_count
    invalid_term = max(invalid_raw, cfg.invalid_penalty_cap)

    breakdown = {
        "success": cfg.success_reward * official,
        "content": cfg.content_progress * c,
        "provenance": cfg.provenance_progress * p,
        "reveal_progress": cfg.reveal_progress * d,
        "invalid": invalid_term,
        "free": cfg.free_action_penalty * o.unnecessary_free_count,
        "turns": cfg.turn_penalty * o.excess_turns(),
        "timeout": cfg.timeout_penalty * (1.0 if o.timed_out else 0.0),
        "failed_settle": cfg.failed_settle_penalty * (
            1.0 if (o.settled and not official) else 0.0),
        "reveal_bonus": cfg.reveal_bonus * o.valid_reveal_count,
        "over_disclosure": cfg.over_disclosure_penalty * o.non_decisive_reveal_count,
    }
    total = float(sum(breakdown.values())) - float(cfg.return_shift)
    breakdown.update({"_content_frac": c, "_provenance_frac": p, "_reveal_frac": d,
                      "_invalid_uncapped": invalid_raw,
                      "_excess_turns": float(o.excess_turns())})
    return total, breakdown




# ============================================================
# 4. RLOO
# ============================================================

def rloo_advantages(rewards, clip=None):
    """REINFORCE Leave-One-Out. One advantage per rollout in a group.

        A_i = r_i - mean_{j != i} r_j
            = (G / (G - 1)) * (r_i - mean_j r_j)

    Unbiased, needs no critic, and - unlike GRPO - performs NO division by the group
    standard deviation. That division is what would make GRPO unstable here: 33% of the
    held-out scenarios showed near-zero spread across policies, and dividing a tiny
    numerator by a tiny denominator manufactures a large advantage out of nothing.

    A group whose members all scored the same returns all zeros, which is correct - it
    carries no information about which rollout was better. Count those (see `degenerate`)
    rather than trying to rescue them.
    """
    g = len(rewards)
    if g < 2:
        return [0.0] * g
    tot = float(sum(rewards))
    out = [(float(r) - (tot - float(r)) / (g - 1)) for r in rewards]
    if clip:
        out = [max(-clip, min(clip, a)) for a in out]
    return out


def grpo_advantages(rewards, clip=None, eps=None):
    """Group Relative Policy Optimisation advantages.

        A_i = (r_i - mean_j r_j) / (std_j r_j + eps)

    TWO differences from `rloo_advantages`, and only the second one matters.

    1. THE MEAN INCLUDES THE SAMPLE ITSELF. Strictly this makes the baseline depend on the
       action being scored, so the estimator is biased where RLOO is not. In magnitude it
       is only a rescale: A_grpo_numerator = ((G-1)/G) * A_rloo, i.e. x0.75 at G=4. On its
       own this is indistinguishable from a learning-rate change and is NOT the
       interesting part of the comparison.

    2. THE DIVISION BY THE GROUP STANDARD DEVIATION. This is the real difference, and on
       this task it is expected to hurt. Two measurements from train_V1/train_V2 say why:

         * 33% of held-out scenarios showed near-zero spread across policies. Dividing a
           tiny numerator by a tiny denominator manufactures a large advantage out of
           what is essentially sampling noise - the update direction is then set by which
           rollout happened to pick up a rounding difference.
         * At G=4 the standard deviation is estimated from four samples (3 dof) and sits
           in the DENOMINATOR of every advantage, so its own estimation error is
           multiplied straight into the gradient.

       It is also the mechanism behind the difficulty bias that Dr. GRPO identifies:
       normalising by std systematically up-weights the easiest and hardest scenarios,
       which are exactly the ones carrying the least information about policy quality.

    Whether that reasoning is right is the point of this directory. Run it and compare.

    Degenerate groups (`degenerate()` is True) return all zeros rather than 0/eps, so the
    comparison against train_V2 is not contaminated by division-by-zero handling.
    """
    g = len(rewards)
    if g < 2:
        return [0.0] * g
    if degenerate(rewards):
        return [0.0] * g
    eps = RL_CFG.ADV_EPS if eps is None else eps
    vals = [float(r) for r in rewards]
    mean = sum(vals) / g
    var = sum((v - mean) ** 2 for v in vals) / (g - 1)      # sample std, matching GRPO
    std = var ** 0.5
    out = [(v - mean) / (std + eps) for v in vals]
    if clip:
        out = [max(-clip, min(clip, a)) for a in out]
    return out


def group_advantages(rewards, clip=None, estimator=None):
    """Dispatch on RL_CFG.ESTIMATOR so main.py has one call site for both variants."""
    est = (estimator or RL_CFG.ESTIMATOR).lower()
    if est == "grpo":
        return grpo_advantages(rewards, clip=clip)
    if est == "rloo":
        return rloo_advantages(rewards, clip=clip)
    raise ValueError("unknown ESTIMATOR {!r} (want 'grpo' or 'rloo')".format(est))


def group_std(rewards):
    """Sample standard deviation of a group's rewards. Logged per step so the
    manufactured-advantage failure mode is visible rather than inferred."""
    g = len(rewards)
    if g < 2:
        return 0.0
    vals = [float(r) for r in rewards]
    mean = sum(vals) / g
    return (sum((v - mean) ** 2 for v in vals) / (g - 1)) ** 0.5


def degenerate(rewards, eps=1e-9):
    """True when every rollout in the group scored the same, so the group teaches nothing.

    Watch the running fraction in train_steps.csv. Above ~0.3 the group is too small or
    the reward has saturated. In train_V1's eval 20 of 90 scenarios scored 0.000 in every
    arm - and all 20 were all-zero purely because nothing ever settled, which is exactly
    what env.ENV_CFG.FORCE_DM_* now prevents.
    """
    if len(rewards) < 2:
        return True
    return (max(rewards) - min(rewards)) <= eps


def reinforce_loss(logp, advantage, mask=None):
    """-A * mean_t log pi(y_t) for one span. The caller sums across spans.

    MEAN over the span's tokens, not sum: a 400-token settlement and a 20-token free line
    are one decision each, and summing would let settlement length set the effective
    learning rate. train_V1 summed, and settlements carried 65% of its token mass.

    `mask` is TTPO's confident-error mask, supplied by the caller for NEGATIVE-advantage
    spans only. Positives are reinforced across their whole span - there is nothing to
    limit when the update direction is already the one you want. The mean is taken over
    the SURVIVING tokens, so masking concentrates the penalty rather than shrinking it.
    """
    if mask is None:
        return -(float(advantage) * logp.mean())
    denom = mask.sum().clamp(min=1.0)
    return -(float(advantage) * (logp * mask).sum() / denom)
