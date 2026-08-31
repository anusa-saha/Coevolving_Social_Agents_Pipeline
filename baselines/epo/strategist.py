"""LLM_s -- the trained strategic reasoner. LoRA over a causal LM.

The one model that gets gradients. LLM_d (the meeting agents) and the PRM stay frozen.

Two departures from vanilla EPO, both forced by budget:

  LoRA, not full fine-tuning. EPO full-FTs Llama3-8B on ~2050 SOTOPIA episodes; this
  runs on 99 scenarios, where a full 8B update memorises.

  lr 3e-5, not 1e-6. EPO's figure is a full-FT learning rate; applied to LoRA adapters
  it barely moves them.

The generate-then-rescore pattern matters: .generate() returns no autograd graph, so the
sampled tokens are re-scored with a teacher-forced forward pass before backward. That is
valid for REINFORCE because the step is taken before the policy moves.
"""
import os

import torch

import compat
import config                                        # noqa: F401  (sys.path shim)
import prompt_epo as pe


class Sample(object):
    """One strategist action, kept until the episode's rewards are known."""

    __slots__ = ('prompt_ids', 'comp_ids', 'text', 'tag_len')

    def __init__(self, prompt_ids, comp_ids, text, tag_len):
        self.prompt_ids = prompt_ids
        self.comp_ids = comp_ids
        self.text = text
        self.tag_len = tag_len


class EPOStrategist(object):
    def __init__(self, cfg, adapter_dir=None):
        self.cfg = cfg
        for p in compat.problems():
            raise SystemExit('cannot build the strategist: %s' % p)
        self.tokenizer = compat.load_tokenizer(cfg.strategist_model)
        base = compat.load_causal_lm(cfg.strategist_model, 'bfloat16',
                                     cfg.strategist_device, trainable=True)

        from peft import LoraConfig, PeftModel, get_peft_model
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                self.policy = PeftModel.from_pretrained(base, adapter_dir,
                                                        is_trainable=True)
            except TypeError:                        # peft < 0.5 has no is_trainable
                self.policy = PeftModel.from_pretrained(base, adapter_dir)
                for n, p in self.policy.named_parameters():
                    if 'lora_' in n:
                        p.requires_grad_(True)
            print('[strategist] loaded adapter from %s' % adapter_dir)
        else:
            self.policy = get_peft_model(base, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                bias='none', task_type='CAUSAL_LM',
                target_modules=list(cfg.lora_targets)))
            self.policy.print_trainable_parameters()

        self.params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.lr, eps=1e-6,
                                           weight_decay=0.0)
        self.scheduler = None
        self._pending = 0

    # ------------------------------------------------------------- rollout
    def build_prompt(self, case, conversation, prior_acts=()):
        msgs = pe.strategist_messages(case, conversation, prior_acts)
        return compat.render_chat(self.tokenizer, msgs)

    @torch.no_grad()
    def act(self, case, conversation, prior_acts=(), is_test=False):
        """Sample one strategy. Returns (text, Sample or None)."""
        text = self.build_prompt(case, conversation, prior_acts)
        enc = self.tokenizer(text, return_tensors='pt').to(self.policy.device)
        kw = dict(max_new_tokens=self.cfg.strategy_max_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if is_test:
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, temperature=self.cfg.strategy_temperature)
        out = self.policy.generate(**enc, **kw)
        comp = out[0][enc['input_ids'].shape[1]:]
        # drop trailing pad/eos so they do not dominate a 15-token mean
        keep = (comp != self.tokenizer.pad_token_id).nonzero()
        comp = comp[:keep[-1].item() + 1] if len(keep) else comp[:1]
        decoded = self.tokenizer.decode(comp, skip_special_tokens=True).strip()

        if is_test:
            return decoded, None
        tau, _sigma, _ok = pe.parse_strategy(decoded)
        tag_len = max(1, len(self.tokenizer(tau + ':', add_special_tokens=False).input_ids))
        return decoded, Sample(enc['input_ids'][0], comp, decoded, tag_len)

    # ------------------------------------------------------------- learning
    def accumulate(self, samples, advantages):
        """Backward for one episode. EPO objective, token-averaged then turn-averaged.

        The act tag is ~2 of ~15 tokens but drives every branch in the environment, so
        those positions are optionally upweighted (cfg.tag_weight). Vanilla EPO has no
        such term -- it has no symbolic tag to protect.
        """
        T = max(1, len(samples))
        total = 0.0
        for s, adv in zip(samples, advantages):
            if s is None or adv == 0.0:
                continue
            ids = torch.cat([s.prompt_ids, s.comp_ids]).unsqueeze(0).to(self.policy.device)
            logits = self.policy(ids).logits[:, :-1]
            lp = torch.log_softmax(logits.float(), dim=-1) \
                      .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # logits[:, :-1] predicts ids[1:], so the first completion token sits at
            # index len(prompt) - 1. Getting this wrong trains on the scenario text.
            lp = lp[:, s.prompt_ids.shape[0] - 1:]

            w = torch.ones_like(lp)
            w[:, :min(s.tag_len, w.shape[1])] = self.cfg.tag_weight
            logp = (lp * w).sum() / w.sum()

            loss = -(adv * logp) / T
            if self.cfg.kl_beta:
                kl = self._kl_to_ref(ids, s, lp)
                if kl is not None:
                    loss = loss + self.cfg.kl_beta * kl / T
            loss.backward()
            total += float(loss.detach())
        self._pending += 1
        return total

    def _kl_to_ref(self, ids, s, lp):
        """KL against the frozen base. PEFT gives the reference model for free by
        disabling the adapters -- no second copy in memory.

        Vanilla EPO reports no KL term. At 175 optimizer steps on 99 scenarios an
        unconstrained LM policy can collapse onto a single strategy string, so this is
        insurance; set kl_beta = 0 to reproduce the paper exactly.

        Returns None if this peft cannot disable adapters. A KL computed against the
        policy itself is identically zero, which would silently disable the term while
        looking like it worked -- better to skip it loudly once.
        """
        with torch.no_grad():
            with compat.adapter_disabled(self.policy) as available:
                if not available:
                    if not getattr(self, '_kl_warned', False):
                        print('[strategist] peft cannot disable adapters; KL term is off. '
                              'Upgrade peft or pass --kl_beta 0 to silence this.')
                        self._kl_warned = True
                    return None
                ref = self.policy(ids).logits[:, :-1]
                ref_lp = torch.log_softmax(ref.float(), dim=-1) \
                              .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                ref_lp = ref_lp[:, s.prompt_ids.shape[0] - 1:]
        return (lp - ref_lp).mean()

    def maybe_step(self, force=False):
        """One optimizer step per cfg.episodes_per_update episodes."""
        if not force and self._pending < self.cfg.episodes_per_update:
            return None
        if self._pending == 0:
            return None
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = 0
        return float(gn)

    def set_schedule(self, total_steps):
        self.scheduler = compat.linear_schedule(
            self.optimizer, int(self.cfg.warmup_frac * total_steps), total_steps)

    # ------------------------------------------------------------- io
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.policy.save_pretrained(out_dir)          # adapters only, ~150-300 MB
        self.tokenizer.save_pretrained(out_dir)
        print('[strategist] saved %s' % out_dir)
