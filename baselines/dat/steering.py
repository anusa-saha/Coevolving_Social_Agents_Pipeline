"""The frozen LM, wrapped so a continuous prefix can be prepended to any prompt.

Three operations, and nothing else:

    state(ids)                  g_theta(e): the last-layer, last-token hidden state.
                                This is the MDP's state -- the whole conversation
                                compressed by the transformer that is about to speak.
    generate(ids, prefix)       f_theta(. | z || e): controlled generation. The prefix
                                embeddings are concatenated in front of the prompt's
                                token embeddings, so the model sees L extra positions
                                that correspond to no token in its vocabulary.
    clone_loss(ids, tgt, pre)   Equation 5: the teacher-forced NLL of an utterance the
                                unsteered model produced, with the prefix in place.

Every parameter of the language model has requires_grad=False. The backward in
`clone_loss` runs through the network only to reach `prefix`, which is the entire trick:
the gradient touches 7B parameters' activations and updates none of them.

The one HF wrinkle worth stating: a decoder-only `generate(inputs_embeds=...)` returns
only the newly generated tokens, because there are no input ids to echo. Older releases
did echo. Rather than pin a version, the returned length is compared against
max_new_tokens -- anything longer than the generation budget must contain the prompt.
"""
import torch

import paths  # noqa: F401  -- puts the repo root and ppdpp/ on sys.path
from csa_core import compat
import prompts_dat as P


class SteeredLM(object):
    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        if model is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.tokenizer = compat.load_tokenizer(cfg.model)
            self.model = compat.load_causal_lm(cfg.model, cfg.dtype, cfg.device)
        # Frozen, explicitly. compat.load_causal_lm calls .eval(), which stops dropout
        # but leaves requires_grad=True -- and a self-clone backward would then allocate
        # a gradient for every one of the 7B parameters before anyone noticed.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed = self.model.get_input_embeddings()
        self.d_model = int(self.model.config.hidden_size)
        self.n_state_calls = 0
        self.n_generate_calls = 0

    @property
    def device(self):
        return self.model.device

    @property
    def param_dtype(self):
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------- encoding
    def render(self, messages, role):
        return compat.render_chat(self.tokenizer, P.to_chat(messages, role))

    def encode(self, messages, role, max_ctx=0):
        """Messages -> input ids on the model's device.

        `max_ctx` left-truncates, which clips the FRONT of the system prompt (the
        persona). It is an out-of-memory lever of last resort, not a default: keep it at
        0 and use gradient checkpointing instead.
        """
        text = self.render(messages, role)
        ids = self.tokenizer([text], return_tensors='pt').input_ids
        if max_ctx and ids.shape[1] > max_ctx:
            ids = ids[:, -max_ctx:]
        return ids.to(self.device)

    # ------------------------------------------------------------- state
    @torch.no_grad()
    def state(self, input_ids):
        """g_theta(e), as float32 on the CPU-friendly side of the planner.

        Read from the UNSTEERED prompt: the planner has to decide what to do from what
        the conversation is, not from what it already did to it. Feeding the steered
        prompt back in would close a loop the paper does not have.
        """
        self.n_state_calls += 1
        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.cfg.state_layer][0, -1]
        return h.detach().float()

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, prefix=None, greedy=True,
                 temperature=None):
        self.n_generate_calls += 1
        kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if greedy:
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True,
                      temperature=float(temperature or self.cfg.temperature))

        if prefix is None:
            out = self.model.generate(input_ids=input_ids, **kw)
            new = out[0][input_ids.shape[1]:]
            return self.tokenizer.decode(new, skip_special_tokens=True).strip()

        embeds = self._with_prefix(input_ids, prefix)
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        out = self.model.generate(inputs_embeds=embeds, attention_mask=attn, **kw)
        seq = out[0]
        # Only-new-tokens on current transformers; prompt-echoing on older ones. A return
        # longer than the generation budget can only be the echoing case.
        if seq.shape[0] > max_new_tokens:
            seq = seq[embeds.shape[1]:]
        return self.tokenizer.decode(seq, skip_special_tokens=True).strip()

    # ------------------------------------------------------------- stage 1 loss
    def clone_loss(self, prompt_ids, target_ids, prefix=None):
        """Equation 5, for one (prompt, utterance) pair. Gradient reaches `prefix` only.

        With prefix=None this is the unsteered NLL of the same utterance, which is the
        number the self-clone loss has to approach: the paper's claim for stage 1 is that
        the steered agent behaves like the unsteered one, and that is measurable rather
        than assumed.
        """
        e_p = self.embed(prompt_ids)
        e_t = self.embed(target_ids)
        parts = [e_p, e_t]
        n_pre = 0
        if prefix is not None:
            pre = prefix.to(e_p.dtype).unsqueeze(0)
            parts.insert(0, pre)
            n_pre = pre.shape[1]
        inp = torch.cat(parts, dim=1)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
        logits = self.model(inputs_embeds=inp, attention_mask=attn,
                            use_cache=False).logits
        start = n_pre + e_p.shape[1] - 1             # position predicting target[0]
        pred = logits[:, start:start + e_t.shape[1]]
        lp = torch.log_softmax(pred.float(), dim=-1) \
                  .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return -lp.mean()

    # ------------------------------------------------------------- memory
    def enable_gradient_checkpointing(self):
        """Needed for stage 1: the backward runs through the whole 7B to reach a prefix.

        use_reentrant=False is required -- the reentrant implementation drops the graph
        when no *parameter* requires grad, which is exactly this arm's situation, and the
        loss then has no gradient at all.
        """
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        except TypeError:                            # transformers < 4.35
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        return True

    def disable_gradient_checkpointing(self):
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:                            # noqa: BLE001
            pass
        self.model.config.use_cache = True

    # ------------------------------------------------------------- internals
    def _with_prefix(self, input_ids, prefix):
        e = self.embed(input_ids)
        pre = prefix.to(e.dtype)
        if pre.dim() == 2:
            pre = pre.unsqueeze(0)
        return torch.cat([pre.to(e.device), e], dim=1)
