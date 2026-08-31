"""One backbone, three roles.

The frozen meeting agents, the trained chair policy and the reward model are all the same
checkpoint. Loading three copies costs ~45 GB of weights in bf16, which does not fit on
the hardware this runs on. Loading it ONCE and switching LoRA adapters costs ~15 GB plus
a few hundred MB:

    agent   base weights, adapters disabled   <- genuinely frozen, by construction
    policy  'policy' adapter active           <- trained by BC, then GRPO
    reward  'rm' adapter + a scalar head      <- trained by MSE on attributed labels

Beyond the memory saving this is also more honest: the frozen agent is now provably the
untouched base model rather than a second copy that merely started equal.

The reward head is a linear layer on the last non-pad hidden state, not
AutoModelForSequenceClassification, because that class brings its own backbone and would
defeat the whole point.
"""
import contextlib
import json
import os

import torch
import torch.nn as nn

import compat

POLICY = 'policy'
REWARD = 'rm'


def _lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, bias='none',
                      task_type='CAUSAL_LM',
                      target_modules=list(cfg.lora_targets))


class SharedBackbone(object):
    """The single resident model. Everything else is a view onto it."""

    def __init__(self, cfg, device=None, adapters=(POLICY,), grad_checkpointing=False):
        for p in compat.problems():
            raise SystemExit('cannot build the backbone: %s' % p)
        self.cfg = cfg
        self.device_str = device or cfg.agent_device
        self.tokenizer = compat.load_tokenizer(cfg.agent_model)
        base = compat.load_causal_lm(cfg.agent_model, cfg.agent_dtype,
                                     self.device_str, trainable=True)
        self.hidden_size = base.config.hidden_size

        from peft import get_peft_model
        names = list(adapters)
        self.model = get_peft_model(base, _lora_config(cfg), adapter_name=names[0])
        for extra in names[1:]:
            self.model.add_adapter(extra, _lora_config(cfg))
        self.adapters = names
        self._active = names[0]

        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

        self.heads = nn.ModuleDict()
        self._report()

    def _report(self):
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        print('[backbone] %s on %s | adapters %s | trainable %.1fM / %.1fB'
              % (self.cfg.agent_model, self.device_str, self.adapters,
                 n_train / 1e6, n_all / 1e9))

    @property
    def device(self):
        return self.model.device

    # ------------------------------------------------------------- roles
    @contextlib.contextmanager
    def as_agent(self):
        """Base weights. This is what advisors and the unprompted chair speak with."""
        with compat.adapter_disabled(self.model) as ok:
            if not ok:
                raise RuntimeError(
                    'this peft cannot disable adapters, so the frozen agent cannot be '
                    'separated from the policy. Upgrade peft, or run with '
                    'shared_backbone=False.')
            yield self.model

    @contextlib.contextmanager
    def as_adapter(self, name):
        prev = self._active
        if name != prev:
            self.model.set_adapter(name)
            self._active = name
        try:
            yield self.model
        finally:
            if name != prev:
                self.model.set_adapter(prev)
                self._active = prev

    def trainable(self, name):
        """Parameters of ONE adapter, plus its head if it has one. Passing every
        trainable parameter to the optimizer would let a GRPO step silently update the
        reward model's adapter too."""
        ps = [p for n, p in self.model.named_parameters()
              if p.requires_grad and ('.%s.' % name) in n]
        if name in self.heads:
            ps += [p for p in self.heads[name].parameters() if p.requires_grad]
        if not ps:
            raise RuntimeError('no trainable parameters found for adapter %r' % name)
        return ps

    def add_head(self, name, dtype=torch.float32):
        head = nn.Linear(self.hidden_size, 1).to(self.device, dtype=dtype)
        self.heads[name] = head
        return head

    def set_trainable_adapter(self, name):
        self.model.set_adapter(name)

        for n, p in self.model.named_parameters():
            if ".policy." in n:
                p.requires_grad = (name == "policy")
            elif ".rm." in n:
                p.requires_grad = (name == "rm")

    # ------------------------------------------------------------- primitives
    @torch.no_grad()
    def generate(self, prompt_text, n=1, max_new_tokens=96, temperature=0.7,
                 greedy=False):
        enc = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if greedy or not temperature:
            kw.update(do_sample=False, num_return_sequences=1)
        else:
            kw.update(do_sample=True, temperature=temperature, num_return_sequences=n)
        out = self.model.generate(**enc, **kw)
        plen = enc['input_ids'].shape[1]
        texts, comps = [], []
        for o in out:
            c = o[plen:]
            keep = (c != self.tokenizer.pad_token_id).nonzero()
            c = c[:keep[-1].item() + 1] if len(keep) else c[:1]
            comps.append(c)
            texts.append(self.tokenizer.decode(c, skip_special_tokens=True).strip())
        return texts, comps, enc['input_ids'][0]

    def logprob(self, prompt_ids, comp_ids):
        """Teacher-forced mean log P(completion | prompt).

        .generate() returns no autograd graph, so sampled tokens are re-scored here. Valid
        for on-policy updates provided the step is taken before the policy moves.
        """
        ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(self.device)
        logits = self.model(ids).logits[:, :-1]
        lp = torch.log_softmax(logits.float(), dim=-1) \
                  .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        # logits[:, :-1] predicts ids[1:], so the first completion token sits at index
        # len(prompt) - 1. Getting this wrong trains on the scenario description.
        return lp[:, prompt_ids.shape[0] - 1:].mean()

    def value(self, ids, head_name=REWARD):
        """Scalar score for a token sequence, from the last position's hidden state."""
        t = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=t, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        head = self.heads[head_name]
        return head(h.to(head.weight.dtype)).squeeze(-1)

    # ------------------------------------------------------------- io
    def save_adapter(self, name, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with self.as_adapter(name):
            # `selected_adapters` only exists in newer peft. Without it save_pretrained
            # writes the ACTIVE adapter, which as_adapter has just made the right one --
            # so the fallback is correct, not merely tolerable.
            try:
                self.model.save_pretrained(out_dir, selected_adapters=[name])
            except TypeError:
                self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        if name in self.heads:
            torch.save(self.heads[name].state_dict(), os.path.join(out_dir, 'head.pt'))
        with open(os.path.join(out_dir, 'role.json'), 'w', encoding='utf-8') as f:
            json.dump({'adapter': name, 'base': self.cfg.agent_model,
                       'has_head': name in self.heads}, f, indent=1)
        print('[backbone] saved adapter %r -> %s' % (name, out_dir))

    def load_adapter(self, name, adapter_dir):
        if not adapter_dir or not os.path.isdir(adapter_dir):
            return False
        # Replace the slot rather than stacking a second copy under a new name. If this
        # peft cannot delete, loading into an occupied name raises -- surface that rather
        # than silently training a stale adapter.
        try:
            self.model.delete_adapter(name)
            if name in self.adapters:
                self.adapters.remove(name)
        except Exception:                            # noqa: BLE001
            pass
        try:
            self.model.load_adapter(adapter_dir, adapter_name=name, is_trainable=(name == "policy"))
            if name == "policy":
                self.model.set_adapter("policy")
        except (ValueError, KeyError) as e:
            raise RuntimeError(
                'could not load adapter %r from %s (%s). This peft may not support '
                'replacing an existing adapter; upgrade peft>=0.6.'
                % (name, adapter_dir, e))
        if name not in self.adapters:
            self.adapters.append(name)
        head_path = os.path.join(adapter_dir, 'head.pt')
        if os.path.exists(head_path):
            head = self.heads.get(name) or self.add_head(name)
            head.load_state_dict(torch.load(head_path, map_location=self.device))
        print('[backbone] loaded adapter %r <- %s' % (name, adapter_dir))
        return True


# ------------------------------------------------------------------ views
class PolicyView(object):
    """The chair. Thin wrapper so call sites read the same as before the refactor."""

    def __init__(self, backbone, adapter_dir=None, name=POLICY):
        self.bb = backbone
        self.name = name
        if adapter_dir:
            backbone.load_adapter(name, adapter_dir)
        self.tokenizer = backbone.tokenizer

    @property
    def params(self):
        return self.bb.trainable(self.name)

    @property
    def device(self):
        return self.bb.device

    @property
    def model(self):
        return self.bb.model

    def sample(self, prompt_text, n=1, max_new_tokens=96, temperature=0.9, greedy=False):
        with self.bb.as_adapter(self.name):
            return self.bb.generate(prompt_text, n=n, max_new_tokens=max_new_tokens,
                                    temperature=temperature, greedy=greedy)

    def logprob(self, prompt_ids, comp_ids):
        with self.bb.as_adapter(self.name):
            return self.bb.logprob(prompt_ids, comp_ids)

    def ref_logprob(self, prompt_ids, comp_ids):
        """The same quantity under the BASE weights -- free, because the base is already
        resident. Returns None if adapters cannot be disabled, so a KL of exactly zero is
        never mistaken for a working penalty."""
        try:
            with torch.no_grad(), self.bb.as_agent():
                return self.bb.logprob(prompt_ids, comp_ids).detach()
        except RuntimeError:
            return None

    def forward_lm(self, ids, labels):
        with self.bb.as_adapter(self.name):
            return self.bb.model(input_ids=ids, labels=labels)

    def train(self):
        self.bb.model.train()

    def eval(self):
        self.bb.model.eval()

    def save(self, out_dir):
        self.bb.save_adapter(self.name, out_dir)


class RewardView(object):
    """Scalar regressor over (state, action), sharing the backbone."""

    def __init__(self, backbone, adapter_dir=None, name=REWARD, cfg=None):
        self.bb = backbone
        self.name = name
        self.cfg = cfg or backbone.cfg
        if name not in backbone.adapters:
            backbone.model.add_adapter(name, _lora_config(backbone.cfg))
            backbone.adapters.append(name)
        if name not in backbone.heads:
            backbone.add_head(name)
        if adapter_dir:
            backbone.load_adapter(name, adapter_dir)
        self.tokenizer = backbone.tokenizer

    @property
    def params(self):
        return self.bb.trainable(self.name)

    @property
    def device(self):
        return self.bb.device

    def encode(self, prompt_text, completion, max_len=None):
        max_len = max_len or self.cfg.max_len
        p = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        c = self.tokenizer(completion, add_special_tokens=False).input_ids
        room = max(16, max_len - len(c))
        return p[-room:] + c[:max_len]

    def forward_ids(self, ids):
        with self.bb.as_adapter(self.name):
            return self.bb.value(ids, head_name=self.name)

    @torch.no_grad()
    def score(self, prompt_text, completion):
        return float(self.forward_ids(self.encode(prompt_text, completion)).item())

    def train(self):
        self.bb.model.train()

    def eval(self):
        self.bb.model.eval()

    def save(self, out_dir):
        self.bb.save_adapter(self.name, out_dir)
