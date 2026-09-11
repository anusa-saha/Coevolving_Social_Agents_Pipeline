"""The student: one backbone, LoRA on top, adapters toggled by role.

At evaluation the trained student plays the CHAIR while the untouched base plays the
advisors. Loading two copies of a 7B costs ~30 GB for no reason -- the adapter is the only
difference between them, so one backbone with the adapter switched on and off gives both,
and guarantees the advisors really are the frozen model rather than a second copy that
merely started equal.

Also doubles as a local expert during generation: `as_base()` is exactly the plain model.
"""
import contextlib
import os

import torch

import paths  # noqa: F401  -- puts the repo root on sys.path for csa_core
from csa_core import compat as compat


class Student(object):
    ADAPTER = 'student'

    def __init__(self, cfg, adapter_dir=None, grad_checkpointing=False):
        for p in compat.problems():
            raise SystemExit('cannot build the student: %s' % p)
        self.cfg = cfg
        self.tokenizer = compat.load_tokenizer(cfg.student_model)
        base = compat.load_causal_lm(cfg.student_model, cfg.dtype, cfg.device,
                                     trainable=True)

        from peft import LoraConfig, PeftModel, get_peft_model
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, bias='none',
                          task_type='CAUSAL_LM', target_modules=list(cfg.lora_targets))
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                self.model = PeftModel.from_pretrained(base, adapter_dir,
                                                       adapter_name=self.ADAPTER,
                                                       is_trainable=True)
            except TypeError:                        # older peft
                self.model = PeftModel.from_pretrained(base, adapter_dir,
                                                       adapter_name=self.ADAPTER)
                for n, p in self.model.named_parameters():
                    if 'lora_' in n:
                        p.requires_grad_(True)
            print('[student] loaded adapter %s' % adapter_dir)
        else:
            self.model = get_peft_model(base, lora, adapter_name=self.ADAPTER)
            self.model.print_trainable_parameters()

        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

        self.params = [p for p in self.model.parameters() if p.requires_grad]

    @property
    def device(self):
        return self.model.device

    @contextlib.contextmanager
    def as_base(self):
        """The untouched model. Advisors speak with this, so they are frozen by
        construction rather than by a promise."""
        with compat.adapter_disabled(self.model) as ok:
            if not ok:
                raise RuntimeError('this peft cannot disable adapters, so the advisors '
                                   'cannot be separated from the trained chair. '
                                   'Upgrade peft>=0.6.')
            yield self.model

    def forward_lm(self, ids, labels):
        return self.model(input_ids=ids, labels=labels)

    @torch.no_grad()
    def generate(self, prompt_text, max_new_tokens=96, temperature=None):
        enc = self.tokenizer([prompt_text], return_tensors='pt').to(self.device)
        kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(**enc, **kw)
        return self.tokenizer.decode(out[0][enc['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        try:
            self.model.save_pretrained(out_dir, selected_adapters=[self.ADAPTER])
        except TypeError:
            self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print('[student] saved %s' % out_dir)


class StudentExpert(object):
    """Adapter so a Student can drive OmegaEnv at evaluation time.

    The chair speaks with the adapter ON, everyone else with it OFF. That is the whole
    point of evaluation: measure the trained chair inside an unchanged environment.
    """

    def __init__(self, student, chair_name):
        self.student = student
        self.chair_name = chair_name
        self.name = student.cfg.student_model
        self.n_calls = 0

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        import compat as _c
        import prompts_om as P
        self.n_calls += 1
        text = _c.render_chat(self.student.tokenizer, P.to_chat(messages, speaker))
        if speaker == self.chair_name:
            return self.student.generate(text, max_new_tokens, temperature)
        with self.student.as_base():
            return self.student.generate(text, max_new_tokens, temperature)
