"""The generator, local weights or an API endpoint, behind one call signature.

Every agent at the table is driven by the SAME backend instance. That matters: if half
the table were a frontier model and half a 7B, any conclusion about "how well do agents
pool information" would be confounded by which agent got which model. A mixed table is a
different experiment, and this arm does not run it.

Keys come from the environment. Never put one in a file.
"""
import os
import time

import _compat as compat

import prompts_rt as P


class LocalBackend(object):
    """Qwen-family weights via transformers, loaded once and shared by every agent."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.model
        self.tokenizer = compat.load_tokenizer(cfg.model)
        self.model = compat.load_causal_lm(cfg.model, cfg.dtype, cfg.device)
        self.n_calls = 0
        self.n_failed = 0

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        import torch
        self.n_calls += 1
        text = compat.render_chat(self.tokenizer, P.to_chat(messages, speaker))
        enc = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=self.tokenizer.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        return self.tokenizer.decode(out[0][enc['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()


class ApiBackend(object):
    """OpenAI-compatible endpoint. Reads OPENAI_API_KEY / OPENROUTER_API_KEY."""

    def __init__(self, cfg, max_retries=5):
        self.cfg = cfg
        self.name = cfg.api_model
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_failed = 0
        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise SystemExit(
                'no API key in the environment. Set OPENAI_API_KEY (or '
                'OPENROUTER_API_KEY) in your shell -- never in a file -- or use '
                '--backend local.')
        from openai import OpenAI
        kw = {'api_key': key}
        if cfg.api_base:
            kw['base_url'] = cfg.api_base
        self.client = OpenAI(**kw)

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        chat = P.to_chat(messages, speaker)
        last = None
        for attempt in range(self.max_retries):
            try:
                self.n_calls += 1
                r = self.client.chat.completions.create(
                    model=self.name, messages=chat, max_tokens=max_new_tokens,
                    temperature=(temperature if temperature is not None else 0.0))
                return (r.choices[0].message.content or '').strip()
            except Exception as e:                   # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 30))
        # An empty string is a valid, visible failure: the turn is recorded empty and the
        # verifier score reflects it, rather than the run dying partway through a split.
        self.n_failed += 1
        print('[backend] giving up on a call after %d attempts: %s'
              % (self.max_retries, last))
        return ''


def build(cfg):
    return LocalBackend(cfg) if cfg.backend == 'local' else ApiBackend(cfg)
