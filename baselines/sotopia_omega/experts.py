"""The generator model. Local weights or an API endpoint, behind one interface.

Omega's expert is GPT-4 in the paper and Qwen2.5-72B in the released code -- either way
much larger than the 9B student. That gradient is where a large part of their headline
("the student beats the teacher") comes from.

Both options are supported here because they answer different questions:

  LocalExpert   Qwen3.5-9B, same as the student. Gives up the distillation gradient and
                isolates the STRATEGY-INJECTION effect on its own. This is a cleaner
                ablation than the paper's, which confounds injection with teacher size.

  ApiExpert     a frontier model. Restores Omega's actual design, but the resulting
                corpus has a data advantage no other arm in this project has -- so a win
                would partly mean "the frontier model is better than Qwen3.5-9B", which
                is not a finding. Run it ALONGSIDE the local corpus, never instead.

The key is read from the environment. Never put one in a file.
"""
import os
import time


class LocalExpert(object):
    """Qwen-family weights via transformers, loaded once and reused."""

    def __init__(self, cfg, model=None, tokenizer=None):
        from csa_core import compat
        self.cfg = cfg
        self.name = cfg.expert_model
        if model is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.tokenizer = compat.load_tokenizer(cfg.expert_model)
            self.model = compat.load_causal_lm(cfg.expert_model, cfg.dtype, cfg.device)
        self.n_calls = 0

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        from csa_core import compat
        import torch
        import prompts_om as P
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


class ApiExpert(object):
    """OpenAI-compatible endpoint. Reads OPENAI_API_KEY / OPENROUTER_API_KEY."""

    def __init__(self, cfg, max_retries=5):
        self.cfg = cfg
        self.name = cfg.expert_model
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_failed = 0
        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise SystemExit(
                'no API key in the environment. Set OPENAI_API_KEY (or '
                'OPENROUTER_API_KEY) in your shell -- never in a file -- or use '
                '--expert local.')
        from openai import OpenAI
        kw = {'api_key': key}
        if cfg.api_base:
            kw['base_url'] = cfg.api_base
        self.client = OpenAI(**kw)

    def __call__(self, messages, speaker, max_new_tokens, temperature=None):
        import prompts_om as P
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
        # An empty string is a valid, visible failure: the turn is recorded as empty and
        # the episode's verifier score reflects it, rather than the run dying at hour six.
        self.n_failed += 1
        print('[expert] giving up on a call after %d attempts: %s'
              % (self.max_retries, repr(last)[:160]), flush=True)
        return ''


def build(cfg):
    if cfg.expert == 'local':
        return LocalExpert(cfg)
    if cfg.expert == 'api':
        return ApiExpert(cfg)
    raise ValueError('unknown expert %r' % cfg.expert)
