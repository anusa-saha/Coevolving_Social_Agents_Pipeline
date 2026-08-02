"""
Loads Qwen/Qwen3.5-9B directly into GPU memory and runs generation in-process.
No server, no base_url -- just the model, loaded once, called with model.generate().

Note: Qwen3.5 uses a hybrid architecture (Gated DeltaNet linear-attention layers
mixed with regular Gated Attention layers). This is new enough that it may not
be recognized by an older `transformers` release -- if `from_pretrained` fails
with an unrecognized-architecture error, upgrade with:
    pip install -U git+https://github.com/huggingface/transformers.git
`trust_remote_code=True` is passed below as a safety net in case the model
still ships custom modeling code at the time you run this.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import config

_tokenizer = None
_model = None


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Implements OpenAI/vLLM-style "presence_penalty": a flat, one-time penalty
    subtracted from the score of any token that has appeared at least once
    already in the sequence so far -- regardless of how many times.

    This is NOT the same thing as `transformers`' built-in `repetition_penalty`,
    which instead scales a token's logit every time it reappears. `transformers`
    has no native presence_penalty, so it's implemented here directly rather
    than approximated or silently dropped.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for i in range(input_ids.shape[0]):
            seen_token_ids = torch.unique(input_ids[i])
            scores[i, seen_token_ids] = scores[i, seen_token_ids] - self.penalty
        return scores


def _load():
    """Loads the model once, the first time it's needed."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.WEAK_ARM_MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            config.WEAK_ARM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model


def generate(messages: list, max_new_tokens: int = None) -> str:
    """
    messages: a list like [{"role": "system", "content": "..."}] (same shape you'd
    send to a chat API). Returns the model's raw text output, generated with the
    exact sampling parameters set in config.py.
    """
    tokenizer, model = _load()
    max_new_tokens = config.WEAK_ARM_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    logits_processor = LogitsProcessorList([
        PresencePenaltyLogitsProcessor(config.WEAK_ARM_PRESENCE_PENALTY),
    ])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=config.WEAK_ARM_TEMPERATURE,
            top_p=config.WEAK_ARM_TOP_P,
            top_k=config.WEAK_ARM_TOP_K,
            min_p=config.WEAK_ARM_MIN_P,
            repetition_penalty=config.WEAK_ARM_REPETITION_PENALTY,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
