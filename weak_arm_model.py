"""
Loads DeepSeek-R1-Distill-Qwen-7B directly into GPU memory and runs generation
in-process. No server, no base_url -- just the model, loaded once, called with
model.generate().
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_tokenizer = None
_model = None


def _load():
    """Loads the model once, the first time it's needed."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.WEAK_ARM_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            config.WEAK_ARM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        _model.eval()
    return _tokenizer, _model


def generate(messages: list, temperature: float = None, max_new_tokens: int = None) -> str:
    """
    messages: a list like [{"role": "system", "content": "..."}] (same shape you'd
    send to a chat API). Returns the model's raw text output.
    """
    tokenizer, model = _load()
    temperature = config.WEAK_ARM_TEMPERATURE if temperature is None else temperature
    max_new_tokens = config.WEAK_ARM_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
