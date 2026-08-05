import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import config

_tokenizer = None
_model = None


class PresencePenaltyLogitsProcessor(LogitsProcessor):
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


def generate(messages: list, max_new_tokens: int = None, temperature: float = 0.4) -> str:
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
            temperature=temperature,
            top_p=config.WEAK_ARM_TOP_P,
            top_k=config.WEAK_ARM_TOP_K,
            min_p=config.WEAK_ARM_MIN_P,
            repetition_penalty=config.WEAK_ARM_REPETITION_PENALTY,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
