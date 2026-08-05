"""
LLM clients.

Everything routes through ONE OpenAI-compatible client pointed at OpenRouter -- same base_url,
same API key, for every role:
  - `gpt_chat()`        -> used by challenger.py and verifier.py (model="openai/gpt-5.4")
  - `strong_arm_chat()` -> used by strong_arm.py (model="z-ai/glm-5.2")

Reads OPENROUTER_API_KEY from the environment. Never hardcode the key here or anywhere else.
"""
import json
import os
import re

from openai import OpenAI

import config

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=os.environ.get(config.OPENROUTER_API_KEY_ENV, ""),
)


def gpt_chat(model: str, messages: list, temperature: float = 0.7,
             max_tokens: int = 2048, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,  # OpenRouter's standard param, regardless of underlying model
        **kwargs,
    )
    return resp.choices[0].message.content


def strong_arm_chat(messages: list, temperature: float = None, max_tokens: int = None) -> str:
    """
    Calls GLM-5.2 via OpenRouter. Disables GLM's reasoning mode via extra_body -- the strong arm
    needs one action per turn, not a reasoning trace.
    """
    resp = client.chat.completions.create(
        model=config.STRONG_ARM_MODEL,
        messages=messages,
        temperature=config.STRONG_ARM_TEMPERATURE if temperature is None else temperature,
        max_tokens=config.STRONG_ARM_MAX_TOKENS if max_tokens is None else max_tokens,
        extra_body={"reasoning": {"enabled": config.STRONG_ARM_REASONING_ENABLED}},
    )
    return resp.choices[0].message.content


def strip_reasoning(text: str) -> str:
    """Strip a <think>...</think> block if one is present -- a safety net in case a model's
    reasoning-suppression setting doesn't fully apply server-side."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of raw model text (after stripping reasoning/markdown)."""
    text = strip_reasoning(text)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    return json.loads(text[start:end + 1])
