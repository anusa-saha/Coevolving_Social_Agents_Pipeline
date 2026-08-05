"""
LLM clients.

- `gpt_chat()` -> gpt-5.4, used by challenger.py and verifier.py. Normal OpenAI client,
  reads OPENAI_API_KEY from the environment, no custom base_url.
- `strong_arm_chat()` -> GLM-5.2, used by strong_arm.py. A second client pointed at
  OpenRouter's OpenAI-compatible endpoint, reads OPENROUTER_API_KEY from the environment.
"""
import json
import os
import re

from openai import OpenAI

import config

client = OpenAI()  # gpt-5.4: requires OPENAI_API_KEY to be set in your environment

strong_arm_client = OpenAI(
    api_key=os.environ.get(config.STRONG_ARM_API_KEY_ENV, ""),
    base_url=config.STRONG_ARM_BASE_URL,
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
        max_completion_tokens=max_tokens,  # newer models (incl. gpt-5.4) reject the old `max_tokens` name
        **kwargs,
    )
    return resp.choices[0].message.content


def strong_arm_chat(messages: list, temperature: float = None, max_tokens: int = None) -> str:
    """
    Calls GLM-5.2 via OpenRouter. OpenRouter's endpoint is a standard OpenAI-compatible
    /chat/completions surface, so this uses the ordinary `max_tokens` param (unlike gpt_chat's
    `max_completion_tokens`), and disables GLM's reasoning mode via extra_body -- the strong arm
    needs one action per turn, not a reasoning trace.
    """
    resp = strong_arm_client.chat.completions.create(
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
