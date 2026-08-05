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
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content


def strong_arm_chat(messages: list, temperature: float = 0.4, max_tokens: int = None,
                     reasoning_enabled: bool = None, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=config.STRONG_ARM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=config.STRONG_ARM_MAX_TOKENS if max_tokens is None else max_tokens,
        extra_body={
            "reasoning": {
                "enabled": config.STRONG_ARM_REASONING_ENABLED if reasoning_enabled is None else reasoning_enabled
            }
        },
        **kwargs,
    )
    return resp.choices[0].message.content


def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    text = strip_reasoning(text)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    return json.loads(text[start:end + 1])
