import json
import os
import re
import time

from openai import OpenAI

import config

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=os.environ.get(config.OPENROUTER_API_KEY_ENV, ""),
)

EMPTY_CONTENT_RETRIES = 2
EMPTY_CONTENT_RETRY_BACKOFF_S = 2.0


def _create_with_content_retry(model: str, **create_kwargs) -> str:
    last_error = None
    for attempt in range(EMPTY_CONTENT_RETRIES + 1):
        resp = client.chat.completions.create(model=model, **create_kwargs)
        choice = resp.choices[0]
        if choice.message.content is not None:
            return choice.message.content
        last_error = ValueError(
            f"Model {model!r} returned no content (finish_reason={choice.finish_reason!r}) on "
            f"attempt {attempt + 1}/{EMPTY_CONTENT_RETRIES + 1}. This usually means the response "
            f"was refused or content-filtered, the call produced only a tool call with no text, or "
            f"it hit a length/stop condition before emitting any content. Full choice: {choice!r}"
        )
        if attempt < EMPTY_CONTENT_RETRIES:
            time.sleep(EMPTY_CONTENT_RETRY_BACKOFF_S * (attempt + 1))
    raise last_error


def gpt_chat(model: str, messages: list, temperature: float = 0.7,
             max_tokens: int = 2048, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return _create_with_content_retry(
        model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
    )


def strong_arm_chat(messages: list, temperature: float = 0.4, max_tokens: int = None,
                     reasoning_enabled: bool = None, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return _create_with_content_retry(
        config.STRONG_ARM_MODEL,
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


def weak_arm_chat(messages: list, temperature: float = 0.4, max_tokens: int = None,
                   reasoning_enabled: bool = None, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return _create_with_content_retry(
        config.WEAK_ARM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=config.WEAK_ARM_MAX_TOKENS if max_tokens is None else max_tokens,
        top_p=config.WEAK_ARM_TOP_P,
        presence_penalty=config.WEAK_ARM_PRESENCE_PENALTY,
        extra_body={
            "reasoning": {
                "enabled": config.WEAK_ARM_REASONING_ENABLED if reasoning_enabled is None else reasoning_enabled
            }
        },
        **kwargs,
    )


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[:text.find("<think>")]
    return text.strip()


def extract_json(text: str) -> dict:
    text = strip_reasoning(text)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end == -1:
        raise ValueError(
            f"JSON object appears truncated (found '{{' but no closing '}}' -- the response likely "
            f"hit max_tokens before finishing). Raw tail: {text[-300:]!r}"
        )
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    return json.loads(text[start:end + 1])