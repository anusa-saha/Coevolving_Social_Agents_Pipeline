"""
Thin wrappers around the OpenAI-compatible chat completions API.

Using the same client class for every model works because:
  - gpt-5.4 (challenger, verifier, strong arm) is called through OpenAI's API directly.
  - DeepSeek-R1-Distill-Qwen-7B (weak arm) is assumed to be served behind any
    OpenAI-compatible endpoint (vLLM, TGI, Together, Fireworks, etc.) — only the
    base_url and api_key differ, per config.py.
"""
import json
import re

from openai import OpenAI

from config import CONFIG


def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


# One client per backend. challenger/verifier/strong-arm share the gpt-5.4 backend;
# weak arm points at wherever DeepSeek-R1-Distill-Qwen-7B is actually served.
challenger_client = get_client(CONFIG.openai_base_url, CONFIG.openai_api_key)
verifier_client = challenger_client
strong_arm_client = get_client(CONFIG.strong_arm_base_url, CONFIG.strong_arm_api_key)
weak_arm_client = get_client(CONFIG.weak_arm_base_url, CONFIG.weak_arm_api_key)


def chat_completion(client: OpenAI, model: str, messages: list, temperature: float = 0.7,
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


def strip_reasoning(text: str) -> str:
    """
    DeepSeek-R1-Distill models emit a <think>...</think> reasoning block before their
    final answer. Strip it before trying to parse JSON out of the response.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """
    Best-effort extraction of a single JSON object from raw model output:
    strips reasoning blocks, strips markdown code fences, then grabs the
    outermost {...} span and parses it.
    """
    text = strip_reasoning(text)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]!r}")
    return json.loads(text[start:end + 1])
