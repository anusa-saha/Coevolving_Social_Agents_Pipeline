"""
gpt-5.4 client used by challenger.py, verifier.py, and strong_arm.py.
Just the normal OpenAI client -- reads OPENAI_API_KEY from the environment,
no custom base_url.
"""
import json
import re

from openai import OpenAI

client = OpenAI()  # requires OPENAI_API_KEY to be set in your environment


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


def strip_reasoning(text: str) -> str:
    """DeepSeek-R1-Distill emits a <think>...</think> block before its answer -- strip it."""
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
