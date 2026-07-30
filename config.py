"""
Central configuration for the pipeline.

Model assignment (per project constraints):
  - Challenger + Verifier -> gpt-5.4 (constraint #1)
  - Weak / lone arm       -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (constraint #2)
  - Strong arm (all agents) -> gpt-5.4 (constraint #3)

Everything is overridable via environment variables so you can point at whatever
OpenAI-compatible endpoint actually serves each model, without touching code.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Challenger + Verifier: gpt-5.4, called through an OpenAI-compatible API ---
    challenger_model: str = os.getenv("CHALLENGER_MODEL", "gpt-5.4")
    verifier_model: str = os.getenv("VERIFIER_MODEL", "gpt-5.4")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # --- Strong arm: every agent is played by gpt-5.4 ---
    strong_arm_model: str = os.getenv("STRONG_ARM_MODEL", "gpt-5.4")
    strong_arm_base_url: str = os.getenv("STRONG_ARM_BASE_URL", "https://api.openai.com/v1")
    strong_arm_api_key: str = os.getenv("STRONG_ARM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    strong_arm_temperature: float = float(os.getenv("STRONG_ARM_TEMPERATURE", "0.9"))
    strong_arm_max_tokens: int = int(os.getenv("STRONG_ARM_MAX_TOKENS", "1500"))

    # --- Weak arm: DeepSeek-R1-Distill-Qwen-7B, served behind any OpenAI-compatible
    #     endpoint (vLLM, TGI, Together, Fireworks, your own server, etc.) ---
    weak_arm_model: str = os.getenv("WEAK_ARM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    weak_arm_base_url: str = os.getenv("WEAK_ARM_BASE_URL", "http://localhost:8000/v1")
    weak_arm_api_key: str = os.getenv("WEAK_ARM_API_KEY", "EMPTY")
    weak_arm_temperature: float = float(os.getenv("WEAK_ARM_TEMPERATURE", "0.8"))
    weak_arm_max_tokens: int = int(os.getenv("WEAK_ARM_MAX_TOKENS", "3000"))

    # --- Rollout counts + exact-count gates ---
    weak_arm_rollouts: int = int(os.getenv("WEAK_ARM_ROLLOUTS", "4"))
    strong_arm_rollouts: int = int(os.getenv("STRONG_ARM_ROLLOUTS", "4"))
    weak_arm_max_pass: int = int(os.getenv("WEAK_ARM_MAX_PASS", "1"))     # at most this many of N may pass
    strong_arm_min_pass: int = int(os.getenv("STRONG_ARM_MIN_PASS", "3"))  # at least this many of N must pass

    # --- Loop control ---
    max_refinement_rounds: int = int(os.getenv("MAX_REFINEMENT_ROUNDS", "4"))

    # --- Paths ---
    prompts_dir: str = os.getenv("PROMPTS_DIR", os.path.join(os.path.dirname(__file__), "prompts"))
    output_dir: str = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))


CONFIG = Config()
