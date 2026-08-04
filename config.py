"""
Config: model names and gate settings.

- Challenger + Verifier: gpt-5.4, normal OpenAI client.
- Strong arm: GLM-5.2-FP8, served via Vultr Inference (OpenAI-compatible, custom base_url).
- Weak / lone arm: loaded directly on your GPU (see weak_arm_model.py).
"""
import os

# --- Challenger + Verifier: gpt-5.4, normal OpenAI API ---
CHALLENGER_MODEL = "gpt-5.4"
VERIFIER_MODEL = "gpt-5.4"

# --- Strong arm: GLM-5.2-FP8 via Vultr Inference (OpenAI-compatible endpoint) ---
STRONG_ARM_MODEL = "zai-org/GLM-5.2-FP8"
STRONG_ARM_BASE_URL = "https://api.vultrinference.com/v1"
STRONG_ARM_API_KEY_ENV = "VULTR_API_KEY"  # set this env var, never hardcode the key
STRONG_ARM_TEMPERATURE = 0.9
STRONG_ARM_MAX_TOKENS = 1500
# GLM's chat template supports a thinking mode; the strong arm wants the final action only,
# not a reasoning trace, so this is passed as extra_body={"chat_template_kwargs": {...}}.
STRONG_ARM_ENABLE_THINKING = False

# --- Weak / lone arm: loaded directly on your GPU, no server ---
WEAK_ARM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
WEAK_ARM_MAX_NEW_TOKENS = 4000  # generous: this model reasons in <think> before answering

# Sampling parameters, as specified:
WEAK_ARM_TEMPERATURE = 0.7
WEAK_ARM_TOP_P = 0.8
WEAK_ARM_TOP_K = 20
WEAK_ARM_MIN_P = 0.0
WEAK_ARM_PRESENCE_PENALTY = 1.5
WEAK_ARM_REPETITION_PENALTY = 1.0

# --- Rollout counts + exact-count gates ---
WEAK_ARM_ROLLOUTS = 4
STRONG_ARM_ROLLOUTS = 4
WEAK_ARM_MAX_PASS = 1      # at most this many of WEAK_ARM_ROLLOUTS may pass
STRONG_ARM_MIN_PASS = 3    # at least this many of STRONG_ARM_ROLLOUTS must pass

# --- Loop control ---
MAX_REFINEMENT_ROUNDS = 10

# --- Paths ---
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
