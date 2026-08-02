"""
Config: model names and gate settings. Nothing about endpoints here --
gpt-5.4 goes through the normal OpenAI client, and the weak-arm model is
loaded directly in-process on the GPU (see weak_arm_model.py).
"""
import os

# --- Challenger + Verifier + Strong arm: all gpt-5.4, normal OpenAI API ---
CHALLENGER_MODEL = "gpt-5.4"
VERIFIER_MODEL = "gpt-5.4"
STRONG_ARM_MODEL = "gpt-5.4"
STRONG_ARM_TEMPERATURE = 0.9

# --- Weak / lone arm: loaded directly on your GPU, no server ---
WEAK_ARM_MODEL_NAME = "Qwen/Qwen3.5-9B"
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
