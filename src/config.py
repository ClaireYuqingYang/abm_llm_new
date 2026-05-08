"""
config.py
---------
Central configuration for the misinformation diffusion ABM.

Three variants share these parameters:
  * pure_abm  — rule-based logistic decision (the existing model).
  * abm_llm   — same logistic rule + per-(agent, story) LLM perception scores.
  * pure_llm  — LLM persona profile drives sampling (no logistic rule).
"""

import os

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "outputs")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Perception / persona engine ────────────────────────────────────────────────
# PERCEPTION_MODE options:
#   "mock"      → deterministic stand-in, no API calls (fast iteration)
#   "anthropic" → Claude  (ANTHROPIC_API_KEY in .env)
#   "openai"    → OpenAI  (OPENAI_API_KEY    in .env)
PERCEPTION_MODE = "openai"

# Specific model strings used when the corresponding mode is active.
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL    = "gpt-4.1-nano"

# ── Output ─────────────────────────────────────────────────────────────────────
CHART_DPI  = 150
SHOW_PLOTS = False        # set True for interactive display

# ── Simulation scale ───────────────────────────────────────────────────────────
SIM_N_AGENTS  = 500
SIM_N_STEPS   = 40
SIM_N_REPEATS = 30

# ── Network: homophilous small-world ───────────────────────────────────────────
NETWORK_MEAN_DEGREE = 10
NETWORK_REWIRE_PROB = 0.06
NETWORK_HOMOPHILY   = 0.70

# ── Initial seeding and content properties ────────────────────────────────────
INITIAL_FAKE_SEEDS = 8
INITIAL_TRUE_SEEDS = 8
FAKE_EMOTIONALITY      = 0.90
TRUE_EMOTIONALITY      = 0.45
FAKE_CREDIBILITY       = 0.55
TRUE_CREDIBILITY       = 0.78
FAKE_IDEOLOGICAL_SLANT = 0.65
TRUE_IDEOLOGICAL_SLANT = 0.10

# ── Intervention strengths (interpretable, not platform-calibrated) ────────────
NUDGE_STRENGTH             = 0.28
FRICTION_STRENGTH          = 0.36
FACT_CHECK_STRENGTH        = 0.42
DOWNRANK_STRENGTH          = 0.45
FRICTION_TRUE_NEWS_PENALTY = 0.10

INTERVENTION_POLICIES = [
    "none",
    "nudge",
    "friction",
    "fact_check_label",
    "downranking",
    "nudge_friction",
    "label_downranking",
]
