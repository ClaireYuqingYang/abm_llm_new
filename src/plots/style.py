"""Shared palette / labels for variant-comparison plots."""

VARIANT_COLORS = {
    "pure_abm": "#1f77b4",
    "abm_llm":  "#2ca02c",
    "pure_llm": "#d62728",
}

VARIANT_LABELS = {
    "pure_abm": "Pure ABM",
    "abm_llm":  "ABM + LLM (hybrid)",
    "pure_llm": "Pure LLM persona",
}

VARIANT_ORDER = ("pure_abm", "abm_llm", "pure_llm")
