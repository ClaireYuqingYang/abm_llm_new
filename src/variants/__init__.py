"""Three diffusion-simulation variants on the same network/seed pair.

  pure_abm  → rule-based agents (logistic decision rule).
  abm_llm   → same rules + per-(agent, story) LLM perception scores.
  pure_llm  → no logistic rule; per-(agent, story) LLM persona profile
              drives belief uptake and sharing.

All three additionally track parent pointers so we can reconstruct the
cascade tree and compute Vosoughi-style structural metrics
(depth, breadth, structural virality, size).
"""

from .runner import (
    VARIANTS,
    VariantConfig,
    default_variant_config,
    run_variant_experiment,
    run_all_variants,
)

__all__ = [
    "VARIANTS",
    "VariantConfig",
    "default_variant_config",
    "run_variant_experiment",
    "run_all_variants",
]
