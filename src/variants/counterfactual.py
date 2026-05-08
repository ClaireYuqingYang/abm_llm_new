"""Trait / content shifts applied per-run for counterfactual experiments.

Supported override keys:
    media_literacy_shift       → add this to every agent's media_literacy
    skepticism_shift           → add this to every agent's skepticism
    confirmation_bias_shift    → add this to every agent's confirmation_bias
    platform_trust_shift       → add this to every agent's platform_trust
    impulsivity_shift          → add this to every agent's impulsivity
    emotionality_shift         → add to FAKE_ and TRUE_ EMOTIONALITY
    fake_emotionality_shift    → add to FAKE_EMOTIONALITY only
    true_emotionality_shift    → add to TRUE_EMOTIONALITY only

All values are clipped to valid ranges after the shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_AGENT_TRAIT_SHIFTS = {
    "media_literacy_shift": ("media_literacy", 0.0, 1.0),
    "skepticism_shift": ("skepticism", 0.0, 1.0),
    "confirmation_bias_shift": ("confirmation_bias", 0.0, 1.0),
    "platform_trust_shift": ("platform_trust", 0.0, 1.0),
    "impulsivity_shift": ("impulsivity", 0.0, 1.0),
}


def apply_agent_cf(agents: pd.DataFrame, cf_overrides: dict) -> pd.DataFrame:
    """Apply trait shifts to a fresh copy of the agent dataframe."""
    if not cf_overrides:
        return agents
    out = agents.copy()
    for key, (col, lo, hi) in _AGENT_TRAIT_SHIFTS.items():
        if key in cf_overrides:
            out[col] = np.clip(out[col].to_numpy() + float(cf_overrides[key]), lo, hi)
    return out


def shifted_emotionality(base_value: float, cf_overrides: dict, story: str) -> float:
    """Return the emotionality of a story after applying CF overrides."""
    if not cf_overrides:
        return base_value
    shift = float(cf_overrides.get("emotionality_shift", 0.0))
    if story == "fake":
        shift += float(cf_overrides.get("fake_emotionality_shift", 0.0))
    else:
        shift += float(cf_overrides.get("true_emotionality_shift", 0.0))
    return float(np.clip(base_value + shift, 0.0, 1.0))
