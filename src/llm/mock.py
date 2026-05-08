"""Deterministic mock fallback for when no LLM API is available."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .client import clip01


def mock_for(mode: str, agent: pd.Series, spec: dict, rng: np.random.Generator) -> dict:
    """Generate a plausible LLM-like output deterministically from the agent traits."""
    ideology = float(agent["ideology"])
    alignment = 1 - abs(ideology - spec["ideological_slant"]) / 2
    media_lit = float(agent["media_literacy"])
    trust = float(agent["platform_trust"])
    impulse = float(agent["impulsivity"])
    confirm = float(agent["confirmation_bias"])
    skepticism = float(agent["skepticism"])

    if mode == "hybrid_perception":
        importance = clip01(0.50 + 0.30 * alignment + 0.15 * confirm + rng.normal(0, 0.07))
        emotion = clip01(
            0.55 * (1 if spec["is_fake"] else 0.6)
            + 0.30 * confirm * alignment
            + 0.20 * impulse
            + rng.normal(0, 0.07)
        )
        relevance = clip01(0.40 + 0.40 * alignment + rng.normal(0, 0.07))
        return {
            "importance": importance,
            "emotional_intensity": emotion,
            "relevance": relevance,
        }

    # persona_decisions
    base_belief = (
        0.30 + 0.45 * alignment * confirm
        - 0.30 * skepticism * (1 if spec["is_fake"] else 0)
    )
    p_believe = clip01(base_belief + 0.20 * (1 - media_lit) + rng.normal(0, 0.07))
    share = (
        0.20
        + 0.45 * alignment
        + 0.25 * impulse
        + (0.20 if spec["is_fake"] else 0.05)
        - 0.30 * media_lit * (1 if spec["is_fake"] else 0)
    )
    share_propensity = clip01(share + rng.normal(0, 0.06))
    belief_lability = clip01(0.40 + 0.30 * impulse - 0.20 * media_lit + rng.normal(0, 0.06))
    resistance = clip01(0.30 + 0.40 * (1 - trust) + rng.normal(0, 0.06))
    return {
        "p_believe": p_believe,
        "share_propensity": share_propensity,
        "belief_lability": belief_lability,
        "resistance": resistance,
    }
