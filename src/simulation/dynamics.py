"""Per-step exposure / belief / share primitives shared by all variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .content import Content, Policy


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def update_beliefs(
    agents: pd.DataFrame,
    receivers: np.ndarray,
    content: Content,
    policy: Policy,
    belief: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Bayes-flavoured belief update for one batch of receivers (pure-ABM rule)."""
    ideology = agents["ideology"].to_numpy()[receivers]
    media_literacy = agents["media_literacy"].to_numpy()[receivers]
    platform_trust = agents["platform_trust"].to_numpy()[receivers]
    confirmation = agents["confirmation_bias"].to_numpy()[receivers]
    skepticism = agents["skepticism"].to_numpy()[receivers]

    alignment = 1 - np.abs(ideology - content.ideological_slant) / 2
    persuasion = (
        0.28
        + 0.35 * content.credibility
        + 0.30 * content.emotionality
        + 0.34 * confirmation * alignment
        - 0.35 * skepticism * content.is_fake
        + rng.normal(0, 0.05, len(receivers))
    )

    if content.is_fake and policy.fact_check_label:
        correction = config.FACT_CHECK_STRENGTH * (
            0.40 + 0.60 * platform_trust + 0.30 * media_literacy
        )
        backfire = 0.12 * (1 - platform_trust) * (alignment > 0.80)
        persuasion -= correction
        persuasion += backfire

    persuasion = np.clip(persuasion, -0.25, 1.0)
    belief[receivers] = np.clip(0.72 * belief[receivers] + 0.28 * persuasion, 0, 1)


def choose_sharers(
    agents: pd.DataFrame,
    candidates: np.ndarray,
    content: Content,
    policy: Policy,
    belief: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Logistic share decision (pure-ABM rule).  Returns (sharer_mask, burden)."""
    sharers = np.zeros(len(agents), dtype=bool)
    if len(candidates) == 0:
        return sharers, 0.0

    media_literacy = agents["media_literacy"].to_numpy()[candidates]
    platform_trust = agents["platform_trust"].to_numpy()[candidates]
    impulsivity = agents["impulsivity"].to_numpy()[candidates]
    activity = agents["activity"].to_numpy()[candidates]
    ideology = agents["ideology"].to_numpy()[candidates]
    alignment = 1 - np.abs(ideology - content.ideological_slant) / 2

    logit = (
        -2.25
        + 2.65 * belief[candidates]
        + 1.00 * content.emotionality
        + 0.75 * impulsivity
        + 0.45 * (activity - 1)
        + 0.60 * alignment
        - 0.75 * media_literacy * content.is_fake
    )
    share_prob = sigmoid(logit)
    burden = 0.0

    if policy.nudge and content.is_fake:
        reduction = config.NUDGE_STRENGTH * (0.35 + 0.65 * platform_trust)
        share_prob *= 1 - reduction
        burden += 0.15 * len(candidates)

    if policy.friction:
        friction = config.FRICTION_STRENGTH
        if not content.is_fake:
            friction = config.FRICTION_TRUE_NEWS_PENALTY
        reduction = friction * (0.45 + 0.55 * impulsivity)
        share_prob *= 1 - reduction
        burden += (0.55 if content.is_fake else 0.22) * len(candidates)

    picked = candidates[rng.random(len(candidates)) < np.clip(share_prob, 0, 1)]
    sharers[picked] = True
    return sharers, burden


def snapshot(
    step: int,
    seen_fake: np.ndarray,
    seen_true: np.ndarray,
    fake_belief: np.ndarray,
    true_belief: np.ndarray,
    fake_frontier: np.ndarray,
    true_frontier: np.ndarray,
    cumulative_fake_shares: int,
    cumulative_true_shares: int,
    cumulative_burden: float,
) -> dict:
    """Per-step record used by every variant's time-series output."""
    return {
        "step": step,
        "fake_reach": float(seen_fake.mean()),
        "true_reach": float(seen_true.mean()),
        "fake_belief_rate": float((fake_belief >= 0.50).mean()),
        "true_belief_rate": float((true_belief >= 0.50).mean()),
        "mean_fake_belief": float(fake_belief.mean()),
        "mean_true_belief": float(true_belief.mean()),
        "fake_new_shares": int(fake_frontier.sum()),
        "true_new_shares": int(true_frontier.sum()),
        "cumulative_fake_shares": cumulative_fake_shares,
        "cumulative_true_shares": cumulative_true_shares,
        "intervention_burden": cumulative_burden,
    }
