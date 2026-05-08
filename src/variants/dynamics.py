"""Variant-aware exposure / belief / share dispatch.

The pure-ABM rules live in ``src.simulation.dynamics``; this module
extends or replaces them for the abm_llm and pure_llm variants and adds
parent-tracking exposure so cascades can be reconstructed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..simulation.content import Content, Policy
from ..simulation import dynamics as base


def send_exposures_track(
    sharers: np.ndarray,
    content: Content,
    policy: Policy,
    neighbors: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Push exposures from `sharers` to neighbors, tracking the *first* exposer
    of each receiver.  Returns (per-receiver exposure count, first-exposer pid).
    """
    exposures = np.zeros(n, dtype=int)
    first_exposer = np.full(n, -1, dtype=int)
    source_ids = np.flatnonzero(sharers)
    rng.shuffle(source_ids)  # fairness across simultaneous sharers
    exposure_prob = 0.58
    if content.is_fake and policy.downranking:
        exposure_prob *= 1 - config.DOWNRANK_STRENGTH
    for source in source_ids:
        targets = neighbors[source]
        if len(targets) == 0:
            continue
        delivered = targets[rng.random(len(targets)) < exposure_prob]
        for t in delivered:
            if exposures[t] == 0:
                first_exposer[t] = int(source)
            exposures[t] += 1
    return exposures, first_exposer


def update_beliefs_variant(
    variant: str,
    agents: pd.DataFrame,
    receivers: np.ndarray,
    content: Content,
    policy: Policy,
    belief: np.ndarray,
    exposure_count: np.ndarray,
    llm_data: dict | None,
    rng: np.random.Generator,
) -> None:
    """Dispatch belief update to the appropriate variant rule."""
    if variant == "pure_abm":
        base.update_beliefs(agents, receivers, content, policy, belief, rng)
        return

    if variant == "abm_llm":
        # Standard rule + an LLM-driven affect bump (emotion / personal relevance).
        story = "fake" if content.is_fake else "true"
        perc = llm_data["perception"][story]
        emo = np.array([perc[int(p)]["emotional_intensity"] for p in receivers])
        rel = np.array([perc[int(p)]["relevance"] for p in receivers])
        base.update_beliefs(agents, receivers, content, policy, belief, rng)
        delta_llm = 0.18 * (emo - 0.5) + 0.10 * (rel - 0.5)
        if content.is_fake:
            skepticism = agents["skepticism"].to_numpy()[receivers]
            delta_llm *= 1.0 - 0.5 * skepticism
        belief[receivers] = np.clip(belief[receivers] + delta_llm * 0.25, 0, 1)
        return

    if variant == "pure_llm":
        story = "fake" if content.is_fake else "true"
        persona = llm_data["persona"][story]
        p_believe = np.array([persona[int(p)]["p_believe"] for p in receivers])
        lability = np.array([persona[int(p)]["belief_lability"] for p in receivers])
        resistance = np.array([persona[int(p)]["resistance"] for p in receivers])

        target = p_believe.copy()
        if content.is_fake and policy.fact_check_label:
            # Corrective force scaled by (1 - resistance).  High-resistance
            # personas barely move; low-resistance ones drop sharply.
            target = target * resistance + 0.05 * (1 - resistance)
        belief[receivers] = np.clip(
            belief[receivers] + lability * (target - belief[receivers])
            + rng.normal(0, 0.04, len(receivers)),
            0, 1,
        )
        return

    raise ValueError(f"Unknown variant: {variant}")


def choose_sharers_variant(
    variant: str,
    agents: pd.DataFrame,
    candidates: np.ndarray,
    content: Content,
    policy: Policy,
    belief: np.ndarray,
    exposure_count: np.ndarray,
    llm_data: dict | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Dispatch share decision to the appropriate variant rule."""
    n = len(agents)
    sharers = np.zeros(n, dtype=bool)
    if len(candidates) == 0:
        return sharers, 0.0

    if variant == "pure_abm":
        return base.choose_sharers(agents, candidates, content, policy, belief, rng)

    if variant == "abm_llm":
        # Base logit + LLM importance / relevance bump.
        story = "fake" if content.is_fake else "true"
        perc = llm_data["perception"][story]
        importance = np.array([perc[int(p)]["importance"] for p in candidates])
        relevance = np.array([perc[int(p)]["relevance"] for p in candidates])

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
            + 0.85 * (importance - 0.5)
            + 0.55 * (relevance - 0.5)
            - 0.75 * media_literacy * content.is_fake
        )
        share_prob = base.sigmoid(logit)
        burden = _apply_policy_costs(
            share_prob, policy, content, platform_trust, impulsivity, candidates
        )
        picked = candidates[rng.random(len(candidates)) < np.clip(share_prob, 0, 1)]
        sharers[picked] = True
        return sharers, burden

    if variant == "pure_llm":
        story = "fake" if content.is_fake else "true"
        persona = llm_data["persona"][story]
        propensity = np.array([persona[int(p)]["share_propensity"] for p in candidates])
        resistance = np.array([persona[int(p)]["resistance"] for p in candidates])
        platform_trust = agents["platform_trust"].to_numpy()[candidates]
        impulsivity = agents["impulsivity"].to_numpy()[candidates]

        share_prob = np.clip(propensity * (0.4 + 0.6 * belief[candidates]), 0, 1)
        burden = _apply_policy_costs(
            share_prob, policy, content, platform_trust, impulsivity, candidates,
            extra_resistance=resistance,
        )
        picked = candidates[rng.random(len(candidates)) < np.clip(share_prob, 0, 1)]
        sharers[picked] = True
        return sharers, burden

    raise ValueError(f"Unknown variant: {variant}")


def _apply_policy_costs(
    share_prob: np.ndarray,
    policy: Policy,
    content: Content,
    platform_trust: np.ndarray,
    impulsivity: np.ndarray,
    candidates: np.ndarray,
    extra_resistance: np.ndarray | None = None,
) -> float:
    """Mutate share_prob in-place to reflect intervention costs.  Returns burden."""
    burden = 0.0
    if policy.nudge and content.is_fake:
        reduction = config.NUDGE_STRENGTH * (0.35 + 0.65 * platform_trust)
        if extra_resistance is not None:
            reduction *= 1.0 - 0.5 * extra_resistance
        share_prob *= 1 - reduction
        burden += 0.15 * len(candidates)
    if policy.friction:
        friction = (
            config.FRICTION_STRENGTH if content.is_fake
            else config.FRICTION_TRUE_NEWS_PENALTY
        )
        reduction = friction * (0.45 + 0.55 * impulsivity)
        share_prob *= 1 - reduction
        burden += (0.55 if content.is_fake else 0.22) * len(candidates)
    return burden
