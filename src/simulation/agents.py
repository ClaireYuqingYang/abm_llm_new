"""Heterogeneous agent generator and homophilous small-world network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def generate_agents(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate heterogeneous agents with misinformation-relevant traits."""
    ideology = np.clip(rng.normal(0, 0.62, n), -1, 1)
    education = rng.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.24, 0.32, 0.22, 0.12])
    need_for_cognition = np.clip(rng.normal(0, 1, n), -2.5, 2.5)
    media_literacy = np.clip(
        0.50
        + 0.11 * (education - 3)
        + 0.13 * need_for_cognition
        + rng.normal(0, 0.16, n),
        0,
        1,
    )
    platform_trust = np.clip(
        0.50 + rng.normal(0, 0.20, n) - 0.12 * np.abs(ideology), 0, 1
    )
    impulsivity = np.clip(rng.beta(2.2, 3.0, n), 0, 1)
    activity = np.clip(rng.lognormal(mean=-0.35, sigma=0.45, size=n), 0.25, 2.5)
    confirmation_bias = np.clip(
        0.28 + 0.42 * np.abs(ideology) + rng.normal(0, 0.12, n), 0, 1
    )
    skepticism = np.clip(
        0.25 + 0.55 * media_literacy - 0.22 * platform_trust + rng.normal(0, 0.10, n),
        0,
        1,
    )

    return pd.DataFrame(
        {
            "pid": np.arange(n),
            "ideology": ideology,
            "education": education,
            "need_for_cognition": need_for_cognition,
            "media_literacy": media_literacy,
            "platform_trust": platform_trust,
            "impulsivity": impulsivity,
            "activity": activity,
            "confirmation_bias": confirmation_bias,
            "skepticism": skepticism,
        }
    )


def build_social_network(
    agents: pd.DataFrame, rng: np.random.Generator
) -> list[np.ndarray]:
    """
    Build an undirected graph with local ties and some cross-cutting rewires.

    Agents are ordered by ideology, connected to near neighbors, and then receive
    random extra ties.  Candidate random ties are accepted more often when
    ideology is similar, controlled by NETWORK_HOMOPHILY.
    """
    n = len(agents)
    k = max(2, config.NETWORK_MEAN_DEGREE)
    if k % 2:
        k += 1

    order = np.argsort(agents["ideology"].to_numpy())
    adjacency = [set() for _ in range(n)]

    for rank, node in enumerate(order):
        for offset in range(1, k // 2 + 1):
            other = order[(rank + offset) % n]
            adjacency[node].add(other)
            adjacency[other].add(node)

    n_extra = int(n * k * config.NETWORK_REWIRE_PROB / 2)
    ideology = agents["ideology"].to_numpy()
    for _ in range(n_extra):
        source = int(rng.integers(0, n))
        for _attempt in range(30):
            target = int(rng.integers(0, n))
            if source == target or target in adjacency[source]:
                continue
            similarity = 1 - abs(ideology[source] - ideology[target]) / 2
            accept_prob = (1 - config.NETWORK_HOMOPHILY) + (
                config.NETWORK_HOMOPHILY * similarity
            )
            if rng.random() < accept_prob:
                adjacency[source].add(target)
                adjacency[target].add(source)
                break

    return [np.array(sorted(nodes), dtype=int) for nodes in adjacency]
