"""Heterogeneous agent generator and homophilous small-world network."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

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


FULL_PERSONA_DIR = (
    Path(config.ROOT_DIR)
    / "raw/full_persona/0.0.0/f883165a3026fde855dfd448e0cd16443ab257b6"
)


def _iter_arrow_rows(path: Path):
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            yield from batch.to_pylist()


def _field(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}: ([^\\n]+?)(?= [A-Z][A-Za-z ]+:|$)", text)
    return match.group(1).strip() if match else ""


def _score(text: str, name: str, default: float = 3.0) -> float:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9.]+)", text)
    if not match:
        return default
    return float(match.group(1))


def _education_level(value: str) -> int:
    value = value.lower()
    if "less than" in value or "some high school" in value:
        return 1
    if "high school" in value:
        return 2
    if "some college" in value or "associate" in value:
        return 3
    if "college graduate" in value or "bachelor" in value:
        return 4
    if "postgrad" in value or "graduate degree" in value:
        return 5
    return 3


def _ideology(value: str, affiliation: str) -> float:
    text = f"{value} {affiliation}".lower()
    if "very liberal" in text:
        return -0.85
    if "liberal" in text or "democrat" in text:
        return -0.55
    if "very conservative" in text:
        return 0.85
    if "conservative" in text or "republican" in text:
        return 0.55
    return 0.0


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def generate_digital_twin_agents(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create ABM agents from real Twin-2K persona summaries.

    The raw survey files stay local.  We map survey-backed demographics and
    personality text into the numeric traits required by the diffusion model,
    while preserving the original persona summary for LLM prompts.
    """
    if not FULL_PERSONA_DIR.exists():
        raise FileNotFoundError(
            f"Digital-twin raw personas not found at {FULL_PERSONA_DIR}"
        )

    rows = []
    for path in sorted(FULL_PERSONA_DIR.glob("*.arrow")):
        for row in _iter_arrow_rows(path):
            text = row["persona_summary"]
            source_pid = int(row["pid"])
            education = _education_level(_field(text, "Education level"))
            ideology = _ideology(
                _field(text, "Political views"),
                _field(text, "Political affiliation"),
            )
            openness = _score(text, "score_openness") / 5
            conscientiousness = _score(text, "wave1_score_conscientiousness") / 5
            neuroticism = _score(text, "score_neuroticism") / 5
            agreeableness = _score(text, "score_agreeableness") / 5
            extraversion = _score(text, "score_extraversion") / 5

            media_literacy = _clip01(0.25 + 0.11 * education + 0.18 * openness)
            platform_trust = _clip01(0.55 - 0.18 * abs(ideology) + 0.10 * agreeableness)
            impulsivity = _clip01(0.72 - 0.50 * conscientiousness + 0.18 * neuroticism)
            activity = float(np.clip(0.65 + 0.9 * extraversion, 0.25, 2.5))
            confirmation_bias = _clip01(0.24 + 0.47 * abs(ideology) + 0.12 * neuroticism)
            skepticism = _clip01(0.20 + 0.58 * media_literacy - 0.20 * platform_trust)

            rows.append(
                {
                    "pid": len(rows),
                    # Offset cache keys so LLM outputs cannot collide with
                    # earlier synthetic-agent cache rows.
                    "cache_pid": 100000 + source_pid,
                    "source_pid": source_pid,
                    "ideology": ideology,
                    "education": education,
                    "need_for_cognition": float((openness - 0.5) * 2.5),
                    "media_literacy": media_literacy,
                    "platform_trust": platform_trust,
                    "impulsivity": impulsivity,
                    "activity": activity,
                    "confirmation_bias": confirmation_bias,
                    "skepticism": skepticism,
                    "persona_summary": text,
                }
            )

    agents = pd.DataFrame(rows).sort_values("source_pid").head(n).reset_index(drop=True)
    agents["pid"] = np.arange(len(agents))
    return agents


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
