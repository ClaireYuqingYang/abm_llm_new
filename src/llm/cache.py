"""On-disk cache for per-(agent, story) LLM outputs."""

from __future__ import annotations

import os

import pandas as pd

from .. import config
from .stories import STORY_SPECS


HYBRID_PERCEPTION_PATH = os.path.join(config.DATA_DIR, "hybrid_perception.csv")
LLM_PERSONA_DECISIONS_PATH = os.path.join(config.DATA_DIR, "llm_persona_decisions.csv")


def load_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def missing_pairs(
    agents: pd.DataFrame, cached: pd.DataFrame
) -> list[tuple[int, str]]:
    """Return (pid, story) pairs not yet present in the cache."""
    pid_col = "cache_pid" if "cache_pid" in agents.columns else "pid"
    pairs = [(int(p), s) for p in agents[pid_col].tolist() for s in STORY_SPECS]
    if cached.empty:
        return pairs
    have = set(zip(cached["pid"].astype(int).tolist(), cached["story"].tolist()))
    return [pair for pair in pairs if pair not in have]


def aligned(
    agents: pd.DataFrame, cached: pd.DataFrame, value_cols: list[str]
) -> pd.DataFrame:
    """Return cached rows aligned to the agent dataframe order."""
    out_rows = []
    cached_indexed = cached.set_index(["pid", "story"])
    pid_col = "cache_pid" if "cache_pid" in agents.columns else "pid"
    for _, agent in agents.iterrows():
        pid = int(agent["pid"])
        cache_pid = int(agent[pid_col])
        for story in STORY_SPECS:
            row = cached_indexed.loc[(cache_pid, story)]
            entry = {"pid": pid, "story": story}
            for col in value_cols:
                entry[col] = float(row[col])
            out_rows.append(entry)
    return pd.DataFrame(out_rows)
