"""Public entry points: cached LLM perception / persona generators."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from .. import config
from . import client as llm_client
from . import cache as llm_cache
from .mock import mock_for
from .stories import STORY_SPECS


PERCEPTION_COLS = ["importance", "emotional_intensity", "relevance"]
PERSONA_COLS = ["p_believe", "share_propensity", "belief_lability", "resistance"]


def get_hybrid_perception(agents: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(agent, story) LLM perception scores (importance, emotional_intensity,
    relevance), all in [0, 1].  Cached to ``data/hybrid_perception.csv``.
    """
    cached = llm_cache.load_cache(llm_cache.HYBRID_PERCEPTION_PATH)
    needed = llm_cache.missing_pairs(agents, cached)
    if needed:
        new_rows = _generate(agents, needed, mode="hybrid_perception")
        cached = pd.concat([cached, new_rows], ignore_index=True)
        cached.to_csv(llm_cache.HYBRID_PERCEPTION_PATH, index=False)
    return llm_cache.aligned(agents, cached, PERCEPTION_COLS)


def get_llm_persona_decisions(agents: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(agent, story) behavioural profile from the LLM
    (p_believe, share_propensity, belief_lability, resistance).
    Cached to ``data/llm_persona_decisions.csv``.
    """
    cached = llm_cache.load_cache(llm_cache.LLM_PERSONA_DECISIONS_PATH)
    needed = llm_cache.missing_pairs(agents, cached)
    if needed:
        new_rows = _generate(agents, needed, mode="persona_decisions")
        cached = pd.concat([cached, new_rows], ignore_index=True)
        cached.to_csv(llm_cache.LLM_PERSONA_DECISIONS_PATH, index=False)
    return llm_cache.aligned(agents, cached, PERSONA_COLS)


# ── Internal generation loop ────────────────────────────────────────────────


def _generate(
    agents: pd.DataFrame, pairs: list[tuple[int, str]], mode: str
) -> pd.DataFrame:
    """Generate LLM outputs for each (pid, story) pair, falling back to mock."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_real = bool(api_key) and config.PERCEPTION_MODE != "mock"
    strict_llm = os.getenv("STRICT_LLM", "0") == "1"
    client = None
    if use_real:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            print(f"[llm.api] OpenAI unavailable ({exc}); using mock.")
            client = None

    rng = np.random.default_rng(config.RANDOM_SEED + 7)
    rows = []
    pid_col = "cache_pid" if "cache_pid" in agents.columns else "pid"
    agent_lookup = agents.set_index(pid_col)
    for pid, story in pairs:
        agent = agent_lookup.loc[int(pid)]
        spec = STORY_SPECS[story]
        if client is not None:
            try:
                if mode == "hybrid_perception":
                    rec = llm_client.llm_hybrid_perception(client, agent, spec)
                else:
                    rec = llm_client.llm_persona_decisions(client, agent, spec)
            except Exception as exc:  # pragma: no cover
                if strict_llm:
                    raise
                print(f"[llm.api] call failed ({exc}); falling back to mock.")
                rec = mock_for(mode, agent, spec, rng)
        else:
            if strict_llm:
                raise RuntimeError("STRICT_LLM=1 but no OpenAI client is available.")
            rec = mock_for(mode, agent, spec, rng)
        rec.update({"pid": int(pid), "story": story})
        rows.append(rec)
    return pd.DataFrame(rows)
