"""Cached LLM perception / persona generators.

Two artefacts:
  - hybrid_perception.csv        per (agent, story) → importance,
    emotional_intensity, relevance.  Used by ABM+LLM hybrid variant.
  - llm_persona_decisions.csv    per (agent, story) → p_believe,
    share_propensity, belief_lability, resistance.  Used by pure-LLM
    persona variant.

Both files live under ``data/`` and are keyed by (pid, story) so re-runs
read from cache.  Falls back to a deterministic mock when no API key is
configured or the OpenAI client raises.
"""

from .api import get_hybrid_perception, get_llm_persona_decisions
from .stories import STORY_SPECS

__all__ = [
    "get_hybrid_perception",
    "get_llm_persona_decisions",
    "STORY_SPECS",
]
