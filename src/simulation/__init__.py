"""Core ABM mechanics: content, agents, network, exposure/belief/share primitives.

These pieces are paradigm-agnostic — the three diffusion variants in
``src.variants`` reuse them as building blocks.
"""

from .content import Content, Policy, POLICIES
from .agents import generate_agents, build_social_network
from .dynamics import (
    update_beliefs,
    choose_sharers,
    sigmoid,
    snapshot,
)

__all__ = [
    "Content",
    "Policy",
    "POLICIES",
    "generate_agents",
    "build_social_network",
    "update_beliefs",
    "choose_sharers",
    "sigmoid",
    "snapshot",
]
