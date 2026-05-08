"""Story content + intervention policy definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Content:
    name: str
    is_fake: bool
    emotionality: float
    credibility: float
    ideological_slant: float


@dataclass(frozen=True)
class Policy:
    name: str
    nudge: bool = False
    friction: bool = False
    fact_check_label: bool = False
    downranking: bool = False


POLICIES: dict[str, Policy] = {
    "none": Policy("No intervention"),
    "nudge": Policy("Nudge", nudge=True),
    "friction": Policy("Friction", friction=True),
    "fact_check_label": Policy("Fact-check label", fact_check_label=True),
    "downranking": Policy("Downranking", downranking=True),
    "nudge_friction": Policy("Nudge + friction", nudge=True, friction=True),
    "label_downranking": Policy(
        "Label + downranking", fact_check_label=True, downranking=True
    ),
}
