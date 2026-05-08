"""Story descriptions used in LLM prompts."""

from __future__ import annotations

from .. import config


STORY_SPECS = {
    "fake": {
        "label": "fake",
        "description": (
            "A sensational political story claiming a major figure secretly "
            "embezzled disaster-relief funds.  The post is emotional, lacks "
            "primary sources, and aligns with one ideological side.  No "
            "established news outlet has confirmed it."
        ),
        "ideological_slant": config.FAKE_IDEOLOGICAL_SLANT,
        "is_fake": True,
    },
    "true": {
        "label": "true",
        "description": (
            "A measured corrective story, citing primary documents and a "
            "national newsroom, that contradicts the sensational claim above. "
            "The post is calmer in tone and ideologically more neutral."
        ),
        "ideological_slant": config.TRUE_IDEOLOGICAL_SLANT,
        "is_fake": False,
    },
}
