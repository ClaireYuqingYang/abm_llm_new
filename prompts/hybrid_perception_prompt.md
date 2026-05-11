# Hybrid Perception Prompt

This prompt was used to generate the `abm_llm` perception scores cached in `data/hybrid_perception.csv`. For each simulated persona and story, OpenAI `gpt-4.1-nano` returned `importance`, `emotional_intensity`, and `relevance` on a 0 to 1 scale. Those scores were then used as inputs to the ABM sharing and belief-update rules.

```text
System message:
You output strict JSON only. No prose, no markdown fences.

User message template:
You are simulating one social-media user's reaction.

Persona:
{persona_text}

Story ({story_label}):
{story_description}

Rate the following from 0.0 to 1.0 from this user's point of view.
Return strict JSON with these keys:
  importance:          how important this user thinks the story is
  emotional_intensity: how strongly the story makes the user feel
  relevance:           how personally relevant the story feels
Each value MUST be a number in [0, 1]. Output JSON only.
```
