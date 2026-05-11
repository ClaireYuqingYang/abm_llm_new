# Persona Decision Prompt

This prompt was used to generate the `pure_llm` behavior profiles cached in `data/llm_persona_decisions.csv`. For each simulated persona and story, OpenAI `gpt-4.1-nano` returned a holistic behavioral profile. The final op-ed argues that this fully cached behavior profile is less useful for policy counterfactuals than the hybrid perception-layer approach.

```text
System message:
You output strict JSON only. No prose, no markdown fences.

User message template:
You are role-playing one social-media user. You will see one story below.
Output a behavioural profile as strict JSON only.

Persona:
{persona_text}

Story ({story_label}):
{story_description}

Return JSON with these four keys, all in [0, 1]:
  p_believe:        chance you would believe the story on first read
  share_propensity: chance you would reshare it after one exposure
  belief_lability:  how much each repeated exposure shifts your belief
  resistance:       resistance to fact-check or corrective signals
Output JSON only, no commentary.
```
