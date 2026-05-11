# Digital Twin Story-Validation Prompt

This prompt was used by `scripts/run_digital_twin_validation.py` to validate LLM-based digital twins against real human responses in the Twin-2K `story_beliefs` split. For each participant, the script provided OpenAI `gpt-4.1-nano` with the participant's rich `persona_summary` plus the same story chapter the human participant read, then asked the model to predict the participant's valence, arousal, and interest ratings.

```text
System message:
You output strict JSON only. No prose, no markdown fences.

User message template:
You are simulating a specific survey participant as faithfully as possible.

Participant persona:
{persona_summary}

Story chapter shown to the participant:
{chapter_text}

Predict how this same participant would answer the survey questions after reading the chapter.
Return strict JSON only with:
  valence_1to7: integer 1-7, where 1 means very negative, 4 neutral, 7 very positive
  arousal_1to7: integer 1-7, where 1 means very low energy, 4 medium energy, 7 very high energy
  interest_1to5: integer 1-5, where 1 means not at all interested, 5 very interested
  rationale: one short sentence explaining the prediction
```
