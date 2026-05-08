"""OpenAI client wrappers + JSON parsing + persona text helper."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import config


def chat_json(client, prompt: str) -> str:
    """Single-shot chat completion that asks for strict JSON.  Retries on error."""
    model = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")
    last_exc = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You output strict JSON only.  No prose, no markdown fences.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content or "{}"
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI call failed after retries: {last_exc}")


def safe_json(text: str) -> dict:
    """Parse JSON tolerantly — strip markdown fences, extract {...} block."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def clip01(x) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.5


def persona_text(agent: pd.Series) -> str:
    """Render an agent row as a short natural-language persona block."""
    def lvl(x):
        x = float(x)
        if x < 0.2:
            return "very low"
        if x < 0.4:
            return "low"
        if x < 0.6:
            return "moderate"
        if x < 0.8:
            return "high"
        return "very high"

    ideology = float(agent["ideology"])
    if ideology < -0.3:
        ide = "progressive-leaning"
    elif ideology > 0.3:
        ide = "conservative-leaning"
    else:
        ide = "centrist"
    return (
        f"- Political leaning: {ide} (ideology={ideology:+.2f}).\n"
        f"- Media literacy: {lvl(agent['media_literacy'])}.\n"
        f"- Trust in platforms / institutions: {lvl(agent['platform_trust'])}.\n"
        f"- Impulsivity: {lvl(agent['impulsivity'])}.\n"
        f"- Confirmation bias: {lvl(agent['confirmation_bias'])}.\n"
        f"- General skepticism toward online claims: {lvl(agent['skepticism'])}.\n"
    )


def llm_hybrid_perception(client, agent: pd.Series, spec: dict) -> dict:
    """Ask the LLM for perception scores (importance / emotion / relevance)."""
    prompt = (
        "You are simulating one social-media user's reaction.\n\n"
        f"Persona:\n{persona_text(agent)}\n\n"
        f"Story ({spec['label']}):\n{spec['description']}\n\n"
        "Rate the following from 0.0 to 1.0 from this user's point of view.\n"
        "Return strict JSON with these keys:\n"
        "  importance:          how important this user thinks the story is\n"
        "  emotional_intensity: how strongly the story makes the user feel\n"
        "  relevance:           how personally relevant the story feels\n"
        "Each value MUST be a number in [0, 1].  Output JSON only."
    )
    text = chat_json(client, prompt)
    data = safe_json(text)
    return {
        "importance": clip01(data.get("importance", 0.5)),
        "emotional_intensity": clip01(data.get("emotional_intensity", 0.5)),
        "relevance": clip01(data.get("relevance", 0.5)),
    }


def llm_persona_decisions(client, agent: pd.Series, spec: dict) -> dict:
    """Ask the LLM for the holistic persona behaviour profile."""
    prompt = (
        "You are role-playing one social-media user.  You will see one story "
        "below.  Output a behavioural profile as strict JSON only.\n\n"
        f"Persona:\n{persona_text(agent)}\n\n"
        f"Story ({spec['label']}):\n{spec['description']}\n\n"
        "Return JSON with these four keys, all in [0, 1]:\n"
        "  p_believe:        chance you would believe the story on first read\n"
        "  share_propensity: chance you would reshare it after one exposure\n"
        "  belief_lability:  how much each repeated exposure shifts your belief\n"
        "  resistance:       resistance to fact-check or corrective signals\n"
        "Output JSON only, no commentary."
    )
    text = chat_json(client, prompt)
    data = safe_json(text)
    return {
        "p_believe": clip01(data.get("p_believe", 0.4)),
        "share_propensity": clip01(data.get("share_propensity", 0.3)),
        "belief_lability": clip01(data.get("belief_lability", 0.3)),
        "resistance": clip01(data.get("resistance", 0.5)),
    }
