"""
perception_engine.py
────────────────────
LLM-based perception engine.

Three modes (controlled by config.PERCEPTION_MODE):

  "mock"      → Deterministic mock using agent traits.
                No API key required.  Good for fast iteration.

  "anthropic" → Real Claude API calls.
                Requires:  pip install anthropic
                           ANTHROPIC_API_KEY in .env

  "openai"    → OpenAI API calls (GPT-4o mini recommended for cost).
                Requires:  pip install openai
                           OPENAI_API_KEY in .env
                ~$0.44 for full 2058-agent dataset with gpt-4o-mini.

In all modes the output is the same shape:
    DataFrame with columns [perceived_importance, emotional_intensity, relevance]
    one row per agent, values on a 1–10 scale.

Prompt design notes
───────────────────
The LLM receives a structured persona description (key trait scores rendered
as natural language) plus a scenario description.  It is asked to rate three
perception dimensions.  These ratings become inputs to the logistic decision
rule — the LLM acts as a *perception engine*, not a decision-maker.
"""

import os
import textwrap
import numpy as np
import pandas as pd


# Load .env automatically if python-dotenv is installed (pip install python-dotenv)
# Falls back silently to system environment variables if not installed.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from . import config

PERCEPTION_PATH = os.path.join(config.DATA_DIR, "perceptions.csv")
OPENAI_USAGE_PATH = os.path.join(config.DATA_DIR, "openai_usage_log.csv")


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_perception(df: pd.DataFrame,
                         force_regenerate: bool = False) -> pd.DataFrame:
    """
    Given a DataFrame of agents, return a DataFrame of perception variables.
    Rows are aligned with input df (same index).

    Results are cached to data/perceptions.csv keyed by pid.
    This is critical for the "anthropic" mode: re-running the pipeline
    won't re-call the API (and won't cost money) if perceptions already exist.

    Args:
        force_regenerate: ignore cache and call the engine again.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Check cache — match by pid, mode, and model so partial results are handled
    # correctly without mixing mock and API-generated perceptions.
    if not force_regenerate and os.path.exists(PERCEPTION_PATH):
        cached = pd.read_csv(PERCEPTION_PATH)
        if {"perception_mode", "perception_model"}.issubset(cached.columns):
            mode_matches = cached["perception_mode"].eq(config.PERCEPTION_MODE)
            model_matches = cached["perception_model"].eq(_current_model_name())
            usable = cached[mode_matches & model_matches]
        else:
            usable = cached.iloc[0:0]
        if set(df["pid"]).issubset(set(usable["pid"])):
            print(f"Loading cached perceptions from {PERCEPTION_PATH} …")
            merged = df[["pid"]].merge(usable, on="pid", how="left")
            return merged[config.PERCEPTION_FEATURES].reset_index(drop=True)

    print(f"Generating perception variables (mode={config.PERCEPTION_MODE}) …")
    if config.PERCEPTION_MODE == "mock":
        result = _mock_perception(df)
    elif config.PERCEPTION_MODE == "anthropic":
        result = _anthropic_perception(df)
    elif config.PERCEPTION_MODE == "openai":
        result = _openai_perception(df)
    else:
        raise ValueError(
            f"Unknown PERCEPTION_MODE: {config.PERCEPTION_MODE!r}\n"
            "Valid options: 'mock', 'anthropic', 'openai'"
        )

    # Save with pid so train/test calls append to the same cache instead of
    # overwriting each other.
    to_save = df[["pid"]].reset_index(drop=True).join(result.reset_index(drop=True))
    to_save["perception_mode"] = config.PERCEPTION_MODE
    to_save["perception_model"] = _current_model_name()
    if os.path.exists(PERCEPTION_PATH):
        cached = pd.read_csv(PERCEPTION_PATH)
        to_save = (
            pd.concat([cached, to_save], ignore_index=True)
            .drop_duplicates(subset=["pid"], keep="last")
            .sort_values("pid")
            .reset_index(drop=True)
        )
    to_save.to_csv(PERCEPTION_PATH, index=False)
    print(f"Perceptions saved to {PERCEPTION_PATH}")
    return result


def _current_model_name() -> str:
    if config.PERCEPTION_MODE == "openai":
        return config.OPENAI_MODEL
    if config.PERCEPTION_MODE == "anthropic":
        return config.ANTHROPIC_MODEL
    return "mock"


# ── Mock perception ────────────────────────────────────────────────────────────

def _mock_perception(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate LLM perception outputs using agent traits.

    The structure here mirrors what a well-prompted LLM would produce:
    perception is heterogeneous across agents because each agent's traits
    shape how they interpret the same scenario.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    noise = lambda: rng.normal(0, config.PERCEPTION_NOISE, len(df))

    # perceived_importance: conscientious, older, higher-income → higher
    perceived_importance = (
        5.0
        + 0.5 * (df["conscientiousness"] - 4.8) / 1.1
        + 0.3 * (df["age"] - 45) / 15
        + 0.2 * (df["income"] - 3) / 1.5
        + 0.2 * df["need_for_cognition"] / 7
        + noise()
    )

    # emotional_intensity: neurotic → high; high CRT → low (more analytical)
    emotional_intensity = (
        5.0
        + 0.6 * (df["neuroticism"] - 3.5) / 1.2
        - 0.4 * df["crt"] / 3
        + 0.2 * df["gender"]
        + 0.2 * df["empathy"] / 7
        + noise()
    )

    # relevance: open, more educated → higher; high income → slightly lower
    relevance = (
        5.0
        + 0.4 * (df["openness"] - 4.5) / 1.0
        + 0.3 * (df["education"] - 3) / 1.5
        - 0.2 * (df["income"] - 3) / 1.5
        + 0.15 * df["need_for_cognition"] / 7
        + noise()
    )

    return pd.DataFrame({
        "perceived_importance": np.clip(perceived_importance, 1, 10),
        "emotional_intensity":  np.clip(emotional_intensity,  1, 10),
        "relevance":            np.clip(relevance,            1, 10),
    }, index=df.index)


# ── Real Anthropic API perception ──────────────────────────────────────────────

def _anthropic_perception(df: pd.DataFrame) -> pd.DataFrame:
    """
    Call Claude to generate perception variables for each agent.

    Cost-saving tips:
      • Use Batch API (50 % discount) — see anthropic docs on message batches.
      • Cache the system prompt (90 % discount on repeated input tokens).
      • Start with claude-haiku-4-5 for prototyping, switch to claude-opus-4-6
        only for final runs.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    import os, json, re

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system_prompt = textwrap.dedent("""
        You are a psychological assessment assistant.
        You will be given a description of a person's traits and a decision scenario.
        Rate the following on a scale of 1–10 for this specific person:

          1. perceived_importance: How important does this person perceive the
             outcome of this decision to be for their life?
          2. emotional_intensity: How strongly does this person feel emotionally
             about this situation (anxiety, excitement, dread, etc.)?
          3. relevance: How personally relevant does this scenario feel to this
             person given their background and values?

        Respond ONLY with valid JSON in this format (no markdown, no explanation):
        {"perceived_importance": <1-10>, "emotional_intensity": <1-10>, "relevance": <1-10>}
    """).strip()

    scenario = textwrap.dedent("""
        Scenario: You are offered a choice between (A) a guaranteed payment of
        $500, or (B) a 50 % chance of winning $1,200 and a 50 % chance of
        winning nothing.  Which do you choose?
    """).strip()

    records = []
    for _, row in df.iterrows():
        persona_text = _build_persona_text(row)
        user_msg = f"{persona_text}\n\n{scenario}"

        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=64,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Robust JSON extraction
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to pull numbers out of messy output
            nums = re.findall(r"\d+(?:\.\d+)?", raw)
            parsed = {
                "perceived_importance": float(nums[0]) if len(nums) > 0 else 5,
                "emotional_intensity":  float(nums[1]) if len(nums) > 1 else 5,
                "relevance":            float(nums[2]) if len(nums) > 2 else 5,
            }

        records.append({
            "perceived_importance": np.clip(parsed.get("perceived_importance", 5), 1, 10),
            "emotional_intensity":  np.clip(parsed.get("emotional_intensity",  5), 1, 10),
            "relevance":            np.clip(parsed.get("relevance",            5), 1, 10),
        })

    return pd.DataFrame(records, index=df.index)


# ── OpenAI API perception ──────────────────────────────────────────────────────

def _openai_perception(df: pd.DataFrame) -> pd.DataFrame:
    """
    Call OpenAI to generate perception variables for each agent.

    Recommended model: gpt-4o-mini  (~$0.44 for full 2058-agent dataset)
    Set in config.py:  OPENAI_MODEL = "gpt-4o-mini"

    Cost-saving tips:
      • Use Batch API (50% discount): set OPENAI_USE_BATCH = True in config.py
        Results are async (up to 24 h turnaround) but half the price.
      • gpt-4o-mini is sufficient for this structured rating task.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install openai")

    import json, re

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    system_prompt = textwrap.dedent("""
        You are a psychological assessment assistant.
        You will be given a description of a person's traits and a decision scenario.
        Rate the following on a scale of 1–10 for this specific person:

          1. perceived_importance: How important does this person perceive the
             outcome of this decision to be for their life?
          2. emotional_intensity: How strongly does this person feel emotionally
             about this situation (anxiety, excitement, dread, etc.)?
          3. relevance: How personally relevant does this scenario feel to this
             person given their background and values?

        Respond ONLY with valid JSON in this format (no markdown, no explanation):
        {"perceived_importance": <1-10>, "emotional_intensity": <1-10>, "relevance": <1-10>}
    """).strip()

    scenario = textwrap.dedent("""
        Scenario: You are offered a choice between (A) a guaranteed payment of
        $500, or (B) a 50% chance of winning $1,200 and a 50% chance of
        winning nothing.  Which do you choose?
    """).strip()

    records = []
    usage_records = []
    for i, (_, row) in enumerate(df.iterrows()):
        persona_text = _build_persona_text(row)
        user_msg     = f"{persona_text}\n\n{scenario}"

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=64,
            temperature=0,              # deterministic output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_records.append({
                "pid": row["pid"],
                "model": config.OPENAI_MODEL,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            })

        # Robust JSON extraction
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            nums = re.findall(r"\d+(?:\.\d+)?", raw)
            parsed = {
                "perceived_importance": float(nums[0]) if len(nums) > 0 else 5,
                "emotional_intensity":  float(nums[1]) if len(nums) > 1 else 5,
                "relevance":            float(nums[2]) if len(nums) > 2 else 5,
            }

        records.append({
            "perceived_importance": np.clip(parsed.get("perceived_importance", 5), 1, 10),
            "emotional_intensity":  np.clip(parsed.get("emotional_intensity",  5), 1, 10),
            "relevance":            np.clip(parsed.get("relevance",            5), 1, 10),
        })

        # Simple progress indicator for large datasets
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(df)} agents processed …")

    _append_openai_usage(usage_records)
    return pd.DataFrame(records, index=df.index)


def _append_openai_usage(records: list[dict]) -> None:
    """Append OpenAI token usage, deduplicating by pid/model."""
    if not records:
        return

    usage_df = pd.DataFrame(records)
    if os.path.exists(OPENAI_USAGE_PATH):
        previous = pd.read_csv(OPENAI_USAGE_PATH)
        usage_df = pd.concat([previous, usage_df], ignore_index=True)

    usage_df = (
        usage_df
        .drop_duplicates(subset=["pid", "model"], keep="last")
        .sort_values(["model", "pid"])
        .reset_index(drop=True)
    )
    usage_df.to_csv(OPENAI_USAGE_PATH, index=False)
    print(f"OpenAI usage saved to {OPENAI_USAGE_PATH}")


def _build_persona_text(row: pd.Series) -> str:
    """
    Render an agent's trait scores as natural language for the LLM prompt.
    Keeping it concise reduces token count and cost.
    """
    def _level(val, low=3.5, high=5.5):
        if val < low:   return "low"
        if val > high:  return "high"
        return "moderate"

    return textwrap.dedent(f"""
        Person profile:
        - Age: {int(row['age'])}, Gender: {'female' if row['gender'] == 1 else 'male'}
        - Education level: {int(row['education'])} / 5, Income level: {int(row['income'])} / 5
        - Personality: Openness={_level(row['openness'])}, Conscientiousness={_level(row['conscientiousness'])},
          Extraversion={_level(row['extraversion'])}, Agreeableness={_level(row['agreeableness'])},
          Neuroticism={_level(row['neuroticism'])}
        - Need for Cognition: {_level(row['need_for_cognition'])}
        - Empathy: {_level(row['empathy'])}
        - Cognitive Reflection Test score: {int(row['crt'])} / 3
        - Risk preference (self-reported): {_level(row['risk_preference'])}
        - Altruism: {_level(row['altruism'])}, Trust in others: {_level(row['trust'])}
    """).strip()
