"""
data_loader.py
──────────────
Loads Twin-2K-500 from HuggingFace OR generates realistic synthetic data.

Real data path (requires `pip install datasets`):
    from data_loader import load_data
    df_train, df_test = load_data()   # USE_REAL_DATA = True in config.py

Synthetic path (default, no API key needed):
    from data_loader import load_data
    df_train, df_test = load_data()   # USE_REAL_DATA = False in config.py
"""

import os
import numpy as np
import pandas as pd

from . import config

# Paths are centralised in config.py
AGENTS_PATH = os.path.join(config.DATA_DIR, "agents.csv")


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(force_regenerate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, test_df) with all structured features + behavior label.

    Data is cached to data/agents.csv so the same agents are reused across
    runs (important when LLM perception costs money — you don't want to
    regenerate agents and re-run perception unnecessarily).

    Args:
        force_regenerate: if True, ignore cache and regenerate from scratch.
    """
    np.random.seed(config.RANDOM_SEED)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not force_regenerate and os.path.exists(AGENTS_PATH):
        print(f"Loading cached agent data from {AGENTS_PATH} …")
        df = pd.read_csv(AGENTS_PATH)
    elif config.USE_REAL_DATA:
        df = _load_real()
        df.to_csv(AGENTS_PATH, index=False)
        print(f"Real data saved to {AGENTS_PATH}")
    else:
        df = _generate_synthetic(config.N_SYNTHETIC)
        df.to_csv(AGENTS_PATH, index=False)
        print(f"Synthetic agents saved to {AGENTS_PATH}")

    return _split(df)


# ── Real data loader ───────────────────────────────────────────────────────────

def _load_real() -> pd.DataFrame:
    """
    Load Twin-2K-500 and reshape into a flat DataFrame.

    Supports two sources:
      - Local parquet files (config.LOCAL_DATA_DIR = "data/raw")
        → run download_data.py once first
      - HuggingFace streaming (config.LOCAL_DATA_DIR = None)
        → requires pip install datasets, downloads on first call

    Column mapping from the raw JSON is handled here so that downstream
    code always sees the standardised feature names defined in config.py.
    """
    if config.LOCAL_DATA_DIR:
        # ── Load from local parquet (fast, no internet needed) ─────────────────
        persona_path = os.path.join(config.LOCAL_DATA_DIR, "persona_json.parquet")
        wave4_path   = os.path.join(config.LOCAL_DATA_DIR, "wave4.parquet")
        if not os.path.exists(persona_path):
            raise FileNotFoundError(
                f"Local data not found at {persona_path}.\n"
                "Run:  python download_data.py"
            )
        print(f"Loading local parquet files from {config.LOCAL_DATA_DIR} …")
        persona_df = pd.read_parquet(persona_path)
        wave4_df   = pd.read_parquet(wave4_path)
    else:
        # ── Stream from HuggingFace ────────────────────────────────────────────
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "Install the HuggingFace `datasets` library to use real data:\n"
                "  pip install datasets"
            )
        print("Streaming from HuggingFace (this may take a minute) …")
        persona_df = load_dataset(
            config.HF_DATASET_NAME, config.HF_PERSONA_CONFIG, split="train"
        ).to_pandas()
        wave4_df = load_dataset(
            config.HF_DATASET_NAME, config.HF_WAVE4_CONFIG, split="train"
        ).to_pandas()

    # ── Parse persona JSON into flat columns ───────────────────────────────────
    # The `survey_json_with_human_response` field holds a dict of measure → score.
    # Adjust key names below to match the actual JSON keys in the dataset.
    def _parse_persona(row):
        j = row["survey_json_with_human_response"]
        if isinstance(j, str):
            import json
            j = json.loads(j)
        return {
            "pid":                row["PID"],
            # Demographics
            "age":                j.get("age", np.nan),
            "gender":             j.get("gender", np.nan),
            "education":          j.get("education", np.nan),
            "income":             j.get("income", np.nan),
            # Big Five
            "openness":           j.get("openness", np.nan),
            "conscientiousness":  j.get("conscientiousness", np.nan),
            "extraversion":       j.get("extraversion", np.nan),
            "agreeableness":      j.get("agreeableness", np.nan),
            "neuroticism":        j.get("neuroticism", np.nan),
            # Extended personality
            "need_for_cognition": j.get("need_for_cognition", np.nan),
            "empathy":            j.get("empathy", np.nan),
            # Cognitive
            "numeracy":           j.get("numeracy", np.nan),
            "crt":                j.get("crt", np.nan),
            "fluid_iq":           j.get("fluid_intelligence", np.nan),
            # Economic preferences
            "risk_preference":    j.get("risk_preference", np.nan),
            "time_preference":    j.get("time_preference", np.nan),
            "altruism":           j.get("altruism", np.nan),
            "trust":              j.get("trust", np.nan),
        }

    flat_persona = pd.DataFrame(persona_df.apply(_parse_persona, axis=1).tolist())

    # ── Extract behavioral outcome from Wave 4 ─────────────────────────────────
    # Using the risky-framing task as the primary outcome (binary 0/1).
    # Adjust the key name to the actual task identifier in the Wave 4 data.
    wave4_outcome = (
        wave4_df[["PID", "risky_choice"]]   # <-- adjust column name as needed
        .rename(columns={"PID": "pid", "risky_choice": "behavior"})
    )

    df = flat_persona.merge(wave4_outcome, on="pid", how="inner")
    df = df.dropna(subset=config.STRUCT_FEATURES + ["behavior"])
    print(f"Loaded {len(df)} participants from real dataset.")
    return df


# ── Synthetic data generator ───────────────────────────────────────────────────

def _generate_synthetic(n: int) -> pd.DataFrame:
    """
    Generate realistic synthetic data that mirrors the Twin-2K-500 variable
    structure.  Each block corresponds to one category of measures in the
    real dataset.
    """
    # ── Demographics ──────────────────────────────────────────────────────────
    age       = np.random.normal(45, 15, n).clip(18, 85).astype(int)
    gender    = np.random.choice([0, 1], n, p=[0.48, 0.52])
    education = np.random.choice([1, 2, 3, 4, 5], n,
                                  p=[0.10, 0.25, 0.30, 0.20, 0.15])
    income    = np.random.choice([1, 2, 3, 4, 5], n,
                                  p=[0.15, 0.25, 0.30, 0.20, 0.10])

    # ── Big Five (1–7 scale) ───────────────────────────────────────────────────
    openness          = np.random.normal(4.5, 1.0, n).clip(1, 7)
    conscientiousness = np.random.normal(4.8, 1.1, n).clip(1, 7)
    extraversion      = np.random.normal(4.0, 1.2, n).clip(1, 7)
    agreeableness     = np.random.normal(4.6, 1.0, n).clip(1, 7)
    neuroticism       = np.random.normal(3.5, 1.2, n).clip(1, 7)

    # ── Extended personality (1–7 scale) ──────────────────────────────────────
    # Need for Cognition: higher = more analytical/reflective
    need_for_cognition = (
        3.5
        + 0.4 * (openness - 4.5)
        + 0.3 * (conscientiousness - 4.8)
        - 0.2 * (neuroticism - 3.5)
        + np.random.normal(0, 0.8, n)
    ).clip(1, 7)

    # Empathy: higher = more empathic
    empathy = (
        4.0
        + 0.3 * (agreeableness - 4.6)
        + 0.2 * gender                   # females slightly higher on average
        - 0.1 * (openness - 4.5)
        + np.random.normal(0, 0.8, n)
    ).clip(1, 7)

    # ── Cognitive (standardised scores) ───────────────────────────────────────
    numeracy  = np.random.normal(0.0, 1.0, n)           # z-scored
    crt       = np.random.randint(0, 4, n)              # 0–3 correct
    fluid_iq  = (                                       # correlated with NFC
        0.5 * need_for_cognition / 7
        + np.random.normal(0, 0.8, n)
    )

    # ── Economic preferences (1–7 scale) ──────────────────────────────────────
    risk_preference = (
        4.0
        - 0.3 * (neuroticism - 3.5)
        + 0.2 * (openness - 4.5)
        + 0.3 * crt / 3
        + np.random.normal(0, 1.0, n)
    ).clip(1, 7)

    time_preference = (                                 # patience
        4.0
        + 0.2 * (conscientiousness - 4.8)
        - 0.1 * (age - 45) / 15
        + np.random.normal(0, 1.0, n)
    ).clip(1, 7)

    altruism = (
        4.0
        + 0.3 * (agreeableness - 4.6)
        + 0.2 * empathy / 7
        + np.random.normal(0, 0.8, n)
    ).clip(1, 7)

    trust = (
        3.8
        + 0.2 * (agreeableness - 4.6)
        - 0.15 * (neuroticism - 3.5)
        + np.random.normal(0, 0.9, n)
    ).clip(1, 7)

    # ── Behavioral outcome: risky-framing task (Wave 4) ───────────────────────
    # Higher risk preference, CRT, openness → more likely to choose risky option.
    # Higher neuroticism, lower income → more risk-averse.
    log_odds = (
        + 0.5 * (risk_preference - 4.0) / 1.0
        + 0.3 * crt / 3
        + 0.2 * (openness - 4.5) / 1.0
        - 0.2 * (neuroticism - 3.5) / 1.2
        + 0.15 * need_for_cognition / 7
        - 0.1 * (age - 45) / 15
        + 0.1 * (income - 3) / 1.5
        - 0.8                                   # intercept (baseline ~30% risky)
        + np.random.normal(0, 0.8, n)
    )
    prob_risky = 1 / (1 + np.exp(-log_odds))
    behavior   = (np.random.rand(n) < prob_risky).astype(int)

    df = pd.DataFrame({
        "pid":                range(n),
        # Demographics
        "age": age, "gender": gender,
        "education": education, "income": income,
        # Big Five
        "openness": openness, "conscientiousness": conscientiousness,
        "extraversion": extraversion, "agreeableness": agreeableness,
        "neuroticism": neuroticism,
        # Extended personality
        "need_for_cognition": need_for_cognition,
        "empathy": empathy,
        # Cognitive
        "numeracy": numeracy, "crt": crt, "fluid_iq": fluid_iq,
        # Economic preferences
        "risk_preference": risk_preference, "time_preference": time_preference,
        "altruism": altruism, "trust": trust,
        # Outcome
        "behavior": behavior,
    })

    print(f"Generated synthetic dataset: {df.shape}, "
          f"risky-choice rate = {behavior.mean():.1%}")
    return df


# ── Train / test split ─────────────────────────────────────────────────────────

def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx     = np.random.permutation(len(df))
    n_train = int((1 - config.TEST_SIZE) * len(df))
    train   = df.iloc[idx[:n_train]].reset_index(drop=True)
    test    = df.iloc[idx[n_train:]].reset_index(drop=True)
    print(f"Split → train: {len(train)}, test: {len(test)}")
    return train, test
