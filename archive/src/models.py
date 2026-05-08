"""
models.py
─────────
Three modelling approaches compared in the experiment:

  1. BaselineModel   — logistic regression on structured variables only
  2. LLMOnlyModel    — black-box LLM prediction (no explicit decision rule)
  3. HybridModel     — LLM perception variables + logistic decision rule

All models expose a consistent interface:
    .fit(train_df)  →  self
    .predict(df)    →  np.ndarray of 0/1 labels
    .predict_proba(df) → np.ndarray of float probabilities
    .coef_df        →  pd.DataFrame of feature names + coefficients (where applicable)
"""

import numpy as np
import pandas as pd

from . import config


# ── Core logistic regression (NumPy, no sklearn required) ─────────────────────

class _LogisticRegression:
    """Gradient-descent logistic regression with L2 regularisation."""

    def __init__(self):
        self.w   = None
        self._mu  = None
        self._std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegression":
        X = self._fit_scale(X)
        X = np.c_[np.ones(len(X)), X]          # prepend bias
        self.w = np.zeros(X.shape[1])

        lr, l2 = config.LR_LEARNING_RATE, config.LR_L2_LAMBDA
        for _ in range(config.LR_N_ITERATIONS):
            p    = _sigmoid(X @ self.w)
            grad = X.T @ (p - y) / len(y)
            grad[1:] += l2 * self.w[1:]        # skip bias in regularisation
            self.w -= lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._transform_scale(X)
        X = np.c_[np.ones(len(X)), X]
        return _sigmoid(X @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Scaling helpers ────────────────────────────────────────────────────────
    def _fit_scale(self, X):
        self._mu  = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        return (X - self._mu) / self._std

    def _transform_scale(self, X):
        return (X - self._mu) / self._std


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ── Model 1: Baseline ──────────────────────────────────────────────────────────

class BaselineModel:
    """
    Logistic regression using only structured variables (demographics +
    personality + cognitive + economic preferences).
    No LLM involved.
    """

    name = "Baseline (structured only)"

    def __init__(self, features: list[str] | None = None):
        self._lr = _LogisticRegression()
        self.features = features or config.STRUCT_FEATURES

    def fit(self, train_df: pd.DataFrame) -> "BaselineModel":
        X = train_df[self.features].values.astype(float)
        y = train_df["behavior"].values.astype(float)
        self._lr.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._lr.predict_proba(df[self.features].values.astype(float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._lr.predict(df[self.features].values.astype(float))

    @property
    def coef_df(self) -> pd.DataFrame:
        """Feature coefficients, sorted by absolute magnitude."""
        return _coef_df(self.features, self._lr.w[1:])


# ── Model 2: LLM-Only ─────────────────────────────────────────────────────────

class LLMOnlyModel:
    """
    Simulates a black-box LLM that predicts behavior directly.

    In a real setting this would call the LLM and ask:
        "Given this person's profile, what is the probability they
         choose the risky option?"

    Limitations modelled here:
      • Over-weights salient/surface traits (neuroticism, openness)
      • Under-weights cognitive measures (CRT, numeracy)
      • Higher irreducible noise → lower calibration

    This is intentionally weaker than the Hybrid to illustrate the
    cost of omitting an explicit decision rule.
    """

    name = "LLM-Only"

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(config.RANDOM_SEED + 99)
        log_odds = (
            - 0.25 * df["neuroticism"].values
            + 0.35 * df["openness"].values
            + 0.10 * df["crt"].values             # under-weighted
            + 0.05 * df["education"].values
            + 0.10 * df["risk_preference"].values
            - 0.8
            + rng.normal(0, 1.5, len(df))         # larger noise
        )
        return _sigmoid(log_odds)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= 0.5).astype(int)

    def fit(self, train_df: pd.DataFrame) -> "LLMOnlyModel":
        # LLM-Only has no learnable parameters in this mock
        return self

    @property
    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["(black box — not available)"], "coef": [np.nan]})


# ── Model 3: Hybrid ───────────────────────────────────────────────────────────

class HybridModel:
    """
    Logistic regression on structured variables PLUS LLM perception variables.

    The LLM acts as a *perception engine*: it reads each agent's persona and
    context, then outputs perceived_importance, emotional_intensity, and
    relevance on a 1–10 scale.  These perception variables are concatenated
    with the structured features before fitting the logistic decision rule.

    Advantages over LLM-Only:
      • Explicit, inspectable decision weights (coef_df)
      • LLM's semantic richness complements structured data
      • Perception step is separately auditable
    """

    name = "Hybrid (LLM perception + rule)"

    def __init__(self, features: list[str] | None = None):
        self._lr = _LogisticRegression()
        self.features = features or config.HYBRID_FEATURES

    def fit(self, train_df: pd.DataFrame) -> "HybridModel":
        X = train_df[self.features].values.astype(float)
        y = train_df["behavior"].values.astype(float)
        self._lr.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._lr.predict_proba(df[self.features].values.astype(float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._lr.predict(df[self.features].values.astype(float))

    @property
    def coef_df(self) -> pd.DataFrame:
        return _coef_df(self.features, self._lr.w[1:])


# ── Helper ─────────────────────────────────────────────────────────────────────

def _coef_df(feature_names, coefs) -> pd.DataFrame:
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    return df.reindex(df["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
