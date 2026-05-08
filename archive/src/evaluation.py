"""
evaluation.py
─────────────
Evaluation metrics for binary classification.

All functions operate on plain NumPy arrays so they work with any model.
No external ML libraries required.
"""

import numpy as np
import pandas as pd


# ── Core metrics ───────────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC (no scipy required)."""
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order    = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return abs(float(_trapz(tpr, fpr)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probability forecasts (lower = better)."""
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-9) -> float:
    """Binary cross-entropy (lower = better)."""
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1}


# ── Convenience: evaluate one model and return a results dict ──────────────────

def evaluate_model(model, test_df: pd.DataFrame, model_name: str = "") -> dict:
    """
    Run all metrics for a fitted model on test_df.

    Returns a flat dict suitable for building a results table.
    """
    y_true = test_df["behavior"].values.astype(float)
    y_prob = model.predict_proba(test_df)
    y_pred = model.predict(test_df)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model":        model_name or getattr(model, "name", str(type(model).__name__)),
        "accuracy":     accuracy(y_true, y_pred),
        "roc_auc":      roc_auc(y_true, y_prob),
        "brier_score":  brier_score(y_true, y_prob),
        "log_loss":     log_loss(y_true, y_prob),
        "precision":    cm["precision"],
        "recall":       cm["recall"],
        "f1":           cm["f1"],
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    """Pretty-print a list of evaluate_model outputs as a DataFrame."""
    df = pd.DataFrame(results).set_index("model")
    fmt = {c: "{:.3f}".format for c in df.columns}
    return df.style.format(fmt)
