"""
feature_selection.py
────────────────────
Train-only feature selection for the ABM prototype.

The goal is not to find a magical feature subset, but to make the model
comparison defensible:
  1. Remove unusable or highly redundant variables on the training split.
  2. Use cross-validated forward selection to keep features that improve AUC.
  3. In the Hybrid model, force perception variables to remain in the rule
     because they are the theoretical contribution being tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .evaluation import roc_auc
from .models import _LogisticRegression


@dataclass
class SelectionResult:
    """Container returned by select_features."""

    selected_features: list[str]
    report: pd.DataFrame


def select_features(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    forced_features: list[str] | None = None,
    max_selected: int | None = None,
    label: str = "model",
) -> SelectionResult:
    """
    Select features using only train_df.

    Args:
        train_df: Training data containing candidate features and behavior.
        candidate_features: Features eligible for selection.
        forced_features: Features always included in the model.
        max_selected: Maximum number of candidate features to select.
        label: Name used in the exported report.
    """
    forced_features = forced_features or []
    max_selected = max_selected or len(candidate_features)

    candidates = _existing_numeric_features(train_df, candidate_features, forced_features)
    candidates, filter_report = _filter_candidates(train_df, candidates, forced_features)

    selected, wrapper_report = _forward_select(
        train_df=train_df,
        candidate_features=candidates,
        forced_features=forced_features,
        max_selected=max_selected,
    )

    selected_features = forced_features + selected
    report = pd.concat([filter_report, wrapper_report], ignore_index=True)
    report.insert(0, "model", label)

    return SelectionResult(selected_features=selected_features, report=report)


def save_feature_selection_report(reports: list[pd.DataFrame]) -> str:
    """Save a combined feature-selection report to outputs/."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "feature_selection_report.csv")
    pd.concat(reports, ignore_index=True).to_csv(path, index=False)
    print(f"  Saved feature-selection report -> {path}")
    return path


def _existing_numeric_features(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    forced_features: list[str],
) -> list[str]:
    missing = [f for f in candidate_features + forced_features if f not in train_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    numeric = []
    skipped = []
    for feature in candidate_features:
        if pd.api.types.is_numeric_dtype(train_df[feature]):
            numeric.append(feature)
        else:
            skipped.append(feature)

    if skipped:
        print(f"  Skipping non-numeric features: {skipped}")
    return numeric


def _filter_candidates(
    train_df: pd.DataFrame,
    candidates: list[str],
    forced_features: list[str],
) -> tuple[list[str], pd.DataFrame]:
    records = []
    kept = []

    y = train_df["behavior"].values.astype(float)
    for feature in candidates:
        values = train_df[feature].values.astype(float)
        variance = float(np.nanvar(values))
        association = abs(_safe_corr(values, y))
        if variance <= 1e-10:
            records.append(_record(feature, "removed_low_variance", np.nan, association))
        else:
            kept.append(feature)
            records.append(_record(feature, "kept_filter", np.nan, association))

    kept = _drop_collinear(train_df, kept, forced_features, records)
    report = pd.DataFrame(records)
    return kept, report


def _drop_collinear(
    train_df: pd.DataFrame,
    candidates: list[str],
    forced_features: list[str],
    records: list[dict],
) -> list[str]:
    if len(candidates) <= 1:
        return candidates

    y = train_df["behavior"].values.astype(float)
    forced = set(forced_features)
    all_features = forced_features + candidates
    corr = train_df[all_features].corr(numeric_only=True).abs()
    association = {
        feature: abs(_safe_corr(train_df[feature].values.astype(float), y))
        for feature in all_features
    }

    dropped = set()
    threshold = config.FEATURE_SELECTION_CORR_THRESHOLD
    ordered = sorted(candidates, key=lambda f: association[f], reverse=True)

    for feature in ordered:
        if feature in dropped:
            continue
        rivals = [
            other for other in candidates
            if other != feature
            and other not in dropped
            and corr.loc[feature, other] >= threshold
        ]
        for other in rivals:
            keep_feature = association[feature] >= association[other]
            loser = other if keep_feature else feature
            winner = feature if keep_feature else other
            dropped.add(loser)
            records.append(_record(
                loser,
                f"removed_collinear_with:{winner}",
                float(corr.loc[winner, loser]),
                association[loser],
            ))
            if loser == feature:
                break

    # If a candidate is strongly redundant with a forced theory feature, drop the
    # candidate.  This keeps the Hybrid coefficient plot centered on perception.
    for feature in list(set(candidates) - dropped):
        for forced_feature in forced:
            if corr.loc[feature, forced_feature] >= threshold:
                dropped.add(feature)
                records.append(_record(
                    feature,
                    f"removed_collinear_with_forced:{forced_feature}",
                    float(corr.loc[feature, forced_feature]),
                    association[feature],
                ))
                break

    return [feature for feature in candidates if feature not in dropped]


def _forward_select(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    forced_features: list[str],
    max_selected: int,
) -> tuple[list[str], pd.DataFrame]:
    selected: list[str] = []
    remaining = list(candidate_features)
    records = []

    baseline_score = _cv_auc(train_df, forced_features) if forced_features else 0.5
    current_score = baseline_score

    records.append({
        "feature": "(forced only)" if forced_features else "(intercept only)",
        "status": "baseline_cv",
        "corr_with_selected": np.nan,
        "abs_behavior_corr": np.nan,
        "cv_auc": current_score,
        "delta_auc": 0.0,
        "step": 0,
    })

    step = 1
    while remaining and len(selected) < max_selected:
        scored = []
        for feature in remaining:
            features = forced_features + selected + [feature]
            score = _cv_auc(train_df, features)
            scored.append((score, feature))

        best_score, best_feature = max(scored, key=lambda item: item[0])
        delta = best_score - current_score
        if delta < config.FEATURE_SELECTION_MIN_DELTA:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        records.append({
            "feature": best_feature,
            "status": "selected_forward",
            "corr_with_selected": _max_abs_corr(train_df, best_feature, forced_features + selected[:-1]),
            "abs_behavior_corr": abs(_safe_corr(
                train_df[best_feature].values.astype(float),
                train_df["behavior"].values.astype(float),
            )),
            "cv_auc": best_score,
            "delta_auc": delta,
            "step": step,
        })
        current_score = best_score
        step += 1

    for feature in remaining:
        records.append({
            "feature": feature,
            "status": "not_selected_no_cv_gain",
            "corr_with_selected": _max_abs_corr(train_df, feature, forced_features + selected),
            "abs_behavior_corr": abs(_safe_corr(
                train_df[feature].values.astype(float),
                train_df["behavior"].values.astype(float),
            )),
            "cv_auc": np.nan,
            "delta_auc": np.nan,
            "step": np.nan,
        })

    return selected, pd.DataFrame(records)


def _cv_auc(train_df: pd.DataFrame, features: list[str]) -> float:
    y = train_df["behavior"].values.astype(float)
    if not features:
        return 0.5

    folds = _stratified_folds(y, config.FEATURE_SELECTION_CV_FOLDS)
    scores = []
    X_all = train_df[features].values.astype(float)

    for val_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)
        model = _LogisticRegression()
        model.fit(X_all[train_idx], y[train_idx])
        y_prob = model.predict_proba(X_all[val_idx])
        scores.append(roc_auc(y[val_idx], y_prob))

    return float(np.mean(scores))


def _stratified_folds(y: np.ndarray, n_folds: int) -> list[np.ndarray]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    folds = [[] for _ in range(n_folds)]

    for value in [0, 1]:
        idx = np.where(y == value)[0]
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            folds[i % n_folds].append(row_idx)

    return [np.array(sorted(fold), dtype=int) for fold in folds if fold]


def _max_abs_corr(train_df: pd.DataFrame, feature: str, selected: list[str]) -> float:
    if not selected:
        return np.nan
    values = train_df[feature].values.astype(float)
    return float(max(abs(_safe_corr(values, train_df[s].values.astype(float))) for s in selected))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) <= 1e-10 or np.nanstd(b) <= 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _record(feature: str, status: str, corr_with_selected: float, abs_behavior_corr: float) -> dict:
    return {
        "feature": feature,
        "status": status,
        "corr_with_selected": corr_with_selected,
        "abs_behavior_corr": abs_behavior_corr,
        "cv_auc": np.nan,
        "delta_auc": np.nan,
        "step": np.nan,
    }
