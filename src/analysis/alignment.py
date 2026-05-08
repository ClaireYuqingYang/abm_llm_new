"""Align the three diffusion variants against published benchmarks.

Three comparison tables:

  cumulative_curve_alignment:
      RMSE / KS / shape-correlation between simulated cumulative reach and
      the Vosoughi-shape target curve, for each variant + story.

  structural_metric_alignment:
      Per-variant fake/true ratio of cascade depth, breadth, virality, and
      size, compared to Vosoughi 2018 ratios.

  intervention_ranking_alignment:
      Per-variant Spearman correlation between policy effectiveness ranking
      and the literature-derived ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmarks as rb


# ── Cumulative reach curve alignment ────────────────────────────────────────


def cumulative_curve_alignment(
    time_series: pd.DataFrame, n_steps: int
) -> pd.DataFrame:
    """One row per (variant, story).  Compares simulated curve (under 'none'
    policy) against the Vosoughi-shape target curve."""
    target = rb.cumulative_reach_curve(n_steps=n_steps)
    rows = []
    sub = time_series[time_series["policy_key"] == "none"]
    for variant, gv in sub.groupby("variant", sort=False):
        mean_curve = (
            gv.groupby("step")[["fake_reach", "true_reach"]].mean().reset_index()
        )
        merged = mean_curve.merge(target, on="step", how="inner")
        for story, sim_col, target_col in [
            ("fake", "fake_reach", "fake_reach_target"),
            ("true", "true_reach", "true_reach_target"),
        ]:
            sim = merged[sim_col].to_numpy()
            tgt = merged[target_col].to_numpy()
            rmse = float(np.sqrt(np.mean((sim - tgt) ** 2)))
            ks = float(np.max(np.abs(sim - tgt)))
            if np.std(sim) > 1e-9 and np.std(tgt) > 1e-9:
                corr = float(np.corrcoef(sim, tgt)[0, 1])
            else:
                corr = float("nan")
            rows.append(
                {
                    "variant": variant,
                    "story": story,
                    "rmse_vs_target": rmse,
                    "ks_distance_vs_target": ks,
                    "shape_correlation": corr,
                    "final_simulated": float(sim[-1]),
                    "final_target": float(tgt[-1]),
                }
            )
    return pd.DataFrame(rows)


# ── Structural metric alignment ─────────────────────────────────────────────


def structural_metric_alignment(run_summaries: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, metric).  Compares fake/true ratio of cascade
    structural metrics to Vosoughi 2018 ratios (under 'none' policy)."""
    metrics = {
        "max_depth": ("fake_max_depth", "true_max_depth", "max_depth"),
        "max_breadth": ("fake_max_breadth", "true_max_breadth", "max_breadth_top_decile"),
        "max_size": ("fake_max_size", "true_max_size", "median_size"),
        "structural_virality": (
            "fake_mean_virality",
            "true_mean_virality",
            "structural_virality_at_size_100",
        ),
    }
    rows = []
    sub = run_summaries[run_summaries["policy_key"] == "none"]
    for variant, gv in sub.groupby("variant", sort=False):
        for metric_name, (fake_col, true_col, bench_key) in metrics.items():
            fake_mean = float(gv[fake_col].mean())
            true_mean = float(gv[true_col].mean())
            ratio = fake_mean / max(true_mean, 1e-9)
            bench = rb.VOSOUGHI_2018[bench_key]
            rows.append(
                {
                    "variant": variant,
                    "metric": metric_name,
                    "fake_value": fake_mean,
                    "true_value": true_mean,
                    "fake_over_true_ratio": ratio,
                    "vosoughi_ratio": bench.ratio_false_over_true,
                    "ratio_gap": ratio - bench.ratio_false_over_true,
                    "log_ratio_gap": float(
                        np.log(max(ratio, 1e-6))
                        - np.log(max(bench.ratio_false_over_true, 1e-6))
                    ),
                    "source": bench.source,
                }
            )
    return pd.DataFrame(rows)


# ── Intervention ranking alignment ──────────────────────────────────────────


def intervention_ranking_alignment(run_summaries: pd.DataFrame) -> pd.DataFrame:
    """Per variant: Spearman correlation between simulated intervention
    effectiveness ranking (lowest fake_reach = best) and literature ranking."""
    expected = rb.expected_intervention_ranking()
    rank_expected = {p: i for i, p in enumerate(expected)}

    rows = []
    for variant, gv in run_summaries.groupby("variant", sort=False):
        agg = (
            gv.groupby("policy_key", sort=False)["final_fake_reach"]
            .mean()
            .reset_index()
            .sort_values("final_fake_reach")
        )
        agg["variant_rank"] = np.arange(len(agg))
        agg["expected_rank"] = agg["policy_key"].map(rank_expected)

        x = agg["variant_rank"].to_numpy(dtype=float)
        y = agg["expected_rank"].to_numpy(dtype=float)
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            spearman = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
        else:
            spearman = float("nan")

        ordering = " > ".join(agg["policy_key"].tolist())
        rows.append(
            {
                "variant": variant,
                "spearman_with_literature": spearman,
                "variant_ordering_best_to_worst": ordering,
                "literature_ordering_best_to_worst": " > ".join(expected),
            }
        )
    return pd.DataFrame(rows)


def _rank(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    return ranks


# ── Convenience wrapper ─────────────────────────────────────────────────────


def evaluate(
    time_series: pd.DataFrame,
    run_summaries: pd.DataFrame,
    n_steps: int,
) -> dict:
    """Return all three alignment tables in one dict."""
    return {
        "curve_alignment": cumulative_curve_alignment(time_series, n_steps),
        "structural_alignment": structural_metric_alignment(run_summaries),
        "ranking_alignment": intervention_ranking_alignment(run_summaries),
    }
