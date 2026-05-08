"""Intervention-effectiveness and overall alignment summary plots."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .style import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER


def intervention_ranking_compare(
    run_summaries: pd.DataFrame, save: bool = True
) -> str:
    """Bar chart of fake_reach reduction vs 'none' baseline for each variant."""
    fig, ax = plt.subplots(figsize=(11, 5))
    policies = list(run_summaries["policy_key"].unique())

    width = 0.25
    x = np.arange(len(policies))
    none_means = (
        run_summaries[run_summaries["policy_key"] == "none"]
        .groupby("variant")["final_fake_reach"].mean()
    )
    for i, variant in enumerate(VARIANT_ORDER):
        baseline = float(none_means.get(variant, np.nan))
        if not np.isfinite(baseline) or baseline < 1e-9:
            continue
        reductions = []
        for policy in policies:
            sub = run_summaries[
                (run_summaries["variant"] == variant)
                & (run_summaries["policy_key"] == policy)
            ]
            if sub.empty:
                reductions.append(0.0)
                continue
            mean = float(sub["final_fake_reach"].mean())
            reductions.append(max(0.0, (baseline - mean) / baseline))
        ax.bar(
            x + (i - 1) * width, reductions, width,
            color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylabel("Fake-reach reduction vs 'none'")
    ax.set_title(
        "Intervention effectiveness: three variants (right column = literature ranking)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, "compare_intervention_effects.png")
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def alignment_summary_chart(
    curve_alignment: pd.DataFrame,
    structural_alignment: pd.DataFrame,
    ranking_alignment: pd.DataFrame,
    save: bool = True,
) -> str:
    """Three-panel summary: curve RMSE, structural log-gap, ranking Spearman."""
    variants = list(VARIANT_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: average curve RMSE (lower is better)
    rmse = (
        curve_alignment.groupby("variant")["rmse_vs_target"].mean()
        .reindex(variants)
    )
    axes[0].bar(
        [VARIANT_LABELS[v] for v in variants], rmse.values,
        color=[VARIANT_COLORS[v] for v in variants],
    )
    axes[0].set_title("Cumulative-reach RMSE\nvs Vosoughi target (lower better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelrotation=20)

    # Panel 2: mean absolute log-ratio gap on structural metrics (lower better)
    log_gap = (
        structural_alignment.assign(abs_gap=structural_alignment["log_ratio_gap"].abs())
        .groupby("variant")["abs_gap"].mean()
        .reindex(variants)
    )
    axes[1].bar(
        [VARIANT_LABELS[v] for v in variants], log_gap.values,
        color=[VARIANT_COLORS[v] for v in variants],
    )
    axes[1].set_title("Structural-ratio gap\n|log(sim) − log(Vosoughi)| (lower better)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelrotation=20)

    # Panel 3: Spearman with literature ranking (higher is better)
    spear = (
        ranking_alignment.set_index("variant")["spearman_with_literature"]
        .reindex(variants)
    )
    axes[2].bar(
        [VARIANT_LABELS[v] for v in variants], spear.values,
        color=[VARIANT_COLORS[v] for v in variants],
    )
    axes[2].set_ylim(-1, 1)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_title("Intervention ranking\nSpearman vs literature (higher better)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", labelrotation=20)

    fig.suptitle("Alignment with real-world misinformation diffusion benchmarks")
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, "compare_alignment_summary.png")
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
