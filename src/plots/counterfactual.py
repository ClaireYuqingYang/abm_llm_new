"""Phase-1 counterfactual stability plots (paired Δ box + signal/noise bars)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .style import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER


def counterfactual_stability_box(
    long: pd.DataFrame, save: bool = True, fname: str = "cf_stability_box.png",
) -> str:
    """Mean paired CF deltas with uncertainty and repeat-level dots.

    The digital-twin run is small and discrete, so a box plot collapses many
    conditions into a flat line at zero.  A mean-dot plot makes the signal and
    the all-zero pure-LLM response easier to read.
    """
    cfs = list(long["cf"].drop_duplicates())
    variants = [v for v in VARIANT_ORDER if v in long["variant"].unique()]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(cfs))
    offsets = np.linspace(-0.24, 0.24, len(variants))
    rng = np.random.default_rng(8)

    for offset, variant in zip(offsets, variants):
        means, errs = [], []
        for cf in cfs:
            vals = long[
                (long["variant"] == variant) & (long["cf"] == cf)
            ]["delta_fake_reach"].to_numpy()
            means.append(float(vals.mean()))
            if len(vals) > 1:
                errs.append(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))))
            else:
                errs.append(0.0)
            jitter = rng.normal(0, 0.012, size=len(vals))
            ax.scatter(
                np.full(len(vals), x[len(means) - 1] + offset) + jitter,
                vals,
                s=18,
                color=VARIANT_COLORS[variant],
                alpha=0.28,
                linewidth=0,
            )
        ax.errorbar(
            x + offset,
            means,
            yerr=errs,
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.5,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )

    ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cfs, rotation=16, ha="right", fontsize=9)
    ax.set_ylabel("Mean Δ fake reach  (counterfactual − baseline)")
    ax.set_title(
        "Counterfactual response with real Twin-2K persona backbone\n"
        "Dots are paired repeats; intervals show approximate 95% CI"
    )
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, fname)
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def counterfactual_stability_summary_bars(
    summary: pd.DataFrame, save: bool = True,
    fname: str = "cf_stability_summary.png",
) -> str:
    """Two side-by-side bar charts: |mean Δ| (signal) and std Δ (noise)."""
    variants = [v for v in VARIANT_ORDER if v in summary["variant"].unique()]
    cfs = list(summary["cf"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.25
    x = np.arange(len(cfs))
    for i, v in enumerate(variants):
        sub = summary[summary["variant"] == v].set_index("cf").reindex(cfs)
        offset = (i - (len(variants) - 1) / 2) * width
        axes[0].bar(
            x + offset, sub["abs_mean_delta"].fillna(0).values,
            width=width, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
        )
        axes[1].bar(
            x + offset, sub["std_delta"].fillna(0).values,
            width=width, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
        )

    for ax, title, ylabel in [
        (axes[0], "|mean Δ fake_reach|  (CF effect size, higher=stronger)",
         "|mean Δ| across reps"),
        (axes[1], "std Δ fake_reach  (CF estimator noise, lower=better)",
         "std Δ across reps"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(cfs, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Counterfactual identifiability: signal vs noise per variant")
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, fname)
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
