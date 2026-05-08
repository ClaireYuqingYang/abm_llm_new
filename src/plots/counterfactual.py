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
    """Box plot of paired CF deltas (delta_fake_reach) by (variant, cf).

    Narrower and more centred boxes mean the CF is more identifiable
    under that paradigm.
    """
    cfs = list(long["cf"].drop_duplicates())
    variants = [v for v in VARIANT_ORDER if v in long["variant"].unique()]

    fig, axes = plt.subplots(
        1, len(cfs), figsize=(4.0 * len(cfs), 4.5), sharey=True,
    )
    if len(cfs) == 1:
        axes = [axes]

    for ax, cf in zip(axes, cfs):
        data = [
            long[(long["variant"] == v) & (long["cf"] == cf)]["delta_fake_reach"].to_numpy()
            for v in variants
        ]
        bp = ax.boxplot(
            data, tick_labels=[VARIANT_LABELS[v] for v in variants],
            patch_artist=True, widths=0.55, showfliers=True,
        )
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(VARIANT_COLORS[v])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(cf, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Δ fake_reach  (CF − baseline)")
    fig.suptitle(
        "Counterfactual stability: paired Δ fake_reach across repeats\n"
        "(tight box centred away from zero = identifiable CF)"
    )
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
