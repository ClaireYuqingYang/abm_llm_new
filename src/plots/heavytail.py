"""IDEAS.md §3 plots: cascade-size CCDF (log-log) and tail summary bars."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .style import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER


def cascade_ccdf_loglog(
    ccdf: pd.DataFrame, save: bool = True, fname: str = "heavytail_ccdf.png",
) -> str:
    """Three-panel log-log CCDF, one panel per variant; fake vs true overlaid."""
    variants = [v for v in VARIANT_ORDER if v in ccdf["variant"].unique()]
    fig, axes = plt.subplots(1, len(variants), figsize=(5.0 * len(variants), 4.5),
                             sharex=False, sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        sub = ccdf[ccdf["variant"] == v]
        for story, color, marker in [
            ("fake", "#d62728", "o"), ("true", "#1f77b4", "s"),
        ]:
            ss = sub[sub["story"] == story].sort_values("size")
            if ss.empty:
                continue
            ax.loglog(
                ss["size"].values, ss["ccdf"].values,
                color=color, marker=marker, markersize=4, linewidth=1.0,
                alpha=0.85, label=story,
            )
        ax.set_title(VARIANT_LABELS.get(v, v), fontsize=10)
        ax.set_xlabel("cascade size  (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("P(X ≥ size)  (log)")
    fig.suptitle(
        "Cascade-size CCDF (policy = none)\n"
        "Heavy right tail = nearly straight line on log-log; fake-above-true = Vosoughi-style"
    )
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, fname)
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def tail_summary_bars(
    tail: pd.DataFrame,
    ks: pd.DataFrame,
    save: bool = True,
    fname: str = "heavytail_summary.png",
) -> str:
    """3-panel bar chart: max-size, Hill α (top 10%), KS(fake, true)."""
    variants = [v for v in VARIANT_ORDER if v in tail["variant"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    width = 0.4
    x = np.arange(len(variants))

    # Panel 1: max cascade size for fake vs true
    fake_max = tail[tail["story"] == "fake"].set_index("variant").reindex(variants)["size_max"].values
    true_max = tail[tail["story"] == "true"].set_index("variant").reindex(variants)["size_max"].values
    axes[0].bar(x - width/2, fake_max, width, color="#d62728", label="fake")
    axes[0].bar(x + width/2, true_max, width, color="#1f77b4", label="true")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=15)
    axes[0].set_ylabel("max cascade size")
    axes[0].set_title("Tail magnitude\n(higher fake than true → Vosoughi-style)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Hill α on the top 10% (lower α = heavier tail)
    fake_a = tail[tail["story"] == "fake"].set_index("variant").reindex(variants)["hill_alpha_top10"].values
    true_a = tail[tail["story"] == "true"].set_index("variant").reindex(variants)["hill_alpha_top10"].values
    axes[1].bar(x - width/2, fake_a, width, color="#d62728", label="fake")
    axes[1].bar(x + width/2, true_a, width, color="#1f77b4", label="true")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=15)
    axes[1].set_ylabel("Hill α (top 10%)")
    axes[1].set_title("Tail exponent\n(lower = heavier tail)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: KS(fake, true) per variant
    ks_d = ks.set_index("variant").reindex(variants)["ks_distance_fake_vs_true"].values
    bar_colors = [VARIANT_COLORS.get(v, "#888") for v in variants]
    axes[2].bar(x, ks_d, color=bar_colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=15)
    axes[2].set_ylabel("KS distance (fake vs true)")
    axes[2].set_title("Distributional separation\n(higher = more Vosoughi-like)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Heavy-tail recovery in cascade-size distributions  (IDEAS §3)")
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, fname)
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
