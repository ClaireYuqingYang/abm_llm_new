"""Cascade structural-ratio comparison plot."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .style import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER


def structural_metric_compare(
    structural_alignment: pd.DataFrame, save: bool = True
) -> str:
    """Bar chart of false/true cascade ratios per variant vs Vosoughi values."""
    metrics = ["max_depth", "max_breadth", "max_size", "structural_virality"]
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metrics))
    for i, variant in enumerate(VARIANT_ORDER):
        values = []
        for metric in metrics:
            row = structural_alignment[
                (structural_alignment["variant"] == variant)
                & (structural_alignment["metric"] == metric)
            ]
            values.append(float(row["fake_over_true_ratio"].iloc[0]) if not row.empty else 0)
        ax.bar(
            x + (i - 1.5) * width, values, width,
            color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
            alpha=0.9,
        )

    # Vosoughi reference values
    ref_values = []
    for metric in metrics:
        row = structural_alignment[structural_alignment["metric"] == metric].iloc[0]
        ref_values.append(float(row["vosoughi_ratio"]))
    ax.bar(
        x + 1.5 * width, ref_values, width, color="black",
        label="Vosoughi 2018 (target)", alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Fake / True ratio")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_title(
        "Cascade structural metrics: fake/true ratios vs Vosoughi 2018 targets"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, "compare_structural_metrics.png")
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
