"""Cumulative reach time-series comparison plot."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from ..analysis import benchmarks as rb
from .style import VARIANT_COLORS, VARIANT_LABELS, VARIANT_ORDER


def cumulative_reach_compare(
    time_series: pd.DataFrame, n_steps: int, save: bool = True
) -> str:
    """Two-panel chart: cumulative reach (fake / true) for each variant + target."""
    target = rb.cumulative_reach_curve(n_steps=n_steps)
    none = time_series[time_series["policy_key"] == "none"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, story, ycol, target_col, title in [
        (axes[0], "fake", "fake_reach", "fake_reach_target", "Fake-news cumulative reach"),
        (axes[1], "true", "true_reach", "true_reach_target", "True-news cumulative reach"),
    ]:
        ax.plot(
            target["step"], target[target_col], color="black",
            linestyle="--", linewidth=2.5, label="Vosoughi-shape target",
        )
        for variant in VARIANT_ORDER:
            sub = none[none["variant"] == variant]
            if sub.empty:
                continue
            mean_curve = sub.groupby("step")[ycol].mean()
            std_curve = sub.groupby("step")[ycol].std().fillna(0.0)
            ax.plot(
                mean_curve.index, mean_curve.values,
                color=VARIANT_COLORS[variant], linewidth=2.0,
                label=VARIANT_LABELS[variant],
            )
            ax.fill_between(
                mean_curve.index,
                np.clip(mean_curve.values - std_curve.values, 0, 1),
                np.clip(mean_curve.values + std_curve.values, 0, 1),
                color=VARIANT_COLORS[variant], alpha=0.15,
            )
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Reach (fraction of population)")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Cumulative reach: three variants vs Vosoughi 2018 shape target",
        fontsize=12,
    )
    fig.tight_layout()

    out = os.path.join(config.OUTPUT_DIR, "compare_cumulative_reach.png")
    if save:
        fig.savefig(out, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    return out
