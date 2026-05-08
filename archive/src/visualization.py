"""
visualization.py
────────────────
All chart-generation functions for the experiment.

Each function:
  • Saves a PNG to config.OUTPUT_DIR
  • Optionally displays interactively (config.SHOW_PLOTS)
  • Returns the matplotlib Figure for further customisation

Charts produced:
  1. model_comparison     — accuracy & AUC bar chart across three models
  2. perception_vs_behavior — distribution of perception variables by outcome
  3. group_behavior        — predicted vs observed rate across personality groups
  4. coefficients          — hybrid model logistic regression coefficients
  5. misinformation_spread — fake/true reach over time by intervention
  6. intervention_comparison — final outcomes by intervention
  7. intervention_tradeoffs — fake-news reduction vs side effects
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from . import config

# Shared aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = config.CHART_DPI

COLORS = {
    "baseline": "#4C72B0",
    "llm_only": "#DD8452",
    "hybrid":   "#55A868",
}
OUTCOME_COLORS = {0: "#4C72B0", 1: "#DD8452"}


def _savefig(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=config.CHART_DPI)
    print(f"  Saved → {path}")
    if config.SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


# ── Chart 1: Model comparison ──────────────────────────────────────────────────

def model_comparison(results: list[dict]) -> plt.Figure:
    """
    Side-by-side bar charts of Accuracy and ROC-AUC for the three models.

    Args:
        results: list of dicts from evaluation.evaluate_model()
    """
    models   = [r["model"] for r in results]
    accs     = [r["accuracy"] for r in results]
    aucs     = [r["roc_auc"]  for r in results]
    bar_cols = [COLORS["baseline"], COLORS["llm_only"], COLORS["hybrid"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, vals, metric in zip(axes, [accs, aucs], ["Accuracy", "ROC-AUC"]):
        bars = ax.bar(models, vals, color=bar_cols, width=0.5,
                      edgecolor="white", linewidth=1.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
        ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6, label="Random baseline")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Predictive Performance Across Three Modelling Approaches",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig(fig, "chart1_model_comparison.png")
    return fig


# ── Chart 2: Perception variables vs behaviour ────────────────────────────────

def perception_vs_behavior(df: pd.DataFrame) -> plt.Figure:
    """
    Overlapping histograms showing distribution of each perception variable
    for agents who chose safe (0) vs risky (1).

    Args:
        df: full DataFrame including perception variables and behavior column
    """
    labels = {
        "perceived_importance": "Perceived Importance",
        "emotional_intensity":  "Emotional Intensity",
        "relevance":            "Personal Relevance",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, col in zip(axes, config.PERCEPTION_FEATURES):
        for val, label, color in [
            (0, "Safe choice",  OUTCOME_COLORS[0]),
            (1, "Risky choice", OUTCOME_COLORS[1]),
        ]:
            subset = df[df["behavior"] == val][col]
            ax.hist(subset, bins=20, alpha=0.6, color=color,
                    label=label, density=True, edgecolor="white")
            ax.axvline(subset.mean(), color=color, ls="--", lw=1.5, alpha=0.9)

        ax.set_xlabel(labels[col], fontsize=11)
        ax.set_ylabel("Density" if col == "perceived_importance" else "",
                      fontsize=11)
        ax.set_title(labels[col], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("LLM Perception Variables by Behavioural Outcome",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "chart2_perception_vs_behavior.png")
    return fig


# ── Chart 3: Behavioural heterogeneity across groups ─────────────────────────

def group_behavior(df: pd.DataFrame, hybrid_model) -> plt.Figure:
    """
    Compare observed behaviour rate and hybrid model predictions across
    four personality quadrants (High/Low Neuroticism × High/Low NFC).

    Args:
        df:           full DataFrame with all features + behavior + perception
        hybrid_model: fitted HybridModel instance
    """
    # Segment agents
    df = df.copy()
    df["neuro_group"] = pd.cut(df["neuroticism"],       [0, 3.5, 7],
                                labels=["Low N", "High N"])
    df["nfc_group"]   = pd.cut(df["need_for_cognition"], [0, 4.0, 7],
                                labels=["Low NFC", "High NFC"])
    df["group"] = df["neuro_group"].astype(str) + " / " + df["nfc_group"].astype(str)

    df["hybrid_pred_proba"] = hybrid_model.predict_proba(df)

    group_stats = (
        df.groupby("group")
        .agg(observed_rate=("behavior", "mean"),
             hybrid_pred=("hybrid_pred_proba", "mean"),
             n=("pid", "count"))
        .reset_index()
    )

    x, w = np.arange(len(group_stats)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - w / 2, group_stats["observed_rate"], w,
           label="Observed", color=COLORS["baseline"], edgecolor="white")
    ax.bar(x + w / 2, group_stats["hybrid_pred"],   w,
           label="Hybrid prediction", color=COLORS["hybrid"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(group_stats["group"], fontsize=11)
    ax.set_ylabel("Probability of Risky Choice", fontsize=12)
    ax.set_title("Behavioural Heterogeneity Across Personality Groups\n"
                 "(Neuroticism × Need for Cognition)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    for i, row in group_stats.iterrows():
        ax.text(i, 0.03, f"n={row['n']}", ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "chart3_group_behavior.png")
    return fig


# ── Chart 4: Hybrid model coefficients ────────────────────────────────────────

def coefficients(hybrid_model) -> plt.Figure:
    """
    Horizontal bar chart of logistic regression coefficients for the Hybrid
    model, illustrating which features drive the decision rule.
    """
    coef = hybrid_model.coef_df.sort_values("coef")  # ascending for barh
    colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coef["coef"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(coef["feature"], coef["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (standardised input)", fontsize=12)
    ax.set_title("Hybrid Model — Decision Rule Coefficients\n"
                 "(transparency of the explicit logistic rule)",
                 fontsize=13, fontweight="bold")

    patches = [
        mpatches.Patch(color="#DD8452", label="↑ risky choice"),
        mpatches.Patch(color="#4C72B0", label="↓ risky choice"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, "chart4_coefficients.png")
    return fig


# ── Fake-news simulation charts ───────────────────────────────────────────────

def misinformation_spread(time_series: pd.DataFrame) -> plt.Figure:
    """Plot mean fake and true news reach over time for each intervention."""
    mean_ts = (
        time_series.groupby(["policy", "step"], sort=False)
        .agg(fake_reach=("fake_reach", "mean"), true_reach=("true_reach", "mean"))
        .reset_index()
    )

    policies = mean_ts["policy"].drop_duplicates().tolist()
    palette = sns.color_palette("tab10", n_colors=len(policies))
    colors = dict(zip(policies, palette))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for policy in policies:
        subset = mean_ts[mean_ts["policy"] == policy]
        ax.plot(
            subset["step"],
            subset["fake_reach"],
            color=colors[policy],
            lw=2.2,
            label=f"{policy} - fake",
        )
        ax.plot(
            subset["step"],
            subset["true_reach"],
            color=colors[policy],
            lw=1.5,
            ls="--",
            alpha=0.75,
            label=f"{policy} - true",
        )

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel("Population reached", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Information Diffusion Under Misinformation Interventions",
                 fontsize=13, fontweight="bold")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, "chart1_misinformation_spread.png")
    return fig


def intervention_comparison(aggregate: pd.DataFrame) -> plt.Figure:
    """Compare final fake reach, fake belief, and true reach by policy."""
    plot_df = aggregate.copy()
    x = np.arange(len(plot_df))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x - w,
        plot_df["final_fake_reach_mean"],
        w,
        label="Fake reach",
        color="#C44E52",
        edgecolor="white",
    )
    ax.bar(
        x,
        plot_df["final_fake_belief_rate_mean"],
        w,
        label="Fake belief",
        color="#DD8452",
        edgecolor="white",
    )
    ax.bar(
        x + w,
        plot_df["final_true_reach_mean"],
        w,
        label="True reach",
        color="#55A868",
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["policy"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean final population share", fontsize=12)
    ax.set_title("Final Outcomes by Intervention",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, "chart2_intervention_comparison.png")
    return fig


def intervention_tradeoffs(aggregate: pd.DataFrame) -> plt.Figure:
    """Plot intervention effectiveness against true-news change and burden."""
    plot_df = aggregate.copy()
    plot_df = plot_df[plot_df["policy_key"] != "none"].copy()

    fig, ax = plt.subplots(figsize=(8, 5.8))
    sizes = 120 + 500 * (
        plot_df["intervention_burden_mean"]
        / max(plot_df["intervention_burden_mean"].max(), 1e-9)
    )
    scatter = ax.scatter(
        plot_df["fake_reach_reduction_vs_none"],
        plot_df["true_reach_change_vs_none"],
        s=sizes,
        c=plot_df["efficiency"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["policy"],
            (row["fake_reach_reduction_vs_none"], row["true_reach_change_vs_none"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("Fake reach reduction vs no intervention", fontsize=12)
    ax.set_ylabel("True reach change vs no intervention", fontsize=12)
    ax.set_title("Intervention Tradeoff: Effectiveness, Side Effects, Burden",
                 fontsize=13, fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Efficiency", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, "chart3_intervention_tradeoffs.png")
    return fig
