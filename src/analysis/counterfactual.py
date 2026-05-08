"""Phase 1 (IDEAS.md §1): identifiability of counterfactual effects.

Idea
~~~~
Social science cares most about counterfactuals: "if the population's
media literacy were 0.2 higher, how much smaller would fake-news reach
be?"  A modelling paradigm only generates *useful* policy guidance if
its CF estimate is identifiable — i.e. has small variance under re-runs.

  Pure ABM:    deterministic given a seed, smooth response to parameter
               shifts → tight CF distribution.

  ABM+LLM:     same logistic rule, perception scores cached and structurally
               stable → CF distribution as tight as pure ABM.

  Pure LLM persona:  behaviour is a black-box of the persona profile.
               Either the cache is invariant to the trait shift (no CF
               response → biased estimate) or each shift triggers fresh
               persona generation (CF estimate jumps around → wide
               distribution).  Either way, the CF estimator fails the
               social-science requirement.

Output
~~~~~~
DataFrame with one row per (variant, cf, rep) and an aggregate summary
with mean / std / IQR of delta_fake_reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import variants as dv


# ── Default counterfactuals (IDEAS.md §1) ────────────────────────────────────

DEFAULT_CFS: dict[str, dict] = {
    "media_literacy_+0.2":   {"media_literacy_shift": +0.2},
    "media_literacy_-0.2":   {"media_literacy_shift": -0.2},
    "fake_emotionality_+0.2": {"fake_emotionality_shift": +0.2},
    "fake_emotionality_-0.2": {"fake_emotionality_shift": -0.2},
}


@dataclass
class StabilityConfig:
    n_agents: int = 200
    n_steps: int = 25
    n_reps: int = 20
    policy: str = "none"
    variants: tuple[str, ...] = dv.VARIANTS
    cfs: dict[str, dict] = field(default_factory=lambda: dict(DEFAULT_CFS))


# ── Driver ───────────────────────────────────────────────────────────────────


def run_stability(cfg: StabilityConfig | None = None) -> pd.DataFrame:
    """Run baseline + each CF for each variant, paired by repeat seed.

    Returns long-form DataFrame with one row per (variant, cf, rep)."""
    cfg = cfg or StabilityConfig()
    vcfg = dv.VariantConfig(
        n_agents=cfg.n_agents, n_steps=cfg.n_steps, n_repeats=cfg.n_reps,
    )

    # Baselines per variant
    baselines: dict[str, pd.DataFrame] = {}
    for variant in cfg.variants:
        _, summaries, _ = dv.run_variant_experiment(
            variant=variant, policies=[cfg.policy], cfg=vcfg, cf_overrides=None,
        )
        baselines[variant] = summaries.set_index("repeat")

    rows: list[dict] = []
    for variant in cfg.variants:
        for cf_name, cf_overrides in cfg.cfs.items():
            _, summaries, _ = dv.run_variant_experiment(
                variant=variant, policies=[cfg.policy], cfg=vcfg, cf_overrides=cf_overrides,
            )
            for _, row in summaries.iterrows():
                rep = int(row["repeat"])
                base_row = baselines[variant].loc[rep]
                rows.append({
                    "variant": variant,
                    "cf": cf_name,
                    "repeat": rep,
                    "fake_reach_baseline": float(base_row["final_fake_reach"]),
                    "fake_reach_cf": float(row["final_fake_reach"]),
                    "delta_fake_reach": float(row["final_fake_reach"]) - float(base_row["final_fake_reach"]),
                    "true_reach_baseline": float(base_row["final_true_reach"]),
                    "true_reach_cf": float(row["final_true_reach"]),
                    "delta_true_reach": float(row["final_true_reach"]) - float(base_row["final_true_reach"]),
                })
    return pd.DataFrame(rows)


def summarize_stability(long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long-form output into per-(variant, cf) statistics."""
    grp = long.groupby(["variant", "cf"])["delta_fake_reach"]
    summary = grp.agg(
        mean_delta="mean",
        std_delta="std",
        median_delta="median",
        q25_delta=lambda s: s.quantile(0.25),
        q75_delta=lambda s: s.quantile(0.75),
        min_delta="min",
        max_delta="max",
        n_reps="count",
    ).reset_index()
    summary["iqr_delta"] = summary["q75_delta"] - summary["q25_delta"]
    summary["abs_mean_delta"] = summary["mean_delta"].abs()
    safe_abs = summary["abs_mean_delta"].replace(0, np.nan)
    summary["noise_to_signal"] = summary["std_delta"] / safe_abs
    return summary
