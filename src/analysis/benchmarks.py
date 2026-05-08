"""Benchmark statistics from published misinformation diffusion studies.

Primary source:
  Vosoughi S., Roy D., Aral S. (2018). "The spread of true and false news online."
  Science 359 (6380): 1146-1151.
  https://www.science.org/doi/10.1126/science.aap9559
  Twitter cascade dataset: ~126,000 cascades, 3 million people, 2006-2017.

Why these benchmarks:
  - The paper reports separately for false and true news the cascade depth,
    breadth, structural virality, size, and time-to-depth distributions.
  - It is the most cited empirical study on online misinformation diffusion
    and provides paired (false, true) statistics that line up directly with
    the dual-story design used in this ABM.

Caveats:
  - We rely on ratios (false / true) for cross-scale comparison rather than
    the absolute Twitter scale, because our ABM uses a closed population of
    a few hundred agents.
  - Where Vosoughi's numbers leave gaps (e.g., believability), we
    supplement with Friggeri et al. (2014) and Allcott & Gentzkow (2017).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CascadeStat:
    """A summary statistic with the value, source, and short note."""

    metric: str
    false_value: float
    true_value: float
    ratio_false_over_true: float
    source: str
    note: str


VOSOUGHI_2018: dict[str, CascadeStat] = {
    "max_depth": CascadeStat(
        metric="max_depth",
        false_value=19.0,
        true_value=10.0,
        ratio_false_over_true=1.9,
        source="Vosoughi et al. 2018, Fig. 2A",
        note="Maximum cascade depth observed across all rumor types.",
    ),
    "median_depth_top_decile": CascadeStat(
        metric="median_depth_top_decile",
        false_value=8.0,
        true_value=4.0,
        ratio_false_over_true=2.0,
        source="Vosoughi et al. 2018, main text",
        note="Median depth among top-10% largest cascades; false reach far deeper.",
    ),
    "max_breadth_top_decile": CascadeStat(
        metric="max_breadth_top_decile",
        false_value=1000.0,
        true_value=120.0,
        ratio_false_over_true=8.3,
        source="Vosoughi et al. 2018, Fig. 2B",
        note="Max users at any depth, among top-10% cascades.",
    ),
    "structural_virality_at_size_100": CascadeStat(
        metric="structural_virality_at_size_100",
        false_value=5.0,
        true_value=4.0,
        ratio_false_over_true=1.25,
        source="Vosoughi et al. 2018, Fig. 2D",
        note="Mean structural virality (Goel et al. 2016) at cascade size 100.",
    ),
    "median_size": CascadeStat(
        metric="median_size",
        false_value=4.0,
        true_value=2.0,
        ratio_false_over_true=2.0,
        source="Vosoughi et al. 2018, supplementary",
        note="Median cascade size in number of unique users.",
    ),
    "retweet_odds_ratio": CascadeStat(
        metric="retweet_odds_ratio",
        false_value=1.70,
        true_value=1.00,
        ratio_false_over_true=1.70,
        source="Vosoughi et al. 2018, Table S6",
        note="Odds ratio of being retweeted, false vs true (controlled).",
    ),
    "time_to_reach_1500_users": CascadeStat(
        metric="time_to_reach_1500_users",
        false_value=10.0,
        true_value=60.0,
        ratio_false_over_true=10.0 / 60.0,
        source="Vosoughi et al. 2018, main text",
        note="Hours to reach 1,500 unique users; true takes ~6x longer.",
    ),
}


# Friggeri et al. 2014 ICWSM "Rumor Cascades" — secondary reference for shape.
FRIGGERI_2014_REACH_GROWTH_HALF_LIFE_HOURS = 6.0

# Allcott & Gentzkow 2017 JEP — exposure does not equal belief.
ALLCOTT_GENTZKOW_2017_BELIEF_GIVEN_EXPOSURE = 0.080
ALLCOTT_GENTZKOW_2017_NOTE = (
    "Among US adults exposed to a fake-news headline before the 2016 election, "
    "roughly 8% reported believing it (Allcott & Gentzkow 2017, JEP)."
)


def cumulative_reach_curve(
    n_steps: int,
    final_fraction_false: float = 0.85,
    final_fraction_true: float = 0.55,
    speed_ratio_false_over_true: float = 6.0,
) -> pd.DataFrame:
    """
    Vosoughi-shaped cumulative reach curve (false vs true) on a per-step basis.

    Both stories saturate logistically; the false curve reaches any given share
    `speed_ratio` times faster than the true curve.  This is a *shape target*,
    not a literal reach forecast — useful for RMSE / KS comparisons.
    """
    steps = np.arange(n_steps + 1)
    rate_true = 4.0 / n_steps
    rate_false = rate_true * speed_ratio_false_over_true

    def logistic(t, rate, asymptote):
        midpoint = math.log(99) / rate
        midpoint = max(midpoint, 1.0)
        return asymptote / (1 + np.exp(-rate * (t - midpoint / 4)))

    false_curve = np.maximum.accumulate(logistic(steps, rate_false, final_fraction_false))
    true_curve = np.maximum.accumulate(logistic(steps, rate_true, final_fraction_true))
    return pd.DataFrame(
        {"step": steps, "fake_reach_target": false_curve, "true_reach_target": true_curve}
    )


def benchmark_table() -> pd.DataFrame:
    """Return all Vosoughi 2018 stylized facts as a dataframe."""
    rows = [asdict(stat) for stat in VOSOUGHI_2018.values()]
    return pd.DataFrame(rows)


def expected_intervention_ranking() -> list[str]:
    """
    Heuristic ranking of intervention effectiveness (largest fake-reach
    reduction first), drawn from meta-analytic syntheses:
      - Pennycook & Rand (2021, Trends Cogn. Sci.) — accuracy nudges
      - Guess et al. (2020, PNAS) — media literacy interventions
      - Roozenbeek et al. (2022) — inoculation
      - Twitter / Facebook field reports — downranking + labels
    Downranking and label+downrank consistently yield the largest reductions;
    nudges and friction yield moderate but real effects; labels alone show
    smaller effects on sharing (they reduce belief more than reach).
    """
    return [
        "label_downranking",
        "downranking",
        "nudge_friction",
        "friction",
        "fact_check_label",
        "nudge",
        "none",
    ]
