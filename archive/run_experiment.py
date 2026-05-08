"""
run_experiment.py
-----------------
Main entry point for the fake-news emergence ABM.

Run:
    python run_experiment.py

The experiment compares misinformation interventions:
    none, nudge, friction, fact-check label, downranking, and combinations.

Outputs are written to outputs/:
    fake_news_time_series.csv
    fake_news_policy_runs.csv
    fake_news_policy_comparison.csv
    chart1_misinformation_spread.png
    chart2_intervention_comparison.png
    chart3_intervention_tradeoffs.png
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, fake_news_simulation, visualization


def run_pipeline(
    make_charts: bool = True,
    verbose: bool = True,
    repeats: int | None = None,
) -> dict:
    """Run all misinformation intervention experiments."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    repeats = repeats or config.SIM_N_REPEATS
    if verbose:
        print("=" * 68)
        print(" Fake-News Emergence ABM — Intervention Comparison")
        print("=" * 68)
        print(f"Agents: {config.SIM_N_AGENTS}")
        print(f"Steps per run: {config.SIM_N_STEPS}")
        print(f"Repeats per policy: {repeats}")
        print("Policies: " + ", ".join(config.INTERVENTION_POLICIES))

    time_series, policy_runs = fake_news_simulation.run_all_policies(
        policies=config.INTERVENTION_POLICIES,
        repeats=repeats,
        n_agents=config.SIM_N_AGENTS,
        n_steps=config.SIM_N_STEPS,
    )
    comparison = fake_news_simulation.aggregate_policy_results(policy_runs)

    time_series_path = os.path.join(config.OUTPUT_DIR, "fake_news_time_series.csv")
    runs_path = os.path.join(config.OUTPUT_DIR, "fake_news_policy_runs.csv")
    comparison_path = os.path.join(config.OUTPUT_DIR, "fake_news_policy_comparison.csv")

    time_series.to_csv(time_series_path, index=False)
    policy_runs.to_csv(runs_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    if verbose:
        cols = [
            "policy",
            "final_fake_reach_mean",
            "final_fake_belief_rate_mean",
            "final_true_reach_mean",
            "fake_reach_reduction_vs_none",
            "true_reach_change_vs_none",
            "intervention_burden_mean",
            "efficiency",
        ]
        display = comparison[cols].copy()
        float_cols = display.select_dtypes("float").columns
        print("\nPolicy comparison:")
        print(display.to_string(index=False, formatters={
            col: (lambda x: f"{x:.3f}") for col in float_cols
        }))
        print(f"\nSaved time series -> {time_series_path}")
        print(f"Saved run summaries -> {runs_path}")
        print(f"Saved comparison -> {comparison_path}")

    if make_charts:
        if verbose:
            print("\nGenerating charts...")
        visualization.misinformation_spread(time_series)
        visualization.intervention_comparison(comparison)
        visualization.intervention_tradeoffs(comparison)

    if verbose:
        print("\n" + "=" * 68)
        print(f"Done. Outputs saved to: {config.OUTPUT_DIR}")
        print("=" * 68)

    return {
        "time_series": time_series,
        "policy_runs": policy_runs,
        "comparison": comparison,
    }


def main() -> pd.DataFrame:
    return run_pipeline()["comparison"]


if __name__ == "__main__":
    main()
