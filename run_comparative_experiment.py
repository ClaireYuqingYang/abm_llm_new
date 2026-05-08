"""
run_comparative_experiment.py
-----------------------------
Compare three diffusion variants (pure ABM, ABM + LLM, pure LLM persona)
against published real-world misinformation benchmarks (Vosoughi et al.
2018 + supporting studies).

Usage:
    python run_comparative_experiment.py

Default scale: 200 agents × 25 steps × 5 repeats × 7 policies × 3 variants.

Outputs (under outputs/):
    compare_time_series.csv
    compare_run_summaries.csv
    compare_cascades.csv
    compare_curve_alignment.csv
    compare_structural_alignment.csv
    compare_ranking_alignment.csv
    compare_cumulative_reach.png
    compare_structural_metrics.png
    compare_intervention_effects.png
    compare_alignment_summary.png
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    config,
    diffusion_variants,
    comparative_evaluation,
    comparison_plots,
)


def main(
    n_agents: int = 200,
    n_steps: int = 25,
    n_repeats: int = 5,
    variants: list[str] | None = None,
    policies: list[str] | None = None,
    make_plots: bool = True,
) -> dict:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cfg = diffusion_variants.VariantConfig(
        n_agents=n_agents, n_steps=n_steps, n_repeats=n_repeats,
    )
    variants = variants or list(diffusion_variants.VARIANTS)
    policies = policies or config.INTERVENTION_POLICIES

    print("=" * 72)
    print(" Comparative misinformation diffusion experiment")
    print("=" * 72)
    print(f" Variants:  {', '.join(variants)}")
    print(f" Policies:  {', '.join(policies)}")
    print(f" Scale:     {n_agents} agents × {n_steps} steps × {n_repeats} repeats")
    print(f" LLM mode:  {config.PERCEPTION_MODE} (model={config.OPENAI_MODEL})")
    print()

    time_series, run_summaries, cascades = diffusion_variants.run_all_variants(
        variants=variants, policies=policies, cfg=cfg,
    )

    # Persist raw outputs
    time_series.to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_time_series.csv"), index=False,
    )
    run_summaries.to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_run_summaries.csv"), index=False,
    )
    cascades.to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_cascades.csv"), index=False,
    )

    # Alignment metrics
    evals = comparative_evaluation.evaluate(time_series, run_summaries, n_steps=n_steps)
    evals["curve_alignment"].to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_curve_alignment.csv"), index=False,
    )
    evals["structural_alignment"].to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_structural_alignment.csv"), index=False,
    )
    evals["ranking_alignment"].to_csv(
        os.path.join(config.OUTPUT_DIR, "compare_ranking_alignment.csv"), index=False,
    )

    print("\n--- Curve alignment (RMSE vs Vosoughi-shape target) ---")
    print(evals["curve_alignment"].to_string(index=False))

    print("\n--- Structural metric alignment (fake/true ratios vs Vosoughi 2018) ---")
    print(evals["structural_alignment"].to_string(index=False))

    print("\n--- Intervention ranking alignment (Spearman vs literature) ---")
    print(evals["ranking_alignment"].to_string(index=False))

    if make_plots:
        print("\nGenerating comparison plots...")
        comparison_plots.cumulative_reach_compare(time_series, n_steps=n_steps)
        comparison_plots.structural_metric_compare(evals["structural_alignment"])
        comparison_plots.intervention_ranking_compare(run_summaries)
        comparison_plots.alignment_summary_chart(
            evals["curve_alignment"],
            evals["structural_alignment"],
            evals["ranking_alignment"],
        )

    print("\nDone. All outputs under:", config.OUTPUT_DIR)
    return {
        "time_series": time_series,
        "run_summaries": run_summaries,
        "cascades": cascades,
        **evals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    main(
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        n_repeats=args.n_repeats,
        variants=args.variants,
        policies=args.policies,
        make_plots=not args.no_plots,
    )
