"""
run_counterfactual_stability.py
-------------------------------
Phase 1 (IDEAS.md §1): Counterfactual stability across the three variants.

For each variant ∈ {pure_abm, abm_llm, pure_llm}, we run the model under
a baseline configuration and under four counterfactual perturbations:
    media_literacy ±0.2, fake_emotionality ±0.2.
Each (variant, CF) pair is repeated N times with paired baseline seeds.

Outputs (under outputs/):
    cf_stability_long.csv      one row per (variant, cf, repeat)
    cf_stability_summary.csv   mean / std / IQR of Δ fake_reach
    cf_stability_box.png       paired-Δ box plot
    cf_stability_summary.png   |mean Δ| vs std Δ bars

Default scale: 100 agents × 20 steps × 10 repeats × 3 variants × 5 conditions
(baseline + 4 CFs).  Use --n_reps 20 once results look right.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, counterfactual_stability as cs, comparison_plots


def main(
    n_agents: int = 100,
    n_steps: int = 20,
    n_reps: int = 10,
    policy: str = "none",
    agent_source: str = "synthetic",
    variants: list[str] | None = None,
    make_plots: bool = True,
) -> dict:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cfg = cs.StabilityConfig(
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        policy=policy,
        agent_source=agent_source,
        variants=tuple(variants) if variants else cs.dv.VARIANTS,
    )

    print("=" * 72)
    print(" Counterfactual stability test (Phase 1)")
    print("=" * 72)
    print(f" Variants: {', '.join(cfg.variants)}")
    print(f" CFs:      {', '.join(cfg.cfs.keys())}")
    print(f" Scale:    {n_agents} agents × {n_steps} steps × {n_reps} reps × {len(cfg.cfs)} CFs")
    print(f" Policy:   {policy}")
    print(f" Agents:   {agent_source}")
    print(f" LLM mode: {config.PERCEPTION_MODE} (model={config.OPENAI_MODEL})")
    print()

    long = cs.run_stability(cfg)
    summary = cs.summarize_stability(long)

    long_path = os.path.join(config.OUTPUT_DIR, "cf_stability_long.csv")
    summary_path = os.path.join(config.OUTPUT_DIR, "cf_stability_summary.csv")
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n--- Summary (mean / std of Δ fake_reach across reps) ---")
    cols = [
        "variant", "cf", "n_reps", "mean_delta", "std_delta",
        "iqr_delta", "noise_to_signal",
    ]
    print(summary[cols].to_string(index=False))

    if make_plots:
        comparison_plots.counterfactual_stability_box(long)
        comparison_plots.counterfactual_stability_summary_bars(summary)

    print("\nDone. Outputs under:", config.OUTPUT_DIR)
    return {"long": long, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--policy", default="none")
    parser.add_argument("--agent_source", choices=["synthetic", "digital_twin"], default="synthetic")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    main(
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        n_reps=args.n_reps,
        policy=args.policy,
        agent_source=args.agent_source,
        variants=args.variants,
        make_plots=not args.no_plots,
    )
