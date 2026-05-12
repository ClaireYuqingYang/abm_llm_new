"""Top-level orchestration: run all (variant × policy × repeat) combos.

The single ``_simulate`` loop is shared by all three variants — variant-
specific behaviour lives in ``src.variants.dynamics``.  Counterfactual
overrides are applied here too, before the per-run agents/network are
generated, so that the same seed yields a *paired* baseline / CF output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..simulation.agents import (
    generate_agents,
    generate_digital_twin_agents,
    build_social_network,
)
from ..simulation.content import Content, POLICIES, Policy
from ..simulation.dynamics import snapshot
from .. import llm as llm_personas
from .cascades import compute_cascades
from .counterfactual import apply_agent_cf, shifted_emotionality
from .dynamics import (
    choose_sharers_variant,
    send_exposures_track,
    update_beliefs_variant,
)


VARIANTS = ("pure_abm", "abm_llm", "pure_llm")


@dataclass
class VariantConfig:
    n_agents: int = 200
    n_steps: int = 25
    n_repeats: int = 5
    agent_source: str = "synthetic"


def default_variant_config() -> VariantConfig:
    return VariantConfig()


# ── Public driver ───────────────────────────────────────────────────────────


def run_variant_experiment(
    variant: str,
    policies: list[str] | None = None,
    cfg: VariantConfig | None = None,
    cf_overrides: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all policies for a single variant.  Returns three DataFrames:

      time_series:     per (variant, policy, repeat, step) reach/belief.
      run_summaries:   per-run final reach + cascade structural metrics.
      cascade_records: per (variant, policy, repeat, story, seed) cascade.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    cfg = cfg or default_variant_config()
    policies = policies or list(POLICIES.keys())
    cf_overrides = cf_overrides or {}

    series_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    cascade_rows: list[dict] = []

    for policy_key in policies:
        for repeat in range(cfg.n_repeats):
            seed = config.RANDOM_SEED + repeat * 997 + hash(policy_key) % 7919
            rng = np.random.default_rng(int(abs(seed)) % (2**32))
            if cfg.agent_source == "digital_twin":
                agents = generate_digital_twin_agents(cfg.n_agents, rng)
            elif cfg.agent_source == "synthetic":
                agents = generate_agents(cfg.n_agents, rng)
            else:
                raise ValueError(f"Unknown agent source: {cfg.agent_source}")
            agents = apply_agent_cf(agents, cf_overrides)
            neighbors = build_social_network(agents, rng)

            llm_data = _prepare_llm_inputs(variant, agents)

            series, cascades = _simulate(
                variant=variant,
                agents=agents,
                neighbors=neighbors,
                policy=POLICIES[policy_key],
                rng=rng,
                n_steps=cfg.n_steps,
                llm_data=llm_data,
                cf_overrides=cf_overrides,
            )
            series["variant"] = variant
            series["policy_key"] = policy_key
            series["policy"] = POLICIES[policy_key].name
            series["repeat"] = repeat
            series_rows.append(series)

            summary_rows.append(
                _summarize(series, cascades, variant, policy_key, repeat)
            )
            for record in cascades:
                cascade_rows.append(
                    {
                        "variant": variant,
                        "policy_key": policy_key,
                        "policy": POLICIES[policy_key].name,
                        "repeat": repeat,
                        "story": record["story"],
                        "seed": record["seed"],
                        "size": record["size"],
                        "depth": record["depth"],
                        "breadth": record["breadth"],
                        "structural_virality": record["structural_virality"],
                    }
                )

    return (
        pd.concat(series_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(cascade_rows),
    )


def run_all_variants(
    variants: list[str] | None = None,
    policies: list[str] | None = None,
    cfg: VariantConfig | None = None,
    cf_overrides: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variants = variants or list(VARIANTS)
    all_series, all_summaries, all_cascades = [], [], []
    for variant in variants:
        s, r, c = run_variant_experiment(
            variant, policies=policies, cfg=cfg, cf_overrides=cf_overrides,
        )
        all_series.append(s)
        all_summaries.append(r)
        all_cascades.append(c)
    return (
        pd.concat(all_series, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
        pd.concat(all_cascades, ignore_index=True),
    )


# ── LLM input preparation ───────────────────────────────────────────────────


def _prepare_llm_inputs(variant: str, agents: pd.DataFrame) -> dict | None:
    """Pre-fetch LLM perception or persona scores for this variant + agent set."""
    if variant == "pure_abm":
        return None
    if variant == "abm_llm":
        df = llm_personas.get_hybrid_perception(agents)
        return {"perception": _index_by_pid_story(df)}
    if variant == "pure_llm":
        df = llm_personas.get_llm_persona_decisions(agents)
        return {"persona": _index_by_pid_story(df)}
    return None


def _index_by_pid_story(df: pd.DataFrame) -> dict:
    out: dict = {}
    for _, row in df.iterrows():
        out.setdefault(row["story"], {})[int(row["pid"])] = row.to_dict()
    return out


# ── Inner simulation loop with cascade tracking ─────────────────────────────


def _simulate(
    variant: str,
    agents: pd.DataFrame,
    neighbors: list[np.ndarray],
    policy: Policy,
    rng: np.random.Generator,
    n_steps: int,
    llm_data: dict | None,
    cf_overrides: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    n = len(agents)
    fake_emo = shifted_emotionality(config.FAKE_EMOTIONALITY, cf_overrides or {}, "fake")
    true_emo = shifted_emotionality(config.TRUE_EMOTIONALITY, cf_overrides or {}, "true")
    fake = Content(
        "fake", True, fake_emo, config.FAKE_CREDIBILITY, config.FAKE_IDEOLOGICAL_SLANT,
    )
    true = Content(
        "true", False, true_emo, config.TRUE_CREDIBILITY, config.TRUE_IDEOLOGICAL_SLANT,
    )

    fake_belief = np.zeros(n)
    true_belief = np.zeros(n)
    seen_fake = np.zeros(n, dtype=bool)
    seen_true = np.zeros(n, dtype=bool)
    shared_fake = np.zeros(n, dtype=bool)
    shared_true = np.zeros(n, dtype=bool)
    fake_frontier = np.zeros(n, dtype=bool)
    true_frontier = np.zeros(n, dtype=bool)

    parent_fake = np.full(n, -1, dtype=int)
    parent_true = np.full(n, -1, dtype=int)
    seed_fake = np.full(n, -1, dtype=int)  # which seed each node belongs to
    seed_true = np.full(n, -1, dtype=int)
    exposure_count_fake = np.zeros(n, dtype=int)
    exposure_count_true = np.zeros(n, dtype=int)

    fake_seeds = rng.choice(n, size=config.INITIAL_FAKE_SEEDS, replace=False)
    true_seeds = rng.choice(n, size=config.INITIAL_TRUE_SEEDS, replace=False)
    for s in fake_seeds:
        fake_frontier[s] = shared_fake[s] = seen_fake[s] = True
        fake_belief[s] = 0.85
        seed_fake[s] = int(s)
    for s in true_seeds:
        true_frontier[s] = shared_true[s] = seen_true[s] = True
        true_belief[s] = 0.78
        seed_true[s] = int(s)

    rows = []
    cum_fake = int(fake_frontier.sum())
    cum_true = int(true_frontier.sum())
    cum_burden = 0.0

    for step in range(n_steps + 1):
        rows.append(
            snapshot(
                step, seen_fake, seen_true, fake_belief, true_belief,
                fake_frontier, true_frontier, cum_fake, cum_true, cum_burden,
            )
        )
        if step == n_steps:
            break

        fake_exposures, fake_first_exposer = send_exposures_track(
            fake_frontier, fake, policy, neighbors, n, rng
        )
        true_exposures, true_first_exposer = send_exposures_track(
            true_frontier, true, policy, neighbors, n, rng
        )

        fake_receivers = np.flatnonzero(fake_exposures)
        true_receivers = np.flatnonzero(true_exposures)
        exposure_count_fake[fake_receivers] += fake_exposures[fake_receivers]
        exposure_count_true[true_receivers] += true_exposures[true_receivers]

        if len(fake_receivers):
            update_beliefs_variant(
                variant, agents, fake_receivers, fake, policy,
                fake_belief, exposure_count_fake, llm_data, rng,
            )
        if len(true_receivers):
            update_beliefs_variant(
                variant, agents, true_receivers, true, policy,
                true_belief, exposure_count_true, llm_data, rng,
            )

        # Track parent / seed for newly-seen receivers.
        for receiver in fake_receivers:
            if not seen_fake[receiver]:
                parent_fake[receiver] = int(fake_first_exposer[receiver])
                seed_fake[receiver] = seed_fake[parent_fake[receiver]]
                seen_fake[receiver] = True
        for receiver in true_receivers:
            if not seen_true[receiver]:
                parent_true[receiver] = int(true_first_exposer[receiver])
                seed_true[receiver] = seed_true[parent_true[receiver]]
                seen_true[receiver] = True

        fake_candidates = fake_receivers[~shared_fake[fake_receivers]]
        true_candidates = true_receivers[~shared_true[true_receivers]]

        fake_frontier, fake_burden = choose_sharers_variant(
            variant, agents, fake_candidates, fake, policy,
            fake_belief, exposure_count_fake, llm_data, rng,
        )
        true_frontier, true_burden = choose_sharers_variant(
            variant, agents, true_candidates, true, policy,
            true_belief, exposure_count_true, llm_data, rng,
        )

        shared_fake |= fake_frontier
        shared_true |= true_frontier
        cum_fake += int(fake_frontier.sum())
        cum_true += int(true_frontier.sum())
        cum_burden += fake_burden + true_burden

    series = pd.DataFrame(rows)

    cascades: list[dict] = []
    for story, parent, seed_arr, shared in [
        ("fake", parent_fake, seed_fake, shared_fake),
        ("true", parent_true, seed_true, shared_true),
    ]:
        cascades.extend(compute_cascades(story, parent, seed_arr, shared))
    return series, cascades


# ── Per-run summary ─────────────────────────────────────────────────────────


def _summarize(
    series: pd.DataFrame,
    cascades: list[dict],
    variant: str,
    policy_key: str,
    repeat: int,
) -> dict:
    final = series.iloc[-1]
    fake_cascades = [c for c in cascades if c["story"] == "fake"]
    true_cascades = [c for c in cascades if c["story"] == "true"]

    def stat(records, key, agg):
        vals = [c[key] for c in records]
        if not vals:
            return 0.0
        return float(agg(vals))

    return {
        "variant": variant,
        "policy_key": policy_key,
        "policy": POLICIES[policy_key].name,
        "repeat": repeat,
        "final_fake_reach": float(final["fake_reach"]),
        "final_true_reach": float(final["true_reach"]),
        "final_fake_belief_rate": float(final["fake_belief_rate"]),
        "final_true_belief_rate": float(final["true_belief_rate"]),
        "intervention_burden": float(final["intervention_burden"]),
        "fake_cascade_count": len(fake_cascades),
        "true_cascade_count": len(true_cascades),
        "fake_max_depth": stat(fake_cascades, "depth", max),
        "true_max_depth": stat(true_cascades, "depth", max),
        "fake_mean_depth": stat(fake_cascades, "depth", np.mean),
        "true_mean_depth": stat(true_cascades, "depth", np.mean),
        "fake_max_breadth": stat(fake_cascades, "breadth", max),
        "true_max_breadth": stat(true_cascades, "breadth", max),
        "fake_max_size": stat(fake_cascades, "size", max),
        "true_max_size": stat(true_cascades, "size", max),
        "fake_mean_virality": stat(fake_cascades, "structural_virality", np.mean),
        "true_mean_virality": stat(true_cascades, "structural_virality", np.mean),
    }
