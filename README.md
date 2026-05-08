# Fake-News Emergence ABM

Misinformation diffusion in a heterogeneous social network, compared across
three behavioural backbones (pure ABM / ABM+LLM / pure LLM persona) and
aligned with published real-world benchmarks.

## Research question

How do misinformation interventions (nudges, friction, fact-check labels,
downranking) change the reach and belief uptake of fake news in a
heterogeneous agent-based social network — and which modelling paradigm
(pure ABM, ABM+LLM hybrid, pure LLM persona) best reproduces the diffusion
patterns observed in real platform data?

## Variants

- `pure_abm`  — rule-based logistic decision (homophilous network +
  hand-coded sharing/believing rule).
- `abm_llm`   — same logistic rule, augmented by per-(agent, story) LLM
  perception scores (importance / emotional intensity / relevance).
- `pure_llm`  — no logistic rule; the LLM returns a per-(agent, story)
  behavioural profile (`p_believe`, `share_propensity`, `belief_lability`,
  `resistance`) and the simulator samples from it.

LLM outputs are cached to `data/hybrid_perception.csv` and
`data/llm_persona_decisions.csv` — re-running the pipeline does not
re-spend tokens. Falls back to a deterministic mock if no API key is set
or `openai` is not installed.

## Run

```bash
python run_comparative_experiment.py --n_agents 200 --n_steps 25 --n_repeats 5
```

Outputs (under `outputs/`):

- `compare_time_series.csv`, `compare_run_summaries.csv`,
  `compare_cascades.csv`
- `compare_curve_alignment.csv` — RMSE / KS / shape-correlation vs target
- `compare_structural_alignment.csv` — fake/true ratios vs Vosoughi 2018
- `compare_ranking_alignment.csv` — Spearman of intervention ranking vs
  literature
- `compare_cumulative_reach.png`, `compare_structural_metrics.png`,
  `compare_intervention_effects.png`, `compare_alignment_summary.png`

## Folder layout

```
fp_abm/
├── run_comparative_experiment.py     # main driver: 3-variant comparison
├── run_counterfactual_stability.py   # IDEAS.md §1: CF identifiability
├── src/
│   ├── config.py                     # all knobs
│   ├── simulation/                   # core ABM mechanics
│   │   ├── content.py                #   Content / Policy dataclasses
│   │   ├── agents.py                 #   agent generator + homophilous network
│   │   └── dynamics.py               #   exposure / belief / share primitives
│   ├── variants/                     # the three diffusion paradigms
│   │   ├── runner.py                 #   run_variant_experiment / run_all_variants
│   │   ├── dynamics.py               #   per-variant belief & share dispatch
│   │   ├── cascades.py               #   cascade tree + structural virality
│   │   └── counterfactual.py         #   CF trait / content shifts
│   ├── llm/                          # cached LLM perception / persona
│   │   ├── api.py                    #   public get_* entry points
│   │   ├── client.py                 #   OpenAI calls + JSON parsing
│   │   ├── cache.py                  #   on-disk cache
│   │   ├── stories.py                #   story descriptions in prompts
│   │   └── mock.py                   #   deterministic fallback
│   ├── analysis/                     # numbers / dataframes only
│   │   ├── benchmarks.py             #   Vosoughi 2018 stylized facts
│   │   ├── alignment.py              #   RMSE / KS / Spearman alignment
│   │   └── counterfactual.py         #   StabilityConfig + run_stability
│   └── plots/                        # all matplotlib output
│       ├── style.py                  #   palette + labels
│       ├── reach.py                  #   cumulative_reach_compare
│       ├── structural.py             #   structural_metric_compare
│       ├── interventions.py          #   intervention + alignment summary
│       └── counterfactual.py         #   CF box + signal/noise bars
├── data/                             # LLM caches live here
├── outputs/                          # comparative experiment outputs
├── archive/                          # earlier single-variant pipeline
├── IDEAS.md                          # 4 sexy points (hybrid claim space)
├── MAJOR_PROMPTS_AND_IDEAS.md        # decisions log
├── README.md
└── requirements.txt
```

The package keeps back-compat aliases in `src/__init__.py` so `from src
import diffusion_variants, comparison_plots, comparative_evaluation,
counterfactual_stability` still works in any existing notebook or
script.

`archive/` contains the earlier single-variant pipeline (feature
selection, perception engine, models, evaluation) and its outputs. The
current pipeline does not import from it.

## Real-world benchmarks

Primary: **Vosoughi, Roy & Aral (2018)** *Science* — cascade
depth/breadth/structural-virality/size + time-to-reach + intervention
ranking. Targets are encoded in `src/analysis/benchmarks.py`.

Supporting references:
- Friggeri et al. (2014) *ICWSM* — rumor cascade growth curves.
- Allcott & Gentzkow (2017) *JEP* — exposure ≠ belief.
- Goel, Anderson, Hofman, Watts (2016) *Management Science* — structural
  virality definition.
- Pennycook & Rand (2021) *Trends in Cognitive Sciences* — belief-vs-share
  decoupling and nudges.

## Research roadmap

See `IDEAS.md` for the 4-point claim space being developed for the
hybrid ABM+LLM paradigm: counterfactual stability, parameter
identifiability, heavy-tail recovery, cross-narrative transfer.
