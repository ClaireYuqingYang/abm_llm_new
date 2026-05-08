"""Top-level package.

Layout (purpose-driven submodules):
    config              central knobs
    simulation/         core ABM mechanics (content, agents, dynamics)
    variants/           three paradigm runners + cascade tracking
    llm/                cached LLM perception / persona generators
    analysis/           benchmarks, alignment, counterfactual stability
    plots/              all matplotlib charts

Back-compat aliases below preserve the older flat module names so that
existing driver scripts (run_*.py) and notebooks keep working unchanged.
"""

from . import config

# New, purpose-driven sub-packages
from . import simulation
from . import variants
from . import llm
from . import analysis
from . import plots

# ── Back-compat aliases (old flat names → new sub-packages) ─────────────────
from . import variants as diffusion_variants
from . import plots as comparison_plots
from . import llm as llm_personas
from .analysis import alignment as comparative_evaluation
from .analysis import counterfactual as counterfactual_stability
from .analysis import benchmarks as real_world_benchmarks

__all__ = [
    "config",
    "simulation",
    "variants",
    "llm",
    "analysis",
    "plots",
    # back-compat
    "diffusion_variants",
    "comparison_plots",
    "llm_personas",
    "comparative_evaluation",
    "counterfactual_stability",
    "real_world_benchmarks",
]
