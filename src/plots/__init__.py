"""All matplotlib charts.  Plots for benchmark alignment + CF stability.

Each submodule produces one or two related figures.  Importing this
package re-exports every figure-producing function under its short name.
"""

from .reach import cumulative_reach_compare
from .structural import structural_metric_compare
from .interventions import (
    intervention_ranking_compare,
    alignment_summary_chart,
)
from .counterfactual import (
    counterfactual_stability_box,
    counterfactual_stability_summary_bars,
)
from .style import VARIANT_COLORS, VARIANT_LABELS

__all__ = [
    "cumulative_reach_compare",
    "structural_metric_compare",
    "intervention_ranking_compare",
    "alignment_summary_chart",
    "counterfactual_stability_box",
    "counterfactual_stability_summary_bars",
    "VARIANT_COLORS",
    "VARIANT_LABELS",
]
