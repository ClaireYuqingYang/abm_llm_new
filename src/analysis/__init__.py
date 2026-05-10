"""Research-question metrics: benchmarks, alignment, counterfactual stability.

These modules contain *only* numbers + dataframes.  Plotting lives in
``src.plots``.  Drivers (top-level run_*.py) compose the pieces.
"""

from . import benchmarks
from . import alignment
from . import counterfactual
from . import heavytail
from . import stats

__all__ = ["benchmarks", "alignment", "counterfactual", "heavytail", "stats"]
