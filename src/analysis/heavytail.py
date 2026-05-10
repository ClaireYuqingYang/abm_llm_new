"""IDEAS.md §3: heavy-tail recovery in cascade size distributions.

Vosoughi 2018 reports that fake news cascades skew far more heavily than
true news cascades — false news has fatter right tails on size, breadth,
and depth.  We ask whether each modelling paradigm reproduces a heavy
right tail at all (closed populations of n agents physically cap each
distribution at n, so we cannot reproduce the n=10^4 tail seen on
Twitter; but we can ask which paradigm comes closest).

For each (variant, story) we compute:
  * the empirical CCDF P(X ≥ x) over cascade sizes
  * a Hill-style upper-tail exponent α via MLE on the top decile
  * a two-sample Kolmogorov–Smirnov distance fake vs true (within a
    variant) — Vosoughi's headline finding is that this distance is
    large (false skews larger).  A faithful model should reproduce a
    sizeable KS distance, not a near-zero one.

Output frames:
  ccdf_long:  one row per (variant, story, x, ccdf)
  tail_summary: one row per (variant, story) with size N, mean, max,
                tail_alpha, p90, p99
  ks_summary: one row per variant with KS(fake, true) and asymptotic p

No scipy dependency: KS p-value uses the Smirnov asymptotic formula
  p = 2 * sum_{k=1..K} (-1)^(k-1) exp(-2 k^2 lam^2),  lam = sqrt(en) * D
where en = n*m/(n+m).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd


# ── CCDF ────────────────────────────────────────────────────────────────────


def empirical_ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ccdf) with ccdf[i] = P(X >= xs[i]) on the empirical sample.

    xs are the sorted unique values; ccdf is non-increasing, ends at 1/n.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return np.array([]), np.array([])
    xs = np.sort(np.unique(v))
    n = len(v)
    ccdf = np.array([(v >= x).sum() / n for x in xs])
    return xs, ccdf


# ── Hill / MLE upper-tail exponent ──────────────────────────────────────────


def hill_alpha(values: np.ndarray, top_frac: float = 0.10) -> float:
    """MLE of α for a Pareto tail X ~ x^{-α}, fit on the top top_frac of data.

    Hill estimator: alpha_hat = 1 / mean(log(x_i / x_min))  for x_i > x_min.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    n = len(v)
    if n < 10:
        return float("nan")
    k = max(5, int(math.ceil(top_frac * n)))
    if k >= n:
        return float("nan")
    sorted_v = np.sort(v)[::-1]
    x_min = sorted_v[k]
    if x_min <= 0:
        return float("nan")
    tail = sorted_v[:k]
    logs = np.log(tail / x_min)
    mean_log = float(logs.mean())
    if mean_log <= 0:
        return float("nan")
    return 1.0 / mean_log + 1.0


# ── Two-sample Kolmogorov–Smirnov (no scipy) ────────────────────────────────


def _smirnov_p(d: float, n: int, m: int) -> float:
    if d <= 0 or n == 0 or m == 0:
        return 1.0
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d  # Stephens correction
    s = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
        s += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, s)))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic D and asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.abs(cdf_a - cdf_b).max())
    p = _smirnov_p(d, len(a), len(b))
    return d, p


# ── Public roll-ups ─────────────────────────────────────────────────────────


@dataclass
class TailRow:
    variant: str
    story: str
    n_cascades: int
    size_mean: float
    size_max: float
    size_p90: float
    size_p99: float
    hill_alpha_top10: float


def tail_summary(cascades: pd.DataFrame, policy_filter: str = "none") -> pd.DataFrame:
    """Per-(variant, story) tail summary.  Filters to a single policy."""
    df = cascades[cascades["policy_key"] == policy_filter] if policy_filter else cascades
    rows: list[dict] = []
    for (variant, story), grp in df.groupby(["variant", "story"]):
        sizes = grp["size"].to_numpy(dtype=float)
        rows.append(asdict(TailRow(
            variant=variant,
            story=story,
            n_cascades=int(len(sizes)),
            size_mean=float(np.mean(sizes)) if len(sizes) else float("nan"),
            size_max=float(np.max(sizes)) if len(sizes) else float("nan"),
            size_p90=float(np.quantile(sizes, 0.90)) if len(sizes) else float("nan"),
            size_p99=float(np.quantile(sizes, 0.99)) if len(sizes) else float("nan"),
            hill_alpha_top10=float(hill_alpha(sizes, top_frac=0.10)),
        )))
    return pd.DataFrame(rows)


def ccdf_long(cascades: pd.DataFrame, policy_filter: str = "none") -> pd.DataFrame:
    df = cascades[cascades["policy_key"] == policy_filter] if policy_filter else cascades
    rows: list[dict] = []
    for (variant, story), grp in df.groupby(["variant", "story"]):
        xs, ccdf = empirical_ccdf(grp["size"].to_numpy(dtype=float))
        for x, p in zip(xs, ccdf):
            rows.append({"variant": variant, "story": story,
                         "size": float(x), "ccdf": float(p)})
    return pd.DataFrame(rows)


def ks_fake_vs_true(cascades: pd.DataFrame, policy_filter: str = "none") -> pd.DataFrame:
    df = cascades[cascades["policy_key"] == policy_filter] if policy_filter else cascades
    rows: list[dict] = []
    for variant, grp in df.groupby("variant"):
        fake = grp[grp["story"] == "fake"]["size"].to_numpy(dtype=float)
        true = grp[grp["story"] == "true"]["size"].to_numpy(dtype=float)
        d, p = ks_two_sample(fake, true)
        rows.append({
            "variant": variant,
            "n_fake": int(len(fake)),
            "n_true": int(len(true)),
            "ks_distance_fake_vs_true": d,
            "ks_pvalue": p,
            "fake_mean": float(np.mean(fake)) if len(fake) else float("nan"),
            "true_mean": float(np.mean(true)) if len(true) else float("nan"),
            "fake_to_true_size_ratio": (float(np.mean(fake)) / float(np.mean(true)))
                if len(fake) and len(true) and np.mean(true) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
