"""Lightweight statistical tests (no SciPy dependency).

Implemented from scratch so the pipeline does not require scipy:

  paired_ttest(x)            — one-sample t on paired-difference vector x
  wilcoxon_signed_rank(x)    — non-parametric paired test (W statistic + p)

For the normal CDF we use math.erf; for the t-distribution we use the
exact relation with the regularised incomplete beta function (Lentz's
algorithm for the continued fraction).  All p-values are two-sided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── Normal CDF via erf ─────────────────────────────────────────────────────


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ── Regularised incomplete beta (Numerical Recipes-style) ──────────────────


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-16) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value for Student's t with given df (exact)."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return _betai(0.5 * df, 0.5, x)


# ── Public test wrappers ───────────────────────────────────────────────────


@dataclass
class TestResult:
    statistic: float
    pvalue: float
    n: int
    note: str = ""


def paired_ttest(diffs: np.ndarray) -> TestResult:
    """Paired t-test on a vector of differences (H0: mean=0)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return TestResult(float("nan"), float("nan"), n, "n<2")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        # Variance collapsed: either Δ is identically 0 (no effect, p=1)
        # or constant non-zero (perfectly identifiable, p=0).
        if mean == 0.0:
            return TestResult(0.0, 1.0, n, "zero variance, mean=0")
        return TestResult(float("inf"), 0.0, n, "zero variance, mean!=0")
    t = mean / (sd / math.sqrt(n))
    p = _t_cdf_two_sided_p(t, n - 1)
    return TestResult(t, p, n)


def wilcoxon_signed_rank(diffs: np.ndarray) -> TestResult:
    """Two-sided Wilcoxon signed-rank test (normal approximation with
    tie correction; pratt-style: zeros excluded)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d) & (d != 0.0)]
    n = len(d)
    if n < 1:
        return TestResult(float("nan"), float("nan"), n, "all zero / empty")
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    sorted_abs = abs_d[order]
    sorted_sign = np.sign(d[order])

    # Average ranks for ties.
    ranks = np.empty(n, dtype=float)
    i = 0
    tie_correction = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[i:j + 1] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_correction += t**3 - t
        i = j + 1

    w_pos = float(np.sum(ranks[sorted_sign > 0]))
    w_neg = float(np.sum(ranks[sorted_sign < 0]))
    W = min(w_pos, w_neg)

    mu = n * (n + 1) / 4.0
    sigma_sq = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if sigma_sq <= 0:
        return TestResult(W, float("nan"), n, "zero variance")
    # Continuity correction.
    z = (W - mu + 0.5) / math.sqrt(sigma_sq) if W < mu else (W - mu - 0.5) / math.sqrt(sigma_sq)
    p = 2.0 * min(_phi(z), 1.0 - _phi(z))
    return TestResult(W, p, n)
