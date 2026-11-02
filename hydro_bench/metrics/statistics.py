"""Statistical significance engine for benchmark comparisons.

A mean-difference table is not evidence. This module produces the quantities a
reviewer actually wants: per-method central tendency and dispersion, bootstrap
confidence intervals, paired Wilcoxon signed-rank tests against a baseline
(and across all method pairs with multiple-comparison control), and effect
sizes.

Everything here is paired by *sample id*: methods are compared on the exact
same images, in the same order. Unpaired comparisons are refused rather than
silently aligned.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


class StatsError(RuntimeError):
    """Raised when a comparison cannot be performed legitimately."""


@dataclass
class SampleStats:
    """Central tendency and dispersion of one method on one metric."""

    method: str
    metric: str
    n: int
    mean: float
    median: float
    std: float
    sem: float
    min: float
    max: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    missing: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def summarize(values: Sequence[float], method: str = "", metric: str = "",
              confidence: float = 0.95, n_resamples: int = 10000,
              rng: Optional[np.random.Generator] = None) -> SampleStats:
    """Central tendency + dispersion + bootstrap CI for one score vector."""
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                     dtype=np.float64)
    n_missing = len(values) - len(arr)
    if arr.size == 0:
        raise StatsError(f"no finite values for metric '{metric}' (method '{method}')")
    if arr.size < 2:
        # A single sample has no dispersion; CI collapses to the point value.
        return SampleStats(method=method, metric=metric, n=int(arr.size),
                           mean=float(arr[0]), median=float(arr[0]), std=0.0,
                           sem=0.0, min=float(arr[0]), max=float(arr[0]),
                           ci_low=float(arr[0]), ci_high=float(arr[0]),
                           missing=n_missing)

    gen = rng or np.random.default_rng(0)
    n = int(arr.size)
    idx = gen.integers(0, n, size=(min(n_resamples, 100000), n))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return SampleStats(
        method=method, metric=metric, n=n,
        mean=float(arr.mean()), median=float(np.median(arr)),
        std=float(arr.std(ddof=1)), sem=float(arr.std(ddof=1) / math.sqrt(n)),
        min=float(arr.min()), max=float(arr.max()),
        ci_low=float(lo), ci_high=float(hi), missing=n_missing,
    )


# ------------------------------------------------------------------ ranking
def rank_methods(values_by_method: Dict[str, Sequence[float]], higher_is_better: bool
                 ) -> Dict[str, int]:
    """Rank methods 1..k by mean score (1 = best). Ties get the same rank."""
    if not values_by_method:
        return {}
    means = {m: float(np.mean([v for v in vs if v is not None]))
             for m, vs in values_by_method.items() if any(v is not None for v in vs)}
    order = sorted(means.items(), key=lambda kv: kv[1],
                   reverse=higher_is_better)
    ranks: Dict[str, int] = {}
    prev_value, prev_rank = None, 0
    for position, (method, value) in enumerate(order, start=1):
        if prev_value is not None and math.isclose(value, prev_value):
            ranks[method] = prev_rank
        else:
            ranks[method] = position
            prev_rank = position
        prev_value = value
    return ranks


# --------------------------------------------------------------- hypothesis
@dataclass
class Comparison:
    """One pairwise significance test on paired samples."""
    metric: str
    method_a: str          # baseline / first method
    method_b: str = ""     # method being compared against the baseline
    n: int = 0
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None     # matched-pairs rank-biserial / Cohen's d
    delta_mean: Optional[float] = None      # mean(b) - mean(a)
    test: str = "wilcoxon"
    alternative: str = "two-sided"
    significant: Optional[bool] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float], method_a: str,
                    method_b: str, metric: str = "", alternative: str = "two-sided",
                    zero_method: str = "wilcox") -> Comparison:
    """Paired Wilcoxon signed-rank test of ``b`` vs. ``a``.

    Nonparametric by design: IQA score distributions are skewed and heavily
    tied, so a paired *t*-test's normality assumption does not hold.
    """
    if len(a) != len(b):
        raise StatsError(
            f"cannot pair {len(a)} samples of '{method_a}' with {len(b)} of "
            f"'{method_b}' on '{metric}'"
        )
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    n = int(arr_a.size)
    comp = Comparison(metric=metric, method_a=method_a, method_b=method_b, n=n,
                      test="wilcoxon", alternative=alternative,
                      delta_mean=float(arr_b.mean() - arr_a.mean()) if n else None)

    if n == 0:
        comp.note = "no paired finite samples"
        return comp
    if n < 5:
        comp.note = f"n={n} is too small for a stable signed-rank test"

    diff = arr_b - arr_a
    if np.allclose(diff, 0.0):
        # scipy raises on all-zero differences; the methods are indistinguishable.
        comp.statistic = 0.0
        comp.p_value = 1.0
        comp.effect_size = 0.0
        comp.significant = False
        comp.note = "all paired differences are zero"
        return comp

    try:
        result = sps.wilcoxon(arr_b, arr_a, alternative=alternative,
                              zero_method=zero_method, method="auto")
        comp.statistic = float(result.statistic)
        comp.p_value = float(result.pvalue)
    except ValueError as exc:  # e.g. degenerate input
        comp.note = f"test failed: {exc}"
        return comp

    # Rank-biserial correlation: a normalised, interpretable effect size for
    # the signed-rank test. r = 1 means b always beats a.
    n_pair = int((diff != 0).sum())
    if n_pair > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = sps.rankdata(np.abs(diff[diff != 0]))
        w_plus = float(ranks[diff[diff != 0] > 0].sum())
        comp.effect_size = (2.0 * w_plus / (n_pair * (n_pair + 1) / 2.0)) - 1.0

    if comp.p_value is not None and math.isfinite(comp.p_value):
        comp.significant = bool(comp.p_value < 0.05)
    return comp


def all_pairs_wilcoxon(values_by_method: Dict[str, Sequence[float]],
                       baseline: Optional[str] = None, metric: str = "",
                       alternative: str = "two-sided",
                       holm_correction: bool = True) -> List[Comparison]:
    """Every pairwise comparison, or all methods vs. one baseline.

    With ``holm_correction`` the returned p-values are adjusted for multiple
    comparisons so a 14-method table does not manufacture false significance.
    """
    names = [m for m, vs in values_by_method.items()
             if any(v is not None and math.isfinite(v) for v in vs)]
    if baseline:
        if baseline not in values_by_method:
            raise StatsError(f"baseline '{baseline}' has no scores on '{metric}'")
        if baseline not in names:
            raise StatsError(f"baseline '{baseline}' has no finite scores "
                             f"on '{metric}'")
    if len(names) < 2:
        return []

    pairs: List[Tuple[str, str]] = []
    if baseline:
        pairs = [(baseline, m) for m in names if m != baseline]
    else:
        pairs = list(itertools.combinations(names, 2))

    comps = [
        paired_wilcoxon(values_by_method[a], values_by_method[b], a, b,
                        metric=metric, alternative=alternative)
        for a, b in pairs
    ]

    if holm_correction and comps:
        pvals = np.array([c.p_value if c.p_value is not None else 1.0
                          for c in comps])
        finite = np.isfinite(pvals)
        adjusted = pvals.copy()
        if finite.any():
            # Holm-Bonferroni: order, multiply by rank, enforce monotonicity.
            order = np.argsort(pvals)
            m = int(finite.sum())
            running_max = 0.0
            for rank, idx in enumerate(order, start=1):
                if not finite[idx]:
                    continue
                adj = pvals[idx] * (m - rank + 1)
                running_max = max(running_max, adj)
                adjusted[idx] = min(running_max, 1.0)
        for comp, adj in zip(comps, adjusted):
            comp.p_value = float(adj)
            comp.note = (comp.note + "; Holm-adjusted" if comp.note
                         else "Holm-adjusted")
            if comp.p_value is not None and math.isfinite(comp.p_value):
                comp.significant = bool(comp.p_value < 0.05)
    return comps


# ------------------------------------------------------------------- tables
@dataclass
class MetricComparisonTable:
    metric: str
    higher_is_better: bool
    stats: List[SampleStats] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    ranks: Dict[str, int] = field(default_factory=dict)

    def method_names(self) -> List[str]:
        return [s.method for s in self.stats]

    def best(self) -> Optional[SampleStats]:
        if not self.stats:
            return None
        key = (lambda s: s.mean) if self.higher_is_better else (lambda s: -s.mean)
        return max(self.stats, key=key)

    def second_best(self) -> Optional[SampleStats]:
        ranked = sorted(self.stats, key=lambda s: (s.mean if self.higher_is_better
                                                   else -s.mean), reverse=True)
        return ranked[1] if len(ranked) > 1 else None

    def p_value_grid(self) -> Dict[Tuple[str, str], Optional[float]]:
        return {(c.method_a, c.method_b): c.p_value for c in self.comparisons}


def build_metric_comparison(values_by_method: Dict[str, Sequence[float]],
                            metric: str, higher_is_better: bool,
                            baseline: Optional[str] = None,
                            confidence: float = 0.95,
                            n_resamples: int = 10000,
                            alternative: str = "two-sided",
                            rng: Optional[np.random.Generator] = None
                            ) -> MetricComparisonTable:
    """Full statistical picture for one metric across all methods."""
    table = MetricComparisonTable(metric=metric, higher_is_better=higher_is_better)
    for method, values in values_by_method.items():
        if all(v is None for v in values):
            continue
        table.stats.append(summarize(values, method=method, metric=metric,
                                     confidence=confidence,
                                     n_resamples=n_resamples, rng=rng))
    table.ranks = rank_methods(values_by_method, higher_is_better)
    table.comparisons = all_pairs_wilcoxon(values_by_method, baseline=baseline,
                                           metric=metric, alternative=alternative)
    return table


def auc_of_delta_curve(deltas: Sequence[float]) -> float:
    """Area under a downstream-improvement curve (trapezoid), used for plots."""
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size < 2:
        return float(arr.sum())
    return float(np.trapezoid(arr, dx=1.0)) if hasattr(np, "trapezoid") \
        else float(np.trapz(arr, dx=1.0))
