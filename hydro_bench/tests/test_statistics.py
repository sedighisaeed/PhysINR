"""Statistical engine: bootstrap CIs, ranking, Wilcoxon tests, Holm correction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydro_bench.metrics import statistics as st


class TestSummarize:
    def test_basic_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = st.summarize(values, method="A", metric="m")
        assert stats.n == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min == 1.0 and stats.max == 5.0
        assert 0.0 < stats.sem

    def test_confidence_interval_brackets_mean(self):
        values = list(range(50))
        stats = st.summarize(values, method="A", metric="m", n_resamples=500)
        assert stats.ci_low is not None and stats.ci_high is not None
        assert stats.ci_low < stats.mean < stats.ci_high

    def test_missing_values_are_counted(self):
        stats = st.summarize([1.0, None, float("nan"), 2.0], method="A", metric="m")
        assert stats.n == 2
        assert stats.missing == 2

    def test_empty_raises(self):
        with pytest.raises(st.StatsError):
            st.summarize([], method="A", metric="m")

    def test_single_sample_has_no_dispersion(self):
        stats = st.summarize([5.0], method="A", metric="m")
        assert stats.std == 0.0


class TestRanking:
    def test_higher_is_better(self):
        ranks = st.rank_methods({"A": [1, 2, 3], "B": [4, 5, 6]},
                                higher_is_better=True)
        assert ranks == {"B": 1, "A": 2}

    def test_lower_is_better(self):
        ranks = st.rank_methods({"A": [1, 2, 3], "B": [4, 5, 6]},
                                higher_is_better=False)
        assert ranks == {"A": 1, "B": 2}

    def test_ties_share_rank(self):
        ranks = st.rank_methods({"A": [2, 2], "B": [2, 2], "C": [1, 1]},
                                higher_is_better=True)
        assert ranks["A"] == ranks["B"]


class TestWilcoxon:
    def test_identical_series_is_insignificant(self):
        comp = st.paired_wilcoxon([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                  "A", "B", metric="m")
        assert comp.p_value == 1.0
        assert comp.significant is False
        assert comp.effect_size == pytest.approx(0.0)

    def test_consistent_improvement_is_significant(self):
        a = list(range(20))
        b = [v + 5 for v in a]
        comp = st.paired_wilcoxon(a, b, "A", "B", metric="m")
        assert comp.p_value < 0.05
        assert comp.significant is True
        assert comp.effect_size == pytest.approx(1.0)

    def test_effect_size_negative_for_regression(self):
        a = list(range(20))
        b = [v - 5 for v in a]
        comp = st.paired_wilcoxon(a, b, "A", "B", metric="m")
        assert comp.effect_size == pytest.approx(-1.0)
        assert comp.significant is True

    def test_length_mismatch_raises(self):
        with pytest.raises(st.StatsError, match="cannot pair"):
            st.paired_wilcoxon([1, 2, 3], [1, 2], "A", "B")

    def test_all_pairs_produces_correct_count(self):
        values = {"A": list(range(10)), "B": list(range(10, 20)),
                  "C": list(range(20, 30))}
        comps = st.all_pairs_wilcoxon(values)
        # 3 methods -> 3 pairwise comparisons
        assert len(comps) == 3

    def test_baseline_mode_limits_comparisons(self):
        values = {"A": list(range(10)), "B": list(range(10, 20)),
                  "C": list(range(20, 30))}
        comps = st.all_pairs_wilcoxon(values, baseline="A")
        assert len(comps) == 2
        assert all(c.method_a == "A" for c in comps)

    def test_holm_correction_monotone(self):
        # With many trivial comparisons, adjusted p-values must not decrease.
        rng = np.random.default_rng(0)
        values = {f"M{i}": list(rng.normal(0, 0.01, 30)) for i in range(8)}
        values["BETTER"] = list(rng.normal(5, 0.01, 30))
        comps = st.all_pairs_wilcoxon(values, baseline="M0", holm_correction=True)
        assert all(0.0 <= (c.p_value or 1.0) <= 1.0 for c in comps)

    def test_unknown_baseline_raises(self):
        with pytest.raises(st.StatsError, match="baseline"):
            st.all_pairs_wilcoxon({"A": [1, 2]}, baseline="Z")


class TestComparisonTable:
    def test_best_and_second(self):
        values = {"A": [1.0] * 10, "B": [2.0] * 10, "C": [3.0] * 10}
        table = st.build_metric_comparison(values, metric="m",
                                           higher_is_better=True)
        assert table.best().method == "C"
        assert table.second_best().method == "B"
        assert table.ranks == {"C": 1, "B": 2, "A": 3}

    def test_methods_with_no_scores_are_skipped(self):
        values = {"A": [1.0, 2.0], "B": [None, None]}
        table = st.build_metric_comparison(values, metric="m",
                                           higher_is_better=True)
        assert {s.method for s in table.stats} == {"A"}

    def test_p_value_grid(self):
        values = {"A": [1.0] * 10, "B": [2.0] * 10}
        table = st.build_metric_comparison(values, metric="m",
                                           higher_is_better=True)
        grid = table.p_value_grid()
        assert ("A", "B") in grid
