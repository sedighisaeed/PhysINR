"""No-reference metrics: UCIQE / UIQM / CCF, and their ordering behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from hydro_bench.metrics import no_reference as nr


class TestUCIQE:
    def test_finite_and_positive(self, hazy_image, clean_image):
        for image in (hazy_image, clean_image):
            value = nr.uciqe(image)
            assert np.isfinite(value)
            assert value > 0.0

    def test_clear_beats_hazy(self, hazy_image, clean_image):
        # A well-contrast, colourful image must score above a hazy one.
        assert nr.uciqe(clean_image) > nr.uciqe(hazy_image)

    def test_constant_image_is_finite(self):
        flat = np.full((32, 32, 3), 100, np.uint8)
        assert np.isfinite(nr.uciqe(flat))


class TestUIQM:
    def test_components_reachable(self, clean_image):
        score, components = nr.uiqm(clean_image, components=True)
        assert set(components) == {"uicm", "uism", "uiconm"}
        assert np.isfinite(score)

    def test_clear_beats_hazy(self, hazy_image, clean_image):
        assert nr.uiqm(clean_image) > nr.uiqm(hazy_image)

    def test_contrast_definitions_both_finite(self, clean_image, hazy_image):
        for image in (clean_image, hazy_image):
            michelson = nr._uiconm(image, contrast="michelson")
            amee = nr._uiconm(image, contrast="amee")
            assert np.isfinite(michelson)
            # Michelson contrast is bounded in [0, 1] by construction.
            assert 0.0 <= michelson <= 1.0
            assert isinstance(amee, float)

    def test_michelson_haze_goes_to_zero(self):
        # A perfectly flat patch has zero contrast under Michelson.
        flat = np.full((32, 32, 3), 120, np.uint8)
        assert nr._uiconm(flat) == pytest.approx(0.0)

    def test_unknown_contrast_raises(self, clean_image):
        with pytest.raises(ValueError, match="unknown contrast definition"):
            nr._uiconm(clean_image, contrast="nope")


class TestCCF:
    def test_grey_image_has_low_cast(self):
        grey = np.full((32, 32, 3), 128, np.uint8)
        assert nr.ccf(grey) <= 0.1

    def test_coloured_image_reports_cast(self, hazy_image):
        # The blue-cast hazy fixture should carry a measurable cast signal.
        assert nr.ccf(hazy_image) > nr.ccf(np.full((32, 32, 3), 128, np.uint8))

    def test_finite(self, clean_image):
        assert np.isfinite(nr.ccf(clean_image))


class TestOptionalMetrics:
    def test_brisque_returns_none_without_model(self, clean_image, monkeypatch, tmp_path):
        # No model file present -> None, never an exception or a fake score.
        monkeypatch.setenv("HYDROBENCH_BRISQUE_MODEL", str(tmp_path / "nope.yml"))
        assert nr.brisque(clean_image) is None

    def test_musiq_returns_none_without_model(self, clean_image):
        assert nr.musiq(clean_image) is None

    def test_unavailable_metrics_never_raise(self, clean_image):
        assert nr.available_nr_metrics()["uciqe"] is True
        # brisque/musiq availability is environment-dependent, not an error.
        for name in ("brisque", "musiq"):
            assert name in nr.available_nr_metrics()


class TestEvaluateNR:
    def test_subset(self, clean_image):
        result = nr.evaluate_nr(clean_image, metrics=("uciqe", "uiqm"))
        assert result.uciqe is not None
        assert result.uiqm is not None
        assert result.ccf is None

    def test_unknown_metric(self, clean_image):
        with pytest.raises(ValueError, match="unknown no-reference metric"):
            nr.evaluate_nr(clean_image, metrics=("bogus",))

    def test_direction_table_complete(self):
        for name in nr.AVAILABLE_NR_METRICS:
            assert name in nr.NR_BETTER_DIRECTION
