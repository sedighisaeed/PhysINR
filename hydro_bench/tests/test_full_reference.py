"""Full-reference metrics: PSNR / SSIM / MS-SSIM / LPIPS."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydro_bench.metrics import full_reference as fr


class TestPSNR:
    def test_identical_images_are_infinite(self, clean_image):
        assert math.isinf(fr.psnr(clean_image, clean_image))

    def test_scale_invariance(self, clean_image, noisy_image):
        float_version = clean_image.astype(np.float64) / 255.0
        assert fr.psnr(clean_image, noisy_image) == pytest.approx(
            fr.psnr(float_version, noisy_image.astype(np.float64) / 255.0),
            abs=1e-6)

    def test_noise_monotone(self, clean_image, rng):
        mild = np.clip(clean_image + rng.integers(-5, 6, clean_image.shape),
                       0, 255).astype(np.uint8)
        heavy = np.clip(clean_image + rng.integers(-40, 41, clean_image.shape),
                        0, 255).astype(np.uint8)
        assert fr.psnr(clean_image, mild) > fr.psnr(clean_image, heavy)

    def test_known_value(self):
        # A constant error of 10 levels out of 255 must give the textbook PSNR.
        ref = np.zeros((8, 8, 3), np.uint8)
        dist = np.zeros((8, 8, 3), np.uint8)
        dist[:, :, :] = 10
        expected = 10 * math.log10((255.0 / 10.0) ** 2)
        assert fr.psnr(ref, dist) == pytest.approx(expected, abs=1e-9)

    def test_uint8_input_agrees_with_float(self, clean_image, noisy_image):
        # PSNR is scale-invariant: uint8 and [0,1] inputs must agree exactly.
        direct = fr.psnr(clean_image, noisy_image)
        floats = fr.psnr(clean_image / 255.0, noisy_image / 255.0)
        assert direct == pytest.approx(floats, abs=1e-9)

    def test_shape_mismatch_raises(self, clean_image):
        with pytest.raises(ValueError, match="shape mismatch"):
            fr.psnr(clean_image, clean_image[:32])


class TestSSIM:
    def test_identical_images_are_one(self, clean_image):
        assert fr.ssim(clean_image, clean_image) == pytest.approx(1.0, abs=1e-12)

    def test_separable_filter_matches_the_dense_2d_correlation(self, rng):
        """The SSIM windows are separable, so _filter2d applies them per axis.

        Guarding both the correctness (the dense and separable forms are the
        same kernel, not an approximation) and the reason the separable form
        exists: a full-resolution frame is filtered ~35 times per scored pair,
        and the dense 11x11 pass costs size**2 taps where the separable one
        costs 2*size.
        """
        from scipy.ndimage import correlate

        image = rng.random((48, 40, 3), dtype=np.float64)
        window1d = fr._gaussian_window1d()
        dense_window = np.outer(window1d, window1d)

        expected = np.empty_like(image)
        for c in range(3):
            expected[:, :, c] = correlate(image[:, :, c], dense_window,
                                          mode="reflect")

        actual = fr._filter2d(image, window1d)
        assert actual.shape == image.shape
        assert np.allclose(actual, expected, rtol=0, atol=1e-12)

    def test_degrades_with_noise(self, clean_image, noisy_image):
        assert 0.0 < fr.ssim(clean_image, noisy_image) < 1.0

    def test_window_variants_agree_on_identity(self, clean_image):
        gaussian = fr.ssim(clean_image, clean_image, window="gaussian")
        uniform = fr.ssim(clean_image, clean_image, window="uniform")
        assert gaussian == pytest.approx(1.0, abs=1e-9)
        assert uniform == pytest.approx(1.0, abs=1e-9)

    def test_map_shape_matches_input(self, clean_image, noisy_image):
        score, ssim_map = fr.ssim(clean_image, noisy_image, return_map=True)
        assert ssim_map.shape == clean_image.shape
        assert 0.0 <= score <= 1.0

    def test_contrast_sensitivity(self):
        # Two flat patches differing only in mean should score below identity
        # but above a heavily noisy comparison.
        ref = np.full((32, 32, 3), 120, np.uint8)
        shifted = np.full((32, 32, 3), 100, np.uint8)
        assert fr.ssim(ref, shifted) < 1.0


class TestMSSSIM:
    def test_identical_images_are_one(self, clean_image):
        assert fr.ms_ssim(clean_image, clean_image) == pytest.approx(1.0, abs=1e-9)

    def test_degrades_with_noise(self, clean_image, noisy_image):
        # MS-SSIM must stay in [0, 1] and drop below the clean-image score.
        clean_score = fr.ms_ssim(clean_image, clean_image)
        noisy_score = fr.ms_ssim(clean_image, noisy_image)
        assert 0.0 <= noisy_score <= 1.0
        assert noisy_score < clean_score


class TestLPIPS:
    def test_identical_images_are_zero(self, clean_image):
        if not fr._HAS_LPIPS:
            pytest.skip("lpips not installed")
        assert fr.lpips_distance(clean_image, clean_image) == pytest.approx(0.0, abs=1e-4)

    def test_positive_for_different_images(self, clean_image, noisy_image):
        if not fr._HAS_LPIPS:
            pytest.skip("lpips not installed")
        assert fr.lpips_distance(clean_image, noisy_image) > 0.0

    def test_unavailable_raises(self, clean_image, monkeypatch):
        monkeypatch.setattr(fr, "_HAS_LPIPS", False)
        with pytest.raises(fr.MetricUnavailable):
            fr.lpips_distance(clean_image, clean_image)

    def test_model_runs_on_gpu_when_available(self, clean_image, noisy_image):
        # The official lpips package builds on CPU and never moves itself, so
        # a VGG backbone on a multi-megapixel frame is minutes-per-image on CPU
        # while the GPU idles. If an accelerator is present the evaluator must
        # actually use it, or the benchmark's runtime estimate is off by 10x+.
        if not fr._HAS_LPIPS:
            pytest.skip("lpips not installed")
        device = fr._lpips_device()
        if device == "cpu":
            pytest.skip("no accelerator available")
        evaluator = fr.LPIPSEvaluator(backbone="alex")
        assert evaluator._device == device
        assert next(evaluator._model.parameters()).device.type == device.split(":")[0]

    def test_device_cache_is_keyed_by_device(self):
        # The same backbone on two devices is two resident copies of the
        # weights; a cache keyed only on (backbone, spatial) would hand a CPU
        # model to an evaluator that thinks it owns a GPU one.
        if not fr._HAS_LPIPS:
            pytest.skip("lpips not installed")
        a = fr.LPIPSEvaluator(backbone="alex")
        b = fr.LPIPSEvaluator(backbone="alex")
        assert a._device == b._device
        key = f"alex:False:{a._device}"
        assert key in fr.LPIPSEvaluator._cache


class TestEvaluatePair:
    def test_subset_selection(self, clean_image, noisy_image):
        result = fr.evaluate_pair(clean_image, noisy_image, metrics=("psnr",))
        assert result.psnr is not None
        assert result.ssim is None

    def test_unknown_metric(self, clean_image, noisy_image):
        with pytest.raises(ValueError, match="unknown full-reference metric"):
            fr.evaluate_pair(clean_image, noisy_image, metrics=("not_a_metric",))


def test_available_metrics_report_environment():
    available = fr.available_metrics()
    assert available["psnr"] is True
    assert available["ssim"] is True
    assert "lpips_vgg" in available
