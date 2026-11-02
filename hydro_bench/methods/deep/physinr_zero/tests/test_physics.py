"""Property tests for the physical layer.

These are the identities the whole method rests on.  If any of them fails the
implementation disagrees with the image-formation model of the methodology,
independent of anything learned.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydro_bench.methods.deep.physinr_zero import physics
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig


cfg = PhysINRConfig()


def _img(h=16, w=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)


class TestJerlovTable:
    def test_all_ten_types_present(self):
        expected = {"I", "IA", "IB", "II", "III", "1", "3", "5", "7", "9"}
        assert set(physics.JERLOV_TABLE) == expected

    def test_table_is_rectangular_over_the_declared_wavelengths(self):
        n = len(physics.JERLOV_WAVELENGTHS)
        for k, v in physics.JERLOV_TABLE.items():
            assert len(v) == n, f"{k} has {len(v)} entries, expected {n}"
            assert all(x >= 0 for x in v), f"{k} has a negative Kd"

    def test_attenuation_is_monotone_across_the_oceanic_progression(self):
        """Clearer water attenuates less.  This holds strictly within the
        oceanic progression I -> III, which is what the physics guarantees."""
        order = ["I", "IA", "IB", "II", "III"]
        for i in range(len(physics.JERLOV_WAVELENGTHS)):
            col = [physics.JERLOV_TABLE[k][i] for k in order]
            assert (np.diff(col) >= -1e-12).all(), \
                f"oceanic Kd decreases at {physics.JERLOV_WAVELENGTHS[i]}nm: {col}"

    def test_the_known_coastal_non_monotonicity_is_preserved(self):
        """At 700 nm Jerlov type III (0.66) attenuates more than coastal type 1
        (0.65) in the published table: the coastal types are not simply
        'oceanic plus more stuff'.  Pinning the value guards against a
        transcription slip that would quietly 'fix' the table."""
        assert physics.JERLOV_TABLE["III"][-1] == pytest.approx(0.66)
        assert physics.JERLOV_TABLE["1"][-1] == pytest.approx(0.65)
        assert physics.JERLOV_TABLE["1"][-1] < physics.JERLOV_TABLE["III"][-1]

    def test_type_i_uses_the_austin_petzold_correction(self):
        """Jerlov's own type-I row lies below pure-water attenuation at most
        wavelengths; the corrected matrix substitutes Austin & Petzold (1986).
        Every other row must be untouched by the substitution."""
        corrected = physics.jerlov_beta_matrix(austin_petzold=True)
        original = physics.jerlov_beta_matrix(austin_petzold=False)
        i = physics.JERLOV_TYPES.index("I")
        # The substituted row is integrated against the same Gaussian channel
        # response as every other row; the weights are recomputed here from the
        # Gaussian itself so the test is not a restatement of the code.
        waves = np.asarray(physics.JERLOV_WAVELENGTHS, dtype=np.float64)
        sigma = physics.RGB_SPECTRAL_SIGMA_NM
        ap = np.asarray(physics.AUSTIN_PETZOLD_TYPE_I, dtype=np.float64)
        for c, channel in enumerate(physics.CHANNELS):
            lam = physics.RGB_WAVELENGTHS[channel]
            w = np.exp(-0.5 * ((waves - lam) / sigma) ** 2)
            w = w / w.sum()
            assert corrected[i, c] == pytest.approx(float(w @ ap), abs=1e-9)
        others = [k for k in range(len(physics.JERLOV_TYPES)) if k != i]
        assert corrected[others] == pytest.approx(original[others], abs=1e-12)

    def test_beta_matrix_shape_and_finiteness(self):
        m = physics.jerlov_beta_matrix()
        assert m.shape == (cfg.num_water_types, 3)
        assert np.isfinite(m).all()
        assert (m > 0).all()


class TestSpectralParameterisation:
    """A camera channel integrates a band of wavelengths, not a line."""

    def test_weights_are_normalised_and_peak_at_the_channel_wavelength(self):
        for channel in physics.CHANNELS:
            lam = physics.RGB_WAVELENGTHS[channel]
            w = physics.channel_spectral_weights(lam)
            assert w.sum() == pytest.approx(1.0)
            waves = np.asarray(physics.JERLOV_WAVELENGTHS)
            assert int(np.argmax(w)) == int(np.argmin(np.abs(waves - lam)))

    def test_spectral_beta_stays_inside_the_curves_range(self):
        """Integrating over a band must stay between the curve's min and max,
        including for the Austin-Petzold-substituted type I."""
        m = physics.jerlov_beta_matrix()
        for k, wtype in enumerate(physics.JERLOV_TYPES):
            curve = np.asarray(physics.AUSTIN_PETZOLD_TYPE_I if wtype == "I"
                               else physics.JERLOV_TABLE[wtype], dtype=np.float64)
            for c in range(3):
                assert curve.min() - 1e-12 <= m[k, c] <= curve.max() + 1e-12

    def test_spectral_beta_converges_to_the_nearest_wavelength(self):
        """As the band narrows the integral must collapse onto the single
        nearest tabulated wavelength -- the limit the single-wavelength reading
        approximates."""
        waves = np.asarray(physics.JERLOV_WAVELENGTHS, dtype=np.float64)
        k = physics.JERLOV_TYPES.index("III")
        lam = physics.RGB_WAVELENGTHS["r"]
        nearest = float(np.asarray(physics.JERLOV_TABLE["III"])[
            int(np.argmin(np.abs(waves - lam)))])
        for sigma in (8.0, 2.0, 0.5):
            m = physics.jerlov_beta_matrix(sigma_nm=sigma)
            tol = 1e-3 if sigma > 5.0 else 1e-9
            assert m[k, 0] == pytest.approx(nearest, abs=tol)

    def test_single_wavelength_mode_is_the_interpolated_table(self):
        """``spectral=False`` reproduces the Assumption A3 reading exactly, so
        the ablation can re-derive the pre-refactoring numbers."""
        interp = physics.jerlov_beta_matrix(spectral=False)
        waves = np.asarray(physics.JERLOV_WAVELENGTHS, dtype=np.float64)
        for k, wtype in enumerate(physics.JERLOV_TYPES):
            curve = physics.AUSTIN_PETZOLD_TYPE_I if wtype == "I" \
                else physics.JERLOV_TABLE[wtype]
            for c, channel in enumerate(physics.CHANNELS):
                assert interp[k, c] == pytest.approx(
                    float(np.interp(physics.RGB_WAVELENGTHS[channel], waves, curve)),
                    abs=1e-12)


class TestTransmission:
    def test_transmission_is_exponential_in_tau(self):
        tau = torch.rand(1, 8, 8, 3) * cfg.tau_max
        t = physics.transmission_from_tau(tau)
        assert torch.allclose(t, torch.exp(-tau), atol=1e-6)

    def test_transmission_bounds(self):
        for v in (0.0, cfg.tau_max, 2.0 * cfg.tau_max):
            t = physics.transmission_from_tau(torch.full((1, 4, 4, 3), v))
            assert bool((t > 0).all()) and bool((t <= 1.0 + 1e-9).all())


class TestImageFormation:
    def test_regrade_inverts_physical_inverse_at_moderate_transmission(self):
        """regrade . inverse = I wherever the eps guard is negligible.

        The residual is not zero by design: the inverse divides by (t + eps),
        so the round trip carries a relative error of eps/(t + eps).  At
        t ~ 0.5 and eps = 1e-4 that is ~2e-4, and it grows as t falls -- which
        is exactly the regime where eps exists to keep the inverse finite.
        """
        img = _img()
        tau = torch.rand_like(img) * 0.7          # t >= ~0.5
        t = physics.transmission_from_tau(tau)
        a = torch.rand_like(img) * 0.6 + 0.2
        raw = physics.physical_inverse(img, a, t, cfg.eps)
        back = physics.regrade(raw, t, a)
        # Predicted bound: eps / min(t), with headroom for float32.
        bound = 4.0 * cfg.eps / float(t.min())
        err = (back - img).abs().max().item()
        assert err < bound, f"{err:.2e} exceeds the eps-bound {bound:.2e}"

    def test_inverse_is_finite_at_zero_transmission(self):
        """The eps guard must keep the inverse finite, never inf/NaN."""
        img = _img()
        tau = torch.full_like(img, cfg.tau_max)   # t = exp(-8) ~ 3e-4
        t = physics.transmission_from_tau(tau)
        a = torch.full_like(img, 0.5)
        raw = physics.physical_inverse(img, a, t, cfg.eps)
        assert torch.isfinite(raw).all()

    def test_degraded_image_stays_in_range(self):
        img = _img()
        tau = torch.rand_like(img) * cfg.tau_max
        t = physics.transmission_from_tau(tau)
        a = torch.rand_like(img) * 0.8 + 0.1
        out = physics.regrade(img, t, a)
        assert bool((out >= 0).all()) and bool((out <= 1.0).all())


class TestBoundedNonlinear:
    def test_identity_at_zero_alpha(self):
        img = _img()
        out = physics.bounded_nonlinear(img, torch.zeros_like(img))
        assert torch.allclose(out, img, atol=1e-6)

    def test_output_range(self):
        img = _img()
        for a in (-2.0, -0.5, 0.5, 2.0):
            alpha = torch.full_like(img, a)
            out = physics.bounded_nonlinear(img, alpha)
            assert bool((out >= 0).all()) and bool((out <= 1.0).all()), \
                f"out of range for alpha={a}"

    def test_monotone_increasing(self):
        """g must be a monotone map of [0,1] onto [0,1]; check on a grid."""
        v = torch.linspace(0, 1, 201).unsqueeze(-1)
        g = physics.bounded_nonlinear(v, torch.zeros_like(v))
        assert bool((torch.diff(g, dim=0) >= -1e-6).all())
        assert g[0].item() == pytest.approx(0.0, abs=1e-6)
        assert g[-1].item() == pytest.approx(1.0, abs=1e-6)


class TestLab:
    def test_grey_maps_to_zero_chroma(self):
        """A neutral grey has no chroma.  The residual (~1.5e-3 in Lab units)
        is float32 accumulation through the sRGB->XYZ matrix against the D65
        white point, not a colour cast: it is four orders of magnitude below
        the just-noticeable difference."""
        grey = torch.full((1, 4, 4, 3), 0.5)
        lab = physics.rgb_to_lab(grey)
        assert lab[..., 0].mean().item() == pytest.approx(
            53.39, abs=0.5)          # L for 50% grey under D65
        assert lab[..., 1].abs().max().item() < 1e-2
        assert lab[..., 2].abs().max().item() < 1e-2

    def test_f_lab_has_no_infinite_gradient(self):
        """The cube root's gradient blows up at zero; the implementation must
        floor the radicand so backward stays finite (this was a real NaN)."""
        t = torch.zeros(5, requires_grad=True)
        y = physics._f_lab(t)
        y.sum().backward()
        assert torch.isfinite(t.grad).all()

    def test_lab_range_is_physical(self):
        lab = physics.rgb_to_lab(_img(24, 24, seed=7))
        assert bool((lab[..., 0] >= 0).all()) and bool((lab[..., 0] <= 100.5).all())


class TestGradients:
    def test_gradient_pair_shapes_and_values(self):
        x = torch.arange(12.0).reshape(1, 3, 4)
        dy, dx = physics.gradient_pair(x)
        assert dy.shape == x.shape and dx.shape == x.shape
        # dy is zero-padded: row 0 is 0, row i = x[i] - x[i-1]
        assert dy[0, 0, 0].item() == 0.0
        assert dy[0, 1, 0].item() == pytest.approx(4.0)
        assert dx[0, 0, 0].item() == 0.0
        assert dx[0, 0, 1].item() == pytest.approx(1.0)

    def test_constant_image_has_zero_gradients(self):
        x = torch.full((1, 5, 7), 0.3)
        dy, dx = physics.gradient_pair(x)
        assert dy.abs().max().item() == 0.0
        assert dx.abs().max().item() == 0.0


class TestYIQ:
    """The colour space of the final alignment stage, and the cast detector
    that drives it."""

    def test_round_trip_is_the_identity(self):
        x = _img(24, 24, seed=3)
        back = physics.yiq_to_rgb(physics.rgb_to_yiq(x))
        assert torch.allclose(back, x, atol=1e-5)

    def test_matrices_are_true_inverses(self):
        m = physics._RGB2YIQ @ physics._YIQ2RGB
        assert torch.allclose(m, torch.eye(3), atol=1e-5)

    def test_luminance_is_untouched_by_the_alignment(self):
        """Scaling I and Q must not move Y -- the invariant the stage is named
        for, and what keeps it from trading brightness for colour."""
        x = _img(24, 24, seed=5)
        y0 = physics.rgb_to_yiq(x)[..., 0]
        aligned = physics.yiq_chroma_align(x, torch.tensor(0.6), torch.tensor(0.4))
        y1 = physics.rgb_to_yiq(aligned)[..., 0]
        assert torch.allclose(y0, y1, atol=1e-6)

    def test_alignment_preserves_the_range(self):
        """Chroma shrinkage moves a point toward the Y axis; the RGB cube is
        convex, so the segment stays inside it."""
        x = _img(24, 24, seed=6)
        a = physics.yiq_chroma_align(x, torch.tensor(0.3), torch.tensor(0.3))
        assert bool((a >= 0).all() and (a <= 1.0).all())
        assert torch.isfinite(a).all()

    def test_unit_gains_are_the_identity(self):
        x = _img(20, 20, seed=8)
        a = physics.yiq_chroma_align(x, torch.tensor(1.0), torch.tensor(1.0))
        assert torch.allclose(a, x, atol=1e-5)

    def test_a_green_cast_is_detected_and_shrunk(self):
        """The failure this refactor targets: a green image has negative mean
        chroma on both axes and must get gains below 1."""
        g = torch.zeros(1, 32, 32, 3)
        g[..., 0] = 0.15
        g[..., 1] = 0.60
        g[..., 2] = 0.15
        mu, _ = physics.yiq_chroma_stats(g)
        assert bool((mu < 0).all()), f"green cast should be negative: {mu}"
        gains = physics.yiq_chroma_gains(g, strength=1.0, floor=0.0)
        assert bool((gains < 1.0).all()), f"gains not shrunk: {gains}"

    def test_a_neutral_image_is_left_alone(self):
        gray = torch.full((1, 32, 32, 3), 0.5)
        gains = physics.yiq_chroma_gains(gray, strength=1.0, floor=0.0)
        assert torch.allclose(gains, torch.ones(1, 2), atol=1e-4)

    def test_chroma_spread_spares_a_colourful_scene(self):
        """The saturation guard: two images with the same chroma *bias* but
        different chroma *spread* must get different gains, so a genuinely
        colourful image is not desaturated like a cast."""
        flat = torch.zeros(1, 64, 64, 3)
        flat[..., 0] = 0.15
        flat[..., 1] = 0.60
        flat[..., 2] = 0.15
        gen = torch.Generator().manual_seed(1)
        colourful = (flat + 0.30 * torch.rand(1, 64, 64, 3, generator=gen)
                     - 0.15).clamp(0.0, 1.0)
        g_flat = physics.yiq_chroma_gains(flat, strength=0.5, floor=0.0)[0]
        g_col = physics.yiq_chroma_gains(colourful, strength=0.5, floor=0.0)[0]
        assert bool((g_flat < 1.0).all())
        assert bool((g_col > g_flat).any()), \
            f"colourful image not spared: {g_col} vs {g_flat}"

    def test_gains_respect_the_floor(self):
        g = torch.zeros(1, 32, 32, 3)
        g[..., 0] = 0.02
        g[..., 1] = 0.90
        g[..., 2] = 0.02
        gains = physics.yiq_chroma_gains(g, strength=1.0, floor=0.7)
        assert bool((gains >= 0.7 - 1e-6).all())
        assert bool((gains <= 1.0).all())


class TestBackgroundLight:
    """The per-channel B_inf estimator that brackets the ambient field."""

    def test_shape_range_and_finiteness(self):
        x = _img(32, 40, seed=11)
        b = physics.estimate_background_light(x)
        assert b.shape == (1, 3)
        assert bool((b >= 0).all() and (b <= 1).all())
        assert torch.isfinite(b).all()

    def test_a_green_deep_region_gives_green_background_light(self):
        """The estimator's whole purpose: the deep region's colour is the
        water column's colour, which is what a green cast consists of."""
        x = torch.full((1, 48, 48, 3), 0.6)
        x[:, :24, :24, 0] = 0.05
        x[:, :24, :24, 1] = 0.55
        x[:, :24, :24, 2] = 0.10
        b = physics.estimate_background_light(x, dark_fraction=0.25,
                                              percentile=0.9)
        assert b[0, 1] > b[0, 0]
        assert b[0, 1] > b[0, 2]

    def test_the_deep_region_is_selected_by_red_not_by_luminance(self):
        """The darkest 10% is taken on the *red* channel: red is extinguished
        first with range, so red-depleted pixels mark optically deep water
        even when the rest of the frame is darker overall."""
        x = torch.zeros(1, 48, 48, 3)
        # Top half: red-deepest, but bright in green and blue.
        x[:, :24] = torch.tensor([0.02, 0.9, 0.9])
        # Bottom half: darker luminance, but red is far from extinguished.
        x[:, 24:] = torch.tensor([0.5, 0.0, 0.0])
        b = physics.estimate_background_light(x, dark_fraction=0.5,
                                              percentile=1.0)
        assert b[0, 1] > 0.8 and b[0, 2] > 0.8, \
            f"the red-deepest region was not selected: {b}"
        assert b[0, 0] < 0.1

    def test_bad_arguments_raise(self):
        x = _img(16, 16, seed=1)
        with pytest.raises(ValueError, match="B,H,W,3"):
            physics.estimate_background_light(x[0])
        with pytest.raises(ValueError, match="dark_fraction"):
            physics.estimate_background_light(x, dark_fraction=1.5)
        with pytest.raises(ValueError, match="percentile"):
            physics.estimate_background_light(x, percentile=0.0)


class TestCdom:
    """The gelbstoff absorption added to the synthetic degradations."""

    def test_the_reference_wavelength_is_exact(self):
        # 440 nm is not on the Jerlov grid, so the exact point is probed
        # directly rather than through it.
        assert physics.cdom_absorption([440], a440=0.3)[0] == pytest.approx(0.3)

    def test_absorption_decays_monotonically_with_wavelength(self):
        a = physics.cdom_absorption(physics.JERLOV_WAVELENGTHS, a440=0.4)
        assert bool((np.diff(a) <= 1e-12).all())
        # Blue attenuates more than red -- the spectral signature of the cast.
        waves = list(physics.JERLOV_WAVELENGTHS)
        assert a[waves.index(450)] > a[waves.index(650)]

    def test_zero_cdom_is_zero_absorption(self):
        a = physics.cdom_absorption(physics.JERLOV_WAVELENGTHS, a440=0.0)
        assert bool((a == 0.0).all())


class TestTransmissionLevel:
    """The v2 data-driven prior level: range(I_c) = t_c * range(J_c)."""

    def test_an_untenuated_scene_reads_t_one(self):
        # A natural-contrast image, no channel favoured: all t = 1.
        rng = np.random.default_rng(0)
        img = rng.random((1, 48, 64, 3)).astype(np.float32)
        t = physics.estimate_transmission_level(torch.from_numpy(img))
        assert t.shape == (1, 3)
        assert bool((t > 0.95).all())

    def test_a_compressed_channel_reads_a_lower_t(self):
        # Blue compressed to a third of the red range: the water removed
        # blue contrast, so t_blue ~ 1/3 while the red channel is the
        # reference at 1.
        img = np.zeros((1, 48, 64, 3), np.float32)
        img[..., 0] = np.linspace(0.1, 0.7, 64)[None, None, :]
        img[..., 2] = 0.4 + 0.1 * np.linspace(0, 1, 64)[None, None, :]
        t = physics.estimate_transmission_level(torch.from_numpy(img))
        assert t[0, 0] == pytest.approx(1.0)
        assert t[0, 2] < 0.45

    def test_the_estimation_is_detached(self):
        img = torch.rand(1, 32, 32, 3, requires_grad=False)
        t = physics.estimate_transmission_level(img)
        assert not t.requires_grad

    def test_flat_images_hit_the_range_floor(self):
        # A near-constant image has no contrast signal; the range floor keeps
        # the estimate finite and t near 1 rather than dividing by ~0.
        img = torch.full((1, 32, 32, 3), 0.5)
        t = physics.estimate_transmission_level(img)
        assert bool((t == 1.0).all())

    def test_bad_percentiles_raise(self):
        img = torch.rand(1, 8, 8, 3)
        with pytest.raises(ValueError, match="p_low"):
            physics.estimate_transmission_level(img, p_low=0.9, p_high=0.1)


class TestInverseFloor:
    """t_min caps the inverse's amplification -- the anti-blowout guard."""

    def test_a_low_transmission_no_longer_explodes(self):
        img = torch.full((1, 8, 8, 3), 0.5)
        amb = torch.full((1, 8, 8, 3), 0.1)
        t = torch.full((1, 8, 8, 3), 0.05)
        raw = physics.physical_inverse(img, amb, t, t_min=0.25)
        # Amplification capped at 1/0.25 = 4x: (0.5 - 0.075)/0.25 = 1.7.
        assert float(raw.max()) <= 1.7 + 1e-6
        unfloored = physics.physical_inverse(img, amb, t)
        assert float(unfloored.max()) > 7.0   # the v1 behaviour

    def test_the_floor_is_inert_when_transmission_is_high(self):
        img = torch.rand(1, 8, 8, 3)
        amb = torch.rand(1, 8, 8, 3) * 0.2
        t = torch.full((1, 8, 8, 3), 0.8)
        assert torch.allclose(physics.physical_inverse(img, amb, t, t_min=0.25),
                              physics.physical_inverse(img, amb, t))

    def test_regrade_with_the_same_floored_t_roundtrips(self):
        img = torch.rand(1, 8, 8, 3)
        amb = torch.rand(1, 8, 8, 3) * 0.2
        t = torch.full((1, 8, 8, 3), 0.05).clamp(min=0.25)
        raw = physics.physical_inverse(img, amb, t)
        back = physics.regrade(raw, t, amb)
        # 1e-3, not tighter: the inverse's eps guard (denominator t + eps)
        # makes the roundtrip approximate by design.
        assert torch.allclose(back, img, atol=1e-3)


class TestDisplayNormalize:
    """§E display normalisation: the reference-free render-time stretch."""

    def test_output_stays_in_unit_range(self):
        out = physics.display_normalize(_img(seed=1))
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_full_stretch_maps_percentiles_to_the_boundaries(self):
        # A channel whose p1/p99 range is wide: after a full-strength stretch
        # the channel minimum is 0 and the maximum is 1.
        img = _img(seed=2)
        out = physics.display_normalize(img, 0.0, 1.0, strength=1.0)
        mn = out.reshape(-1, 3).min(dim=0).values
        mx = out.reshape(-1, 3).max(dim=0).values
        assert torch.allclose(mn, torch.zeros(3), atol=1e-5)
        assert torch.allclose(mx, torch.ones(3), atol=1e-5)

    def test_strength_zero_is_the_identity(self):
        img = _img(seed=3)
        out = physics.display_normalize(img, strength=0.0)
        assert torch.allclose(out, img, atol=1e-6)

    def test_strength_blend_is_the_exact_convex_combination(self):
        img = _img(seed=4)
        full = physics.display_normalize(img, strength=1.0)
        half = physics.display_normalize(img, strength=0.5)
        expected = 0.5 * (img + full)
        assert torch.allclose(half, expected, atol=1e-5)

    def test_contrast_increases_on_low_contrast_frame(self):
        # The actual motivating case: a compressed-histogram frame must come
        # out with a wider spread.
        g = torch.Generator().manual_seed(6)
        base = 0.5 + 0.05 * torch.rand(1, 24, 32, 3, generator=g)
        out = physics.display_normalize(base, 0.01, 0.99)
        assert out.std() > base.std()

    def test_constant_channel_maps_to_zero_not_explosion(self):
        flat = torch.full((1, 8, 8, 1), 0.37)
        img = flat.expand(1, 8, 8, 3).contiguous()
        out = physics.display_normalize(img, 0.01, 0.99)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)
        assert torch.isfinite(out).all()

    def test_batched_input_matches_single_images(self):
        img = _img(seed=7)
        two = torch.cat([img, img], dim=0)
        out_batch = physics.display_normalize(two)
        out_single = physics.display_normalize(img)
        assert torch.allclose(out_batch[0], out_single[0], atol=1e-6)

    def test_accepts_flat_batched_coordinates(self):
        img = torch.rand(2, 64, 3, generator=torch.Generator().manual_seed(8))
        out = physics.display_normalize(img)
        assert out.shape == img.shape
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_invalid_percentiles_raise(self):
        with pytest.raises(ValueError):
            physics.display_normalize(_img(), 0.9, 0.5)
        with pytest.raises(ValueError):
            physics.display_normalize(_img(), -0.1, 0.99)

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            physics.display_normalize(_img(), strength=1.5)

    def test_wrong_channel_count_raises(self):
        with pytest.raises(ValueError):
            physics.display_normalize(torch.rand(1, 8, 8, 4))
