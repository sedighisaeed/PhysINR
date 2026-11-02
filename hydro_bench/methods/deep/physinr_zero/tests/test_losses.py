"""Tests for the objective: schedule, monotone improvement, and no NaNs.

The loss is the only place a silent numerical fault can hide for a whole run
(a zero group weight masks a NaN gradient as 0 * NaN = NaN and everything
downstream goes quiet).  These tests pin the properties that keep that from
happening again.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

from hydro_bench.methods.deep.physinr_zero import physics
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.losses import PhysINRLoss
from hydro_bench.methods.deep.physinr_zero.model import PhysINR, PhysOutputs, strided_image
from hydro_bench.methods.deep.physinr_zero.synthesize import SyntheticDegradation

cfg = PhysINRConfig()


def _run(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, n, n, 3, generator=g)


def _outputs(tau, tau_jerlov=None, ambient=None, ambient_estimate=None):
    """A PhysOutputs whose only non-zero fields are the ones the bound and
    ambient-prior terms read.

    The refactor's new terms are functions of ``tau``, ``tau_jerlov``,
    ``ambient`` and ``ambient_estimate`` alone, so a hand-built tuple is the
    honest way to test them against a known input -- through the model the
    fields would all be coupled.
    """
    b, h, w, _ = tau.shape
    z = tau.new_zeros
    tj = z(b, 3) if tau_jerlov is None else tau_jerlov
    amb = z(b, h, w, 3) if ambient is None else ambient
    est = z(b, 3) if ambient_estimate is None else ambient_estimate
    return PhysOutputs(
        tau=tau,
        transmission=torch.exp(-tau.clamp(min=0.0)),
        tau_jerlov=tj,
        ambient=amb,
        alpha=z(b, h, w, 3),
        restored=z(b, h, w, 3),
        regraded=z(b, h, w, 3),
        regraded_plus=z(b, h, w, 3),
        water_weights=z(b, cfg.num_water_types),
        tau_delta=z(b, h, w, 3),
        ambient_estimate=est,
    )


class TestSchedule:
    def test_phys_and_cycle_are_at_maximum_from_step_zero(self):
        fn = PhysINRLoss(cfg)
        assert fn.group_weight("phys", 0, 10) == cfg.w_phys
        assert fn.group_weight("cycle", 0, 10) == cfg.w_cycle

    def test_scheduled_groups_ramp_up(self):
        fn = PhysINRLoss(cfg)
        for g in ("color", "struct", "spatial", "jerlov", "cast", "bound"):
            w0 = fn.group_weight(g, 0, 10)
            wm = fn.group_weight(g, 5, 10)
            assert 0 < w0 < wm, f"{g} did not ramp up: {w0} -> {wm}"
            assert wm <= fn.group_weight(g, 9, 10)

    def test_the_two_new_groups_are_also_scheduled(self):
        """The cast and bound groups are regularisers, so they follow the same
        cosine warm-up as the other secondary losses rather than switching on
        at full strength while the physical solution is still forming."""
        fn = PhysINRLoss(cfg)
        for g in ("cast", "bound"):
            assert fn.group_weight(g, 0, 10) < fn.group_weight(g, 9, 10)

    def test_no_group_weight_is_ever_exactly_zero(self):
        """The ramp floor exists because 0 * NaN = NaN: a zeroed group would
        mask a divergent gradient and let it poison the optimiser silently."""
        fn = PhysINRLoss(cfg)
        for g in PhysINRLoss.GROUPS:
            for s in range(3 * cfg.warmup_steps):
                assert fn.group_weight(g, s, 10) > 0.0, f"{g} is 0 at step {s}"

    def test_water_weight_grows_linearly(self):
        fn = PhysINRLoss(cfg)
        w0 = fn._water_weight(0, 10)
        w9 = fn._water_weight(9, 10)
        assert 0 < w0 < w9
        assert w9 == pytest.approx(cfg.w_water)


class TestLossComposition:
    def test_breakdown_terms_sum_is_the_total(self):
        """The total must be the weighted sum of the groups, and every term
        in the breakdown must be finite -- a NaN in one term is the failure
        mode this test exists to catch."""
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        img = _run()
        out = model(img.permute(0, 3, 1, 2), step=0)
        bd = loss_fn(img, out, 0, 8)

        for k, v in bd.terms.items():
            assert torch.isfinite(torch.tensor(v)).item(), f"term {k} is {v}"
        assert torch.isfinite(bd.total).item()
        assert bd.weights  # non-empty

    def test_total_is_differentiable(self):
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        img = _run()
        out = model(img.permute(0, 3, 1, 2), step=3)
        bd = loss_fn(img, out, 3, 8)
        bd.total.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None])
        assert torch.isfinite(g).all()
        assert g.abs().max().item() > 0.0

    def test_shape_mismatch_raises(self):
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        out = model(torch.rand(1, 3, 32, 32), step=0)
        with pytest.raises(ValueError, match="must match in shape"):
            loss_fn(_run(40), out, 0, 8)

    def test_adaptation_lowers_the_objective_under_fixed_weights(self):
        """A real optimisation signal, not just a finite one.

        Comparing L_total at step 0 against step K is not valid: the §20
        schedule deliberately changes the weights as adaptation proceeds, so
        L_total is expected to *rise* on a perfectly converging run as groups
        switch on.  Instead evaluate both checkpoints under the step-0 weight
        vector, which makes the comparison apples-to-apples."""
        torch.manual_seed(0)
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        synth = SyntheticDegradation(cfg)
        import copy
        rng = np.random.default_rng(0)
        clean = _run(96, seed=1)
        tp = synth.sample_task(rng, (96, 96), clean.reshape(-1, 3).mean(0))
        deg, _ = synth.degrade(clean.permute(0, 3, 1, 2), tp, rng)
        img = deg.permute(0, 2, 3, 1)

        w0 = {g: loss_fn.group_weight(g, 0, 20) for g in loss_fn.GROUPS}

        def evaluate(m):
            m.eval()
            with torch.no_grad():
                out = m(deg, step=0)
                return float(loss_fn(img, out, 0, 20, weights=w0).total)

        before = evaluate(model)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        for s in range(25):
            opt.zero_grad(set_to_none=True)
            out = model(deg, step=s)
            bd = loss_fn(img, out, s, 25)
            bd.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        after = evaluate(model)
        assert after < before, f"objective did not fall: {before:.4f} -> {after:.4f}"

    def test_the_total_loss_is_finite_throughout_adaptation(self):
        """The NaN failure mode this suite guards against: every step of a
        real adaptation loop must stay finite, including the scheduled
        groups that are near their ramp floor early on."""
        torch.manual_seed(1)
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(1)
        clean = _run(64, seed=9)
        tp = synth.sample_task(rng, (64, 64), clean.reshape(-1, 3).mean(0))
        deg, _ = synth.degrade(clean.permute(0, 3, 1, 2), tp, rng)

        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        for s in range(20):
            opt.zero_grad(set_to_none=True)
            out = model(deg, step=s)
            bd = loss_fn(deg.permute(0, 2, 3, 1), out, s, 20)
            bd.total.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(bd.total).item()
            assert bool(torch.isfinite(gn).all()), f"gradnorm NaN at step {s}"
            opt.step()


class TestDegradationSynthesizer:
    def test_degradation_preserves_shape_and_range(self):
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(0)
        clean = _run(64, seed=2)
        tp = synth.sample_task(rng, (64, 64), clean.reshape(-1, 3).mean(0))
        deg, ref = synth.degrade(clean.permute(0, 3, 1, 2), tp, rng)
        assert deg.shape == clean.permute(0, 3, 1, 2).shape
        assert ref.shape == clean.permute(0, 3, 1, 2).shape
        assert (deg >= 0).all() and (deg <= 1.0).all()
        assert (ref >= 0).all() and (ref <= 1.0).all()

    def test_degradation_grows_monotonically_with_the_water_column(self):
        """The physically meaningful claim is not "the image gets darker" -- a
        thick water column drives the observation toward the ambient veiling
        luminance, which can be *brighter* than a dark scene.  What must hold
        is that the deviation from the clean image grows with depth."""
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(3)
        clean = _run(64, seed=4) * 0.6 + 0.2
        base = synth.sample_task(rng, (64, 64), clean.reshape(-1, 3).mean(0))
        c = clean.permute(0, 3, 1, 2)
        prev = 0.0
        for depth in (0.5, 2.0, 5.0, 10.0):
            tp = replace(base, depth_scale=depth, noise_sigma=0.0,
                         blur_sigma=0.0)
            deg, _ = synth.degrade(c, tp, rng)
            dev = (deg - c).abs().mean().item()
            assert dev > prev, f"deviation did not grow at depth {depth}"
            prev = dev

    def test_degradation_converges_to_the_ambient_asymptote(self):
        """As tau grows the image-formation model sends I toward A_inf;
        check the synthesiser reproduces that limit."""
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(4)
        clean = _run(48, seed=6) * 0.5 + 0.25
        base = synth.sample_task(rng, (48, 48), clean.reshape(-1, 3).mean(0))
        c = clean.permute(0, 3, 1, 2)
        deep = replace(base, depth_scale=60.0, local_tau=0.0,
                       noise_sigma=0.0, blur_sigma=0.0)
        deg, _ = synth.degrade(c, deep, rng)
        a_inf = torch.tensor(base.ambient).view(1, 3, 1, 1)
        # tau is clamped to 10, so t = exp(-10) ~ 4.5e-5, not exactly 0.
        expected = a_inf * (1.0 - float(torch.exp(torch.tensor(-10.0))))
        assert (deg.mean(dim=(2, 3)) - expected.flatten()).abs().max() < 5e-3

    def test_the_blur_path_runs(self):
        """The separable Gaussian used to crash on its second pass: the
        horizontal kernel was expanded into a vertical-shaped slot.  This
        pins it, and covers every blur strength in the configured range."""
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(11)
        clean = _run(48, seed=12)
        tp = synth.sample_task(rng, (48, 48), clean.reshape(-1, 3).mean(0))
        for sigma in (0.0, 0.05, 0.4, 1.2):
            tp = replace(tp, blur_sigma=sigma)
            deg, ref = synth.degrade(clean.permute(0, 3, 1, 2), tp, rng)
            assert deg.shape == clean.permute(0, 3, 1, 2).shape
            assert torch.isfinite(deg).all()
            assert (deg >= 0).all() and (deg <= 1.0).all()
            assert (ref >= 0).all() and (ref <= 1.0).all()

    def test_blur_is_gaussian_and_separable(self):
        """A separable Gaussian blur must match the dense 2-D kernel, up to
        the border treatment.  Tested on the interior, away from the padded
        border where the two differ by construction."""
        synth = SyntheticDegradation(cfg)
        import torch.nn.functional as F
        rng = np.random.default_rng(13)
        clean = torch.zeros(1, 3, 33, 33)
        clean[0, 1, 16, 16] = 1.0        # impulse in ONE channel only
        sigma = 1.3
        tp = synth.sample_task(rng, (33, 33), clean.reshape(-1, 3).mean(0))
        tp = replace(tp, blur_sigma=sigma, noise_sigma=0.0, depth_scale=0.0,
                     local_tau=0.0, color_gain=(1, 1, 1), color_bias=(0, 0, 0))
        deg, _ = synth.degrade(clean, tp, rng)
        k = int(2 * round(3 * sigma) + 1)
        g = torch.arange(k) - k // 2
        g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).float()
        # Per-channel dense kernel (groups=3), the operation two separable
        # passes are supposed to compose into.
        dense = torch.outer(g, g)[None].expand(3, 1, k, k).contiguous()
        ref = F.conv2d(clean, dense, padding=k // 2, groups=3)
        err = (deg[:, :, 8:25, 8:25] - ref[:, :, 8:25, 8:25]).abs().max()
        assert err.item() < 1e-5, f"separable blur != dense Gaussian: {err}"

    def test_tasks_sample_within_the_configured_ranges(self):
        synth = SyntheticDegradation(cfg)
        rng = np.random.default_rng(5)
        for _ in range(50):
            tp = synth.sample_task(rng, (32, 32), (0.3, 0.3, 0.3))
            lo, hi = cfg.synth_depth_range
            assert lo <= tp.depth_scale <= hi
            lo, hi = cfg.synth_noise_sigma
            assert lo <= tp.noise_sigma <= hi + 1e-9
            lo, hi = cfg.synth_blur_sigma
            assert lo <= tp.blur_sigma <= hi + 1e-9
            assert tp.water_type in physics.JERLOV_TABLE
            assert all(b > 0 for b in tp.beta)


class TestBatchSize:
    """A loss written and tested only at B=1 will broadcast by accident: a
    (B,1,1) mask against a (B,h,w,1) map is silent at B=1 and a hard crash at
    B=4.  Meta-training runs at ``meta_batch`` images, so this class exists to
    keep the loss honest at every batch size the pipeline actually uses."""

    @pytest.mark.parametrize("b", [1, 2, 4, 7])
    def test_terms_and_total_are_finite_at_any_batch_size(self, b):
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        g = torch.Generator().manual_seed(42)
        x = torch.rand(b, 3, 40, 44, generator=g)
        for stride in (1, 2):
            sub = model(x, step=2, coord_stride=stride)
            bd = loss_fn(strided_image(x, stride), sub, 2, 8)
            for k, v in bd.terms.items():
                assert np.isfinite(v), \
                    f"B={b} stride={stride}: term {k}={v}"
            assert torch.isfinite(bd.total).item()

    @pytest.mark.parametrize("b", [2, 4])
    def test_backward_is_finite_at_batch_size(self, b):
        model = PhysINR(cfg)
        loss_fn = PhysINRLoss(cfg)
        x = torch.rand(b, 3, 32, 36)
        out = model(x, step=1)
        bd = loss_fn(x.permute(0, 2, 3, 1), out, 1, 6)
        bd.total.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()
                           if p.grad is not None])
        assert torch.isfinite(grads).all()
        assert grads.abs().max().item() > 0.0


class TestSSIM:
    """The structural-similarity term.

    This had no coverage at all, and the denominator's variance term was
    written as a covariance: ``mxx + myy - 2*mxy`` evaluates to zero for
    identical inputs, which collapsed the denominator to the small constant
    ``c2`` and returned SSIM ~186 on a plain image instead of 1. The outer
    meta-loss uses ``1 - SSIM`` and therefore ran strongly negative for a
    whole meta-training run before the falling loss curve exposed it.
    """

    def test_identical_inputs_are_perfectly_similar(self):
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        x = torch.rand(1, 48, 64)
        assert _ssim(x, x).item() == pytest.approx(1.0, abs=1e-6)

    def test_the_range_is_bounded(self):
        """SSIM is bounded in [-1, 1]; anything else is an algebra fault."""
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        rng = np.random.default_rng(0)
        for _ in range(30):
            a = torch.from_numpy(rng.random((1, 48, 64))).float()
            b = torch.from_numpy(rng.random((1, 48, 64))).float()
            s = _ssim(a, b).item()
            assert -1.0 - 1e-4 <= s <= 1.0 + 1e-4, f"SSIM {s} out of range"

    def test_mild_distortion_stays_near_one(self):
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        x = torch.rand(1, 48, 64)
        noisy = x + torch.randn(1, 48, 64) * 0.02
        s = _ssim(x, noisy).item()
        assert 0.9 < s < 1.0

    def test_unrelated_inputs_score_low(self):
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        a = torch.rand(1, 48, 64)
        b = torch.rand(1, 48, 64)
        assert _ssim(a, b).item() < 0.5

    def test_a_constant_image_pair_is_defined(self):
        """A degenerate input must not produce NaN or inf."""
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        a = torch.full((1, 48, 64), 0.5)
        assert _ssim(a, a).item() == pytest.approx(1.0, abs=1e-6)

    def test_the_loss_term_is_non_negative(self):
        """``1 - SSIM`` is used as a loss and must never go negative."""
        from hydro_bench.methods.deep.physinr_zero.losses import _ssim

        x = torch.rand(1, 48, 64)
        noisy = x + torch.randn(1, 48, 64) * 0.05
        assert (1.0 - _ssim(x, noisy)).item() >= -1e-6


class TestColourCast:
    """The three regularisers the refactor adds, plus the ambient prior.

    Each is tested against the failure mode it exists to close: a green cast
    must be penalised, a neutral image must not be, and a physically reversed
    attenuation ordering must be caught.
    """

    def test_lab_signed_mode_penalises_green_but_not_red(self):
        """The default mode targets the green/blueward half of each opponent
        axis, which is the cast this refactor is for; a warm cast is left
        alone."""
        fn = PhysINRLoss(cfg)
        green = torch.zeros(1, 32, 32, 3)
        green[..., 0] = green[..., 2] = 0.15
        green[..., 1] = 0.60
        red = torch.zeros(1, 32, 32, 3)
        red[..., 1] = red[..., 2] = 0.15
        red[..., 0] = 0.60
        lab_g = physics.rgb_to_lab(green)
        lab_r = physics.rgb_to_lab(red)
        assert fn._lab_cast_loss(lab_g) > 0.5
        assert fn._lab_cast_loss(lab_r) == pytest.approx(0.0, abs=1e-6)

    def test_lab_symmetric_mode_penalises_every_cast(self):
        """The mode the design spec states verbatim: |mu_a| + |mu_b|, which
        pulls a cast in any direction toward neutral."""
        sym = PhysINRLoss(replace(cfg, lab_mode="symmetric"))
        for c in (0, 1, 2):                     # a red, green and blue cast
            x = torch.full((1, 32, 32, 3), 0.15)
            x[..., c] = 0.60
            assert sym._lab_cast_loss(physics.rgb_to_lab(x)) > 0.5
        # The two modes agree only on the chroma axis the cast actually lies
        # on: a green cast is a*<0 with a *yellowward* b*>0 (sRGB green is not
        # cyan), so the signed mode counts the a* term and spares the b* one.
        green = torch.full((1, 32, 32, 3), 0.15)
        green[..., 1] = 0.60
        lab_g = physics.rgb_to_lab(green)
        signed = PhysINRLoss(cfg)
        a_star = float(lab_g[..., 1].mean())
        assert signed._lab_cast_loss(lab_g) == pytest.approx(abs(min(a_star, 0.0)))
        assert signed._lab_cast_loss(lab_g) < sym._lab_cast_loss(lab_g)
        # A blue cast is negative on b* and, like sRGB green, positive on the
        # other axis, so again the signed mode counts exactly one term.
        blue = torch.full((1, 32, 32, 3), 0.15)
        blue[..., 2] = 0.60
        lab_b = physics.rgb_to_lab(blue)
        b_star = float(lab_b[..., 2].mean())
        assert signed._lab_cast_loss(lab_b) == pytest.approx(abs(min(b_star, 0.0)))
        assert signed._lab_cast_loss(lab_b) < sym._lab_cast_loss(lab_b)

    def test_a_neutral_image_has_no_lab_cast(self):
        fn = PhysINRLoss(cfg)
        gray = torch.full((1, 32, 32, 3), 0.5)
        assert fn._lab_cast_loss(physics.rgb_to_lab(gray)) == \
            pytest.approx(0.0, abs=5e-3)

    def test_a_bad_lab_mode_raises(self):
        fn = PhysINRLoss(replace(cfg, lab_mode="bogus"))
        with pytest.raises(ValueError, match="lab_mode"):
            fn._lab_cast_loss(physics.rgb_to_lab(_run(16)))

    def test_color_constancy_penalises_channel_imbalance(self):
        fn = PhysINRLoss(cfg)
        unbalanced = torch.zeros(1, 32, 32, 3)
        unbalanced[..., 1] = 0.7
        gray = torch.full((1, 32, 32, 3), 0.4)
        assert fn._color_constancy_loss(unbalanced) > 0.1
        assert fn._color_constancy_loss(gray) == pytest.approx(0.0, abs=1e-6)

    def test_color_constancy_matches_the_pair_formula(self):
        """The spec's formula: the sum over (R,G), (R,B), (G,B) of
        |mu(I^p) - mu(I^q)|, per image."""
        fn = PhysINRLoss(cfg)
        x = _run(24, seed=4)
        mu = x.mean(dim=(1, 2))[0]
        expected = (abs(mu[0] - mu[1]) + abs(mu[0] - mu[2])
                    + abs(mu[1] - mu[2])).item()
        assert fn._color_constancy_loss(x).item() == \
            pytest.approx(expected, abs=1e-6)

    def test_bound_loss_penalises_a_reversed_ordering(self):
        """With the universal ordering, tau_g > tau_r is a violation."""
        fn = PhysINRLoss(replace(cfg, bound_margin_from_table=False))
        bad = torch.zeros(1, 8, 8, 3)
        bad[..., 0] = 0.2
        bad[..., 1] = 1.5
        bad[..., 2] = 1.0
        assert fn._attenuation_bound_loss(_outputs(bad)) > 0.0
        ok = torch.zeros(1, 8, 8, 3)
        ok[..., 0] = 1.5
        ok[..., 1] = 1.0
        ok[..., 2] = 0.5
        assert fn._attenuation_bound_loss(_outputs(ok)) == \
            pytest.approx(0.0, abs=1e-9)

    def test_bound_margins_come_from_the_assigned_water_type(self):
        """Jerlov type 9 (coastal) has beta_g > beta_r, so with table margins
        the prior *requires* green to attenuate more: a clear-water ordering is
        then the violation.  A universal margin would get this backwards."""
        fn = PhysINRLoss(cfg)                    # bound_margin_from_table = True
        tj = torch.tensor([[0.6256, 0.6660, 1.5260]])     # Jerlov type 9
        coastal = torch.zeros(1, 8, 8, 3)
        coastal[..., 0] = 0.3
        coastal[..., 1] = 0.5
        coastal[..., 2] = 0.9
        clearlike = torch.zeros(1, 8, 8, 3)
        clearlike[..., 0] = 1.5
        clearlike[..., 1] = 0.5
        clearlike[..., 2] = 0.3
        assert fn._attenuation_bound_loss(_outputs(coastal, tj)) < \
            fn._attenuation_bound_loss(_outputs(clearlike, tj))

    def test_ambient_prior_is_the_deviation_from_the_estimate(self):
        fn = PhysINRLoss(cfg)
        out = _outputs(torch.zeros(1, 8, 8, 3))
        out.ambient.fill_(0.8)
        out.ambient_estimate[:, :] = 0.2
        assert fn._ambient_prior_loss(out) == pytest.approx(0.6, abs=1e-6)
        out.ambient_estimate[:, :] = 0.8
        assert fn._ambient_prior_loss(out) == pytest.approx(0.0, abs=1e-6)

    def test_the_new_terms_are_finite_and_differentiable(self):
        model = PhysINR(cfg)
        fn = PhysINRLoss(cfg)
        img = _run(32)
        out = model(img.permute(0, 3, 1, 2), step=2)
        bd = fn(img, out, 2, 8)
        for k in ("lab_cast", "cc", "bound_order", "ambient_prior"):
            assert np.isfinite(bd.terms[k]), f"term {k} is {bd.terms[k]}"
        bd.total.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()
                           if p.grad is not None])
        assert torch.isfinite(grads).all()
        assert grads.abs().max().item() > 0.0

    def test_legacy_weights_take_the_new_groups_out_entirely(self):
        """Zeroing w_cast / w_bound must remove both sub-terms of each group,
        so the ablation is a group ablation and not a partial one."""
        legacy = PhysINRConfig.legacy()
        assert legacy.w_cast == 0.0 and legacy.w_bound == 0.0
        fn = PhysINRLoss(legacy)
        img = _run(32)
        out = PhysINR(legacy)(img.permute(0, 3, 1, 2), step=0)
        bd = fn(img, out, 0, 8)
        for k in ("lab_cast", "cc", "bound_order", "ambient_prior"):
            assert bd.terms[k] >= 0.0          # computed, but weighted by zero
        # The total must not depend on the four new terms at all.
        w = {g: 0.0 if g in ("cast", "bound") else bd.weights[g]
             for g in fn.GROUPS}
        legacy_only = fn(img, out, 0, 8, weights=w).total
        assert torch.allclose(legacy_only, bd.total, atol=1e-9)


class TestAdaptation:
    def test_enhance_preserves_geometry_and_range(self):
        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter
        adapter = TestTimeAdapter(replace(cfg, adapt_steps=2, device="cpu"),
                                  PhysINR(cfg))
        img = (np.random.default_rng(0).random((48, 56, 3))
               * 0.5 + 0.2).astype(np.float32)
        out, _ = adapter.adapt(img)
        assert out.shape == img.shape
        assert (out >= 0).all() and (out <= 1.0).all()
        assert np.isfinite(out).all()

    def test_parameters_reset_between_images(self):
        """The per-image deepcopy is what makes the method zero-shot: the
        second image must start from the same theta_0 as the first."""
        import copy
        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter
        c = replace(cfg, adapt_steps=3, device="cpu")
        adapter = TestTimeAdapter(c, PhysINR(c))
        before = copy.deepcopy(adapter.model.state_dict())
        img = (np.random.default_rng(1).random((32, 32, 3))).astype(np.float32)
        adapter.adapt(img)
        adapter.adapt(img)
        after = adapter.model.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), \
                f"parameter '{k}' leaked between images"

    def test_trace_records_every_step(self):
        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter
        n = 4
        c = replace(cfg, adapt_steps=n, device="cpu")
        adapter = TestTimeAdapter(c, PhysINR(c))
        img = (np.random.default_rng(2).random((32, 32, 3))).astype(np.float32)
        out, trace = adapter.adapt(img, return_trace=True)
        assert trace is not None
        assert len(trace.losses) == n
        assert len(trace.mean_tau) == n
        assert all(0.0 <= t <= cfg.tau_max for t in trace.mean_tau)
        assert trace.final_loss == trace.losses[-1]

    def test_step_override_is_used(self):
        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter
        c = replace(cfg, adapt_steps=20, device="cpu")
        adapter = TestTimeAdapter(c, PhysINR(c))
        img = (np.random.default_rng(3).random((32, 32, 3))).astype(np.float32)
        _, trace = adapter.adapt(img, steps=3, return_trace=True)
        assert trace.steps == 3 and len(trace.losses) == 3
