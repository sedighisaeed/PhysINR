"""Tests for the network: parameter budget, bounds, and coordinate sampling.

The parameter budget and the physical bounds are *design invariants* stated
by the methodology, so a test is the right way to keep them honest: a future
edit that quietly doubles ``mlp_hidden`` would break §22's efficiency claim
without changing a single number the reader could see.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from dataclasses import replace

from hydro_bench.methods.deep.physinr_zero import physics
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR, strided_image


cfg = PhysINRConfig()


def _model(**over):
    return PhysINR(replace(cfg, **over))


class TestArchitecture:
    def test_parameter_budget_is_respected(self):
        model = _model()
        assert model.count_parameters() < cfg.param_budget
        # And the budget is actively enforced.
        with pytest.raises(ValueError, match="exceeding the budget"):
            _model(mlp_hidden=4096)

    def test_parameter_count_is_reproducible(self):
        # A regression test on the headline efficiency number of the paper.
        # 9,813 is the published architecture (type_head_features="none");
        # the content-dependent one swaps its Linear(1, K) head weight for a
        # Linear(41, K) and the standalone depth logit for a Linear(41, 1).
        assert _model(type_head_features="none").count_parameters() == 9813
        # v3 adds the level head, another Linear(41, 1) -> 42 parameters.
        assert _model().count_parameters() == 9813 - 10 - 1 + 410 + 42 + 42

    def test_the_type_head_rejects_an_unknown_feature_mode(self):
        with pytest.raises(ValueError, match="type_head_features"):
            _model(type_head_features="vibes")

    def test_feature_extractor_downsamples_by_at_most_four(self):
        m = _model()
        x = torch.rand(1, 3, 64, 64)
        out = m.feature_extractor(x)
        assert out.shape[-2] >= 64 // 4 and out.shape[-1] >= 64 // 4

    def test_two_learning_rate_levels_partition_every_parameter(self):
        """§19: the field parameters and the extractor get separate rates, and
        no parameter may be unassigned."""
        m = _model()
        groups = m.parameter_groups()
        assert len(groups) == 2
        assert groups[0]["lr"] == cfg.lr_field
        assert groups[1]["lr"] == cfg.lr_extractor
        total = sum(p.numel() for g in groups for p in g["params"])
        assert total == m.count_parameters()
        assert groups[0]["lr"] == pytest.approx(10.0 * groups[1]["lr"])

    def test_extractor_rate_is_a_tenth_of_the_field_rate(self):
        m = _model()
        g0, g1 = m.parameter_groups()
        assert g1["lr"] == pytest.approx(g0["lr"] / 10.0)


class TestBoundsAndShapes:
    def test_all_outputs_are_in_range(self):
        m = _model()
        x = torch.rand(1, 3, 48, 56)
        o = m(x, step=0)
        assert (o.tau >= 0).all() and (o.tau <= cfg.tau_max).all()
        assert (o.transmission > 0).all() and (o.transmission <= 1.0 + 1e-6).all()
        lo = cfg.ambient_min - cfg.kappa_ambient - 1e-6
        hi = cfg.ambient_max + cfg.kappa_ambient + 1e-6
        assert (o.ambient >= lo).all() and (o.ambient <= hi).all()
        assert (o.alpha.abs() <= cfg.alpha_max + 1e-6).all()
        assert (o.restored >= 0).all() and (o.restored <= 1.0).all()
        assert (o.regraded >= 0).all() and (o.regraded <= 1.0).all()

    def test_water_weights_are_a_distribution(self):
        m = _model()
        o = m(torch.rand(1, 3, 32, 32), step=0)
        assert o.water_weights.shape == (1, cfg.num_water_types)
        assert torch.allclose(o.water_weights.sum(-1),
                              torch.ones(1), atol=1e-5)
        assert (o.water_weights >= 0).all()

    def test_tau_jerlov_is_the_beta_depth_product(self):
        """The Jerlov component is a (K,3) attenuation matrix times one scalar
        effective depth -- no spatial freedom in the prior itself.  The depth
        scalar may be a per-image output of the depth head, but it must be a
        single scalar per image, shared by all three channels."""
        from hydro_bench.methods.deep.physinr_zero import physics
        m = _model()
        o = m(torch.rand(1, 3, 32, 32), step=0)
        beta = torch.tensor(physics.jerlov_beta_matrix(), dtype=torch.float32)
        ratio = o.tau_jerlov / (o.water_weights @ beta + 1e-8)   # (B, 3)
        # One scalar d per image: all three channel ratios agree.
        assert torch.allclose(ratio, ratio[:, :1], atol=1e-4)
        assert bool((ratio > 0).all())

    def test_tau_delta_is_bounded_by_delta_max(self):
        m = _model()
        o = m(torch.rand(1, 3, 32, 32), step=0)
        assert o.tau_delta.abs().max().item() <= cfg.delta_max + 1e-6

    def test_batch_of_images(self):
        m = _model()
        o = m(torch.rand(3, 3, 40, 40), step=0)
        assert o.restored.shape == (3, 40, 40, 3)
        assert o.tau.shape == (3, 40, 40, 3)

    def test_bad_input_shape_raises(self):
        m = _model()
        with pytest.raises(ValueError, match=r"\(B,3,H,W\)"):
            m(torch.rand(3, 48, 48), step=0)
        with pytest.raises(ValueError, match=r"\(B,3,H,W\)"):
            m(torch.rand(1, 4, 48, 48), step=0)

    def test_bad_stride_raises(self):
        m = _model()
        with pytest.raises(ValueError, match="coord_stride"):
            m(torch.rand(1, 3, 32, 32), step=0, coord_stride=0)


class TestCoordinateSampling:
    """The stride must sample the same field, not define a new one."""

    def test_strided_forward_matches_on_shared_pixels(self):
        m = _model()
        x = torch.rand(1, 3, 64, 64)
        full = m(x, step=0)
        stride = 4
        sub = m(x, step=0, coord_stride=stride)
        # Per-pixel quantities are indexed from the full-grid feature field,
        # so they agree on the shared pixels up to the two global poolings.
        err = (full.tau[:, ::stride, ::stride] - sub.tau).abs().max().item()
        assert err < 2e-3

    def test_strided_forward_covers_the_image(self):
        m = _model()
        x = torch.rand(1, 3, 63, 67)
        sub = m(x, step=0, coord_stride=2)
        assert sub.tau.shape == (1, 32, 34, 3)

    def test_features_are_evaluated_not_resized(self):
        """The stride used to fail this test: resampling the feature map to a
        smaller grid gave a *different* f(x).  Now the field is built on the
        full grid and indexed, so it is a genuine function of position."""
        m = _model()
        x = torch.rand(1, 3, 64, 64)
        full = m(x, step=0)
        sub = m(x, step=0, coord_stride=2)
        # Fourier features are pure functions of the coordinates; with a
        # consistent feature field the shared-pixel outputs must agree far
        # better than the resampling artefact did (~1e-1).
        err = (full.transmission[:, ::2, ::2] - sub.transmission).abs().max()
        assert err.item() < 5e-3

    def test_strided_image_helper_lines_up_with_the_field(self):
        x = torch.rand(2, 3, 50, 58)
        for s in (1, 2, 3):
            img = strided_image(x, s)
            assert img.shape == (2, -(-50 // s), -(-58 // s), 3)
            # And equals a direct slice of the channel-last image.
            ref = x.permute(0, 2, 3, 1)[:, ::s, ::s]
            assert torch.equal(img, ref)


class TestFourierRamp:
    def test_low_octaves_are_active_first(self):
        m = _model()
        coords = torch.rand(1, 100, 2)
        early = m._fourier(coords, step=0)
        late = m._fourier(coords, step=64)
        # Early on, the highest-frequency octave is nearly off.
        hi = slice(-4, None)
        assert early[..., hi].abs().max() < late[..., hi].abs().max()
        # Late, every octave is fully on (gate = 1).
        assert torch.allclose(late, m._fourier(coords, step=10_000))

    def test_fourier_dimension_is_two_per_level_per_axis(self):
        m = _model()
        assert m.fourier_dim == 2 * 2 * cfg.fourier_levels
        out = m._fourier(torch.rand(1, 7, 2), step=0)
        assert out.shape == (1, 7, m.fourier_dim)


class TestColourCastRefactor:
    """The colour-cast interventions: adaptive background light and the YIQ
    alignment stage.  Neither adds parameters, which is what keeps the §4/§22
    efficiency claim and the published meta-initialisation intact."""

    def test_the_refactor_adds_no_parameters(self):
        assert _model().count_parameters() == _model(yiq_align=False,
                                                     ambient_adaptive=False
                                                     ).count_parameters()

    def test_the_type_head_reads_the_image(self):
        """The collapse fix, as a property: the type assignment must be a
        function of the image.  Two inputs with opposite attenuation
        signatures (green-dominant vs blue-dominant) must produce clearly
        different assignments, and the assignment must be differentiable to
        the head's weights through the content path."""
        torch.manual_seed(7)
        m = _model()
        green = torch.zeros(1, 3, 48, 56)
        green[:, 1] = 0.6
        green[:, 0] = green[:, 2] = 0.15
        blue = torch.zeros(1, 3, 48, 56)
        blue[:, 2] = 0.6
        blue[:, 0] = blue[:, 1] = 0.15
        w_green = m(green, step=0).water_weights
        w_blue = m(blue, step=0).water_weights
        assert (w_green - w_blue).abs().max() > 1e-3, \
            "water_weights did not respond to the image content"
        # The content path carries gradient: the head must be reachable from
        # the image statistics, not only from the field's r channel.
        loss = -(w_green * torch.arange(cfg.num_water_types)).sum()
        loss.backward()
        assert m.jerlov_head.weight.grad is not None
        assert m.jerlov_head.weight.grad.abs().sum() > 0.0

    def test_the_legacy_head_keeps_the_published_bottleneck(self):
        """type_head_features='none' is the published architecture: Linear(1,
        K) on the pooled water logit, so the released checkpoint loads."""
        m = _model(type_head_features="none")
        assert m.jerlov_head.weight.shape == (cfg.num_water_types, 1)
        o = m(torch.rand(1, 3, 32, 32), step=0)
        assert o.water_weights.shape == (1, cfg.num_water_types)

    def test_ambient_estimate_shape_and_range(self):
        m = _model()
        o = m(torch.rand(1, 3, 48, 56), step=0)
        assert o.ambient_estimate.shape == (1, 3)
        assert bool((o.ambient_estimate >= 0).all()
                    and (o.ambient_estimate <= 1).all())

    def test_adaptive_bracket_bounds_the_ambient_field(self):
        """The learned field is sigmoid-mapped into a window centred on the
        estimate and clipped to the physical range, so its per-channel mean
        cannot wander outside that window (plus the tanh offset of §7)."""
        m = _model()
        x = torch.rand(1, 3, 48, 56)
        o = m(x, step=0)
        lo = (o.ambient_estimate - cfg.ambient_window).clamp(cfg.ambient_min,
                                                             cfg.ambient_max)
        hi = (o.ambient_estimate + cfg.ambient_window).clamp(cfg.ambient_min,
                                                             cfg.ambient_max)
        mu = o.ambient.mean(dim=(1, 2))                  # (B, 3)
        slack = cfg.kappa_ambient + 1e-5
        assert bool((mu >= lo - slack).all()), f"{mu} below the bracket {lo}"
        assert bool((mu <= hi + slack).all()), f"{mu} above the bracket {hi}"

    def test_the_alignment_reduces_the_cast_of_the_output(self):
        """Same weights, same predicted physics; the only difference between
        the two forwards is the chroma alignment, and it must reduce the
        output's chroma cast."""
        torch.manual_seed(0)
        x = torch.zeros(1, 3, 64, 64)
        x[:, 1] = 0.62
        x[:, 0] = x[:, 2] = 0.15
        # Structure for the extractor to lock onto; a flat image is not what
        # the field is adapted to.
        x = (x + 0.05 * torch.rand(1, 3, 64, 64)).clamp(0.0, 1.0)

        m_on = _model()
        m_off = _model(yiq_align=False)
        m_off.load_state_dict(m_on.state_dict())         # identical weights
        o_on = m_on(x, step=12)
        o_off = m_off(x, step=12)

        # The physics is untouched by the alignment.
        assert torch.allclose(o_on.tau, o_off.tau, atol=1e-6)
        assert torch.allclose(o_on.ambient, o_off.ambient, atol=1e-6)
        # The cast shrank.
        def cast(lab):
            return (lab[..., 1].mean() ** 2 + lab[..., 2].mean() ** 2).sqrt()

        cast_on = cast(physics.rgb_to_lab(o_on.restored))
        cast_off = cast(physics.rgb_to_lab(o_off.restored))
        assert cast_on < cast_off, \
            f"alignment did not reduce the cast: {cast_on:.3f} vs {cast_off:.3f}"

    def test_the_alignment_moves_a_green_image_toward_neutral(self):
        """The headline property: a green input must not stay as green."""
        torch.manual_seed(3)
        x = torch.zeros(1, 3, 64, 64)
        x[:, 1] = 0.62
        x[:, 0] = x[:, 2] = 0.15
        x = (x + 0.05 * torch.rand(1, 3, 64, 64)).clamp(0.0, 1.0)
        o = _model()(x, step=12)
        mu_a_in = float(physics.rgb_to_lab(x.permute(0, 2, 3, 1))[..., 1].mean())
        mu_a_out = float(physics.rgb_to_lab(o.restored.detach())[..., 1].mean())
        # a* rises toward zero: less green.
        assert mu_a_out > mu_a_in, f"a* did not rise: {mu_a_in:.2f} -> {mu_a_out:.2f}"

    def test_a_strided_forward_is_the_same_field_for_the_physics(self):
        """The alignment gains are global statistics computed on the
        full-resolution frame, so the physical fields are stride-independent
        and the stride-consistency property of the model survives."""
        m = _model()
        x = torch.rand(1, 3, 64, 64)
        full = m(x, step=0)
        sub = m(x, step=0, coord_stride=2)
        err = (full.tau[:, ::2, ::2] - sub.tau).abs().max().item()
        assert err < 2e-3


class TestLegacyConfig:
    """``legacy()`` is the definition of "before the refactor".  It must differ
    from the default *only* on the intervention fields -- anything else would
    make a before/after comparison silently measure something else too."""

    def test_legacy_turns_off_every_intervention(self):
        c = PhysINRConfig.legacy()
        assert not c.use_spectral_beta
        assert not c.ambient_adaptive
        assert not c.yiq_align
        assert c.w_cast == 0.0
        assert c.w_bound == 0.0
        assert c.type_head_features == "none"
        assert c.w_type_sup == 0.0
        assert c.w_tau_sup == 0.0
        # And the synthesis interventions: the published checkpoint was
        # meta-trained on the uniform task family.
        assert c.synth_coastal_weight == 0.0
        assert c.synth_cdom_a440 == (0.0, 0.0)
        assert not c.synth_ambient_tied
        # And the v2 prior interventions: no transmission floor, no
        # data-driven level -- the published forward, exactly.
        assert c.t_min == 0.0
        assert c.tau_data_weight == 0.0

    def test_legacy_differs_from_the_default_only_on_the_interventions(self):
        interventions = {"use_spectral_beta", "ambient_adaptive", "yiq_align",
                        "display_norm",
                        "w_cast", "w_bound", "type_head_features",
                        "w_type_sup", "w_tau_sup", "synth_coastal_weight",
                        "synth_cdom_a440", "synth_ambient_tied",
                        "t_min", "tau_data_weight",
                        "tau_jerlov_centered", "level_head", "w_level_sup",
                        "synth_depth_range"}
        default, legacy = PhysINRConfig(), PhysINRConfig.legacy()
        for f in dataclasses.fields(default):
            if f.name in interventions:
                continue
            assert getattr(default, f.name) == getattr(legacy, f.name), \
                f"legacy changed {f.name}"


class TestDataDrivenPrior:
    """The v2 prior: level from the image, Jerlov table for the ratio."""

    def test_the_prior_tracks_the_image_contrast(self):
        """Two images with the same means but different per-channel ranges
        must get different priors: the level is measured, not looked up."""
        torch.manual_seed(1)
        base = torch.rand(1, 3, 48, 56) * 0.4 + 0.3
        compressed = base.clone()
        compressed[:, 2] = 0.45               # blue flattened to a plane
        m = _model()
        with torch.no_grad():
            t_full = physics.estimate_transmission_level(
                base.permute(0, 2, 3, 1))
            t_comp = physics.estimate_transmission_level(
                compressed.permute(0, 2, 3, 1))
        assert float(t_comp[0, 2]) < float(t_full[0, 2]) - 0.3
        o_full = m(base, step=0)
        o_comp = m(compressed, step=0)
        # The effective prior (tau minus the field's bounded deviation) must
        # differ across the two inputs by more than the field's own freedom.
        d_full = (o_full.tau - o_full.tau_delta).mean(dim=(1, 2, 3))
        d_comp = (o_comp.tau - o_comp.tau_delta).mean(dim=(1, 2, 3))
        assert float((d_full - d_comp).abs().max()) > 0.1

    def test_the_transmission_floor_reaches_the_inverse(self):
        """The model must pass cfg.t_min down to the physical inverse: the
        floor is what bounds the 1/t amplification when the prior (or the
        field) over-estimates the optical thickness."""
        import unittest.mock

        m = _model(t_min=0.25)
        seen = []
        real = physics.physical_inverse
        with unittest.mock.patch.object(
                physics, "physical_inverse",
                lambda img, amb, t, eps=1e-4, t_min=0.0:
                    seen.append(float(t.min()))
                    or real(img, amb, t, eps, t_min)):
            m(torch.rand(1, 3, 32, 32), step=0)
        assert seen and min(seen) >= 0.25 - 1e-6

    def test_the_floor_is_absent_in_legacy(self):
        from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig as _C
        m = PhysINR(_C.legacy())
        assert m.cfg.t_min == 0.0
        assert m.cfg.tau_data_weight == 0.0


class TestLevelHead:
    """The v3 level head: the range estimator normalises the best channel to
    t = 1, but that channel is itself attenuated in real water; the head
    supplies the missing absolute scale."""

    def test_the_head_exists_and_starts_at_the_v2_behaviour(self):
        m = _model()
        assert m.level_head is not None
        x = torch.rand(2, 3, 40, 44)
        with torch.no_grad():
            out = m(x, step=0)
        # Zero-init sigmoid -> 0.5 -> tau = -log(0.5) = 0.693 for every image,
        # so training starts from a known, neutral level.
        assert out.tau_level.shape == (2,)
        assert torch.allclose(out.tau_level, torch.full((2,), 0.6931), atol=2e-3)

    def test_the_head_moves_the_prior_without_touching_the_ratio(self):
        """Sweeping the level must scale all channels' tau by the same amount:
        the spectral ratio the estimator supplies is preserved."""
        torch.manual_seed(2)
        x = torch.rand(1, 3, 40, 44) * 0.5 + 0.2
        m = _model()
        with torch.no_grad():
            m.level_head.bias.fill_(0.0)      # sigmoid(0) = 0.5
            a = (m(x, step=0).tau - m(x, step=0).tau_delta).mean(dim=(1, 2, 3))
            m.level_head.bias.fill_(-3.0)     # sigmoid(-3) ~ 0.047 -> higher tau
            b = (m(x, step=0).tau - m(x, step=0).tau_delta).mean(dim=(1, 2, 3))
        # The whole prior shifts up, and the inter-channel SPACING is
        # preserved: the level moves the DC, not the shape.
        assert float((b - a)) > 0.2
        # tau_jerlov is centered (its DC removed), so the spacing difference
        # comes only from the level acting on the data term.
        with torch.no_grad():
            m.level_head.bias.fill_(0.0)
            ta = m(x, step=0).tau
            m.level_head.bias.fill_(-3.0)
            tb = m(x, step=0).tau
        gap_a = (ta[..., 0] - ta[..., 2]).mean()
        gap_b = (tb[..., 0] - tb[..., 2]).mean()
        # The t_min floor saturates the smallest channel at extreme levels, so
        # require the spacing to be close but not exact.
        assert abs(float(gap_b - gap_a)) < 0.35 * max(1.0, abs(float(gap_a)))

    def test_legacy_has_no_level_head(self):
        from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig as _C
        m = PhysINR(_C.legacy())
        assert m.level_head is None
        out = m(torch.rand(1, 3, 40, 44), step=0)
        assert torch.allclose(out.tau_level, torch.zeros(1))