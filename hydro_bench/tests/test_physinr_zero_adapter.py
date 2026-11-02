"""Tests for the PhysINR-Zero HydroBench adapter.

The adapter is the only thing standing between the method and the benchmark's
metric pipeline, so the contract it promises is tested directly against
``BaseAlgorithm``: geometry preserved, dtype preserved, and -- the part that
matters for the paper -- the zero-shot provenance that ``run_metadata``
asserts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hydro_bench.config_defaults import method_config
from hydro_bench.core.base_algorithm import (BaseAlgorithm, discover_algorithm,
                                             registered_algorithms)
from hydro_bench.methods.deep.physinr_zero.adapter import PhysINRZero

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", params=["published_arch", "content_head"])
def checkpoint(request, tmp_path_factory):
    """A plumbing checkpoint with the *shape* of a meta-trained one.

    Parametrised over both released architectures: the published
    one-dimensional type-head bottleneck and the content-dependent head.  The
    adapter must load either, because the configuration travels inside the
    checkpoint blob.

    The tests here only exercise the plumbing, not any learned behaviour. The
    adapter refuses to build a checkpoint with ``tasks == 0`` (a placeholder
    written for exactly this kind of test), so the fixture reports a small
    nonzero task count; the numbers it produces are still meaningless and must
    never be reported. The real checkpoint comes from
    ``hydro_bench.methods.deep.physinr_zero meta-train``.
    """
    import torch

    from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
    from hydro_bench.methods.deep.physinr_zero.model import PhysINR

    d = tmp_path_factory.mktemp(f"weights_{request.param}")
    out = d / "physinr_zero_meta.pt"
    if request.param == "published_arch":
        cfg = PhysINRConfig(type_head_features="none")
    else:
        cfg = PhysINRConfig()
    model = PhysINR(cfg)
    torch.save({"model": model.state_dict(),
                "config": cfg.to_dict(),
                "param_count": model.count_parameters(),
                "tasks": 4,
                "meta_loss": 1.0}, out)
    return out


@pytest.fixture
def placeholder(tmp_path):
    """An unmeta-trained placeholder, which the adapter must refuse to build."""
    import torch

    from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
    from hydro_bench.methods.deep.physinr_zero.model import PhysINR

    out = tmp_path / "placeholder.pt"
    cfg = PhysINRConfig()
    model = PhysINR(cfg)
    torch.save({"model": model.state_dict(),
                "config": cfg.to_dict(),
                "param_count": model.count_parameters(),
                "tasks": 0,
                "meta_loss": float("nan")}, out)
    return out


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return (rng.random((48, 64, 3)) * 0.5 + 0.25).astype(np.uint8)


class TestRegistration:
    def test_registered_and_configured(self, checkpoint):
        assert "PhysINR-Zero" in registered_algorithms()
        cls = discover_algorithm("PhysINR-Zero")
        assert cls is PhysINRZero
        entry = method_config("PhysINR-Zero")
        assert entry["mode"] == "online"
        assert entry["adapter"].endswith("hydro_bench.methods.deep.physinr_zero.adapter")
        assert "checkpoint" in entry["params"]

    def test_it_is_a_base_algorithm(self):
        assert issubclass(PhysINRZero, BaseAlgorithm)
        assert PhysINRZero.name == "PhysINR-Zero"

    def test_no_declared_training_sets(self):
        """The leakage warning fires when a method is evaluated on data it was
        developed on.  PhysINR-Zero must never trip it: the whole method is
        that no real underwater image was used."""
        assert PhysINRZero.capabilities.training_sets == ()


class TestContract:
    def test_geometry_and_dtype_preserved(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        out = m.enhance(image)
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    def test_output_is_valid_uint8(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        out = m.enhance(image)
        assert out.min() >= 0 and out.max() <= 255
        assert bool(np.isfinite(out).all())

    def test_bad_shape_is_rejected(self, checkpoint):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        with pytest.raises(ValueError, match="HxWx3"):
            m.enhance(np.random.rand(48, 48).astype(np.uint8))
        with pytest.raises(ValueError, match="HxWx3"):
            m.enhance(np.random.rand(48, 48, 4).astype(np.uint8))

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PhysINRZero(checkpoint=tmp_path / "nope.pt", device="cpu")

    def test_an_unmeta_trained_checkpoint_is_refused(self, placeholder):
        """A placeholder must never be benchmarked.

        The guard is what stops a plumbing checkpoint from producing reported
        numbers: a random initialisation diverges on a real task, so its scores
        are not a result of anything.
        """
        with pytest.raises(ValueError, match="no meta-trained state"):
            PhysINRZero(checkpoint=placeholder, device="cpu")

    def test_warmup_does_not_break_the_contract(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        m.warmup(image.shape)
        assert m.enhance(image).shape == image.shape


class TestProvenance:
    def test_metadata_reports_the_checkpoint_record(self, checkpoint):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        md = m.run_metadata()
        assert md["method"] == "PhysINR-Zero"
        assert md["used_fallback"] is False
        # The provenance of the *initialisation*, not just a path.
        rec = md["checkpoint_record"]
        assert "tasks" in rec and "param_count" in rec
        assert rec["param_count"] == m._adapter.model.count_parameters()

    def test_metadata_names_the_zero_shot_regime(self, checkpoint):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        md = m.run_metadata()
        assert "zero-shot" in md["note"]
        assert "reference-free" in md["note"]

    def test_ablation_overrides_reach_the_configuration(self, checkpoint, image):
        """The benchmark can vary steps and stride per run; each must land in
        the configuration the results file describes."""
        m = PhysINRZero(checkpoint=checkpoint, device="cpu",
                        adapt_steps=5, loss_stride=2)
        m.enhance(image)
        cfg = m._adapter.cfg
        assert cfg.adapt_steps == 5
        assert cfg.loss_stride == 2

    def test_configuration_is_read_from_the_checkpoint(self, checkpoint):
        """A checkpoint is paired with the configuration that produced it, not
        with whatever happens to be the default at load time."""
        m = PhysINRZero(checkpoint=checkpoint, device="cpu")
        cfg_from_checkpoint = m.run_metadata()["checkpoint_record"]["config"]
        assert cfg_from_checkpoint["adapt_steps"] == m._adapter.cfg.adapt_steps
        assert cfg_from_checkpoint["tau_max"] == m._adapter.cfg.tau_max


class TestAblationKnobs:
    """The §22 ablations vary the objective and the physical bounds at
    benchmark time, so each knob must land in the configuration the results
    file describes."""

    @pytest.mark.parametrize("group", ["phys", "cycle", "color", "struct",
                                       "spatial", "jerlov", "prior", "cast",
                                       "bound"])
    def test_a_loss_group_can_be_dropped(self, checkpoint, image, group):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu",
                        drop_group=group)
        m.enhance(image)
        assert m._adapter.cfg.__dict__[f"w_{group}"] == 0.0
        # The other groups must be untouched -- a drop that also nudged its
        # neighbours would not be the one-at-a-time ablation the table reports.
        for other in ("phys", "cycle", "color", "struct", "spatial",
                      "jerlov", "prior", "cast", "bound"):
            if other != group:
                assert m._adapter.cfg.__dict__[f"w_{other}"] > 0.0

    @pytest.mark.parametrize("bad", ["bogus", "all", ""])
    def test_an_unknown_group_is_rejected(self, checkpoint, bad):
        with pytest.raises(ValueError, match="drop_group"):
            PhysINRZero(checkpoint=checkpoint, device="cpu", drop_group=bad)

    def test_group_names_are_case_insensitive(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu", drop_group="COLOR")
        m.enhance(image)
        assert m._adapter.cfg.w_color == 0.0

    def test_physical_bounds_can_be_varied(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu",
                        tau_max=4.0, delta_max=0.5)
        m.enhance(image)
        assert m._adapter.cfg.tau_max == 4.0
        assert m._adapter.cfg.delta_max == 0.5

    def test_all_overrides_compose(self, checkpoint, image):
        m = PhysINRZero(checkpoint=checkpoint, device="cpu",
                        adapt_steps=5, loss_stride=2, tau_max=4.0,
                        delta_max=0.5, drop_group="color")
        m.enhance(image)
        c = m._adapter.cfg
        assert (c.adapt_steps, c.loss_stride, c.tau_max, c.delta_max,
                c.w_color) == (5, 2, 4.0, 0.5, 0.0)

    def test_legacy_mode_switches_off_the_refactor(self, checkpoint, image):
        """``legacy`` runs a pre-refactoring checkpoint under the
        pre-refactoring objective and physics, so a before/after comparison
        measures the colour-cast work and not a change of defaults."""
        import torch
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob["config"]["type_head_features"] != "none":
            pytest.skip("legacy reproduces the published configuration; "
                        "it does not apply to content-head checkpoints")
        m = PhysINRZero(checkpoint=checkpoint, device="cpu", legacy=True)
        m.enhance(image)
        c = m._adapter.cfg
        assert not c.use_spectral_beta
        assert not c.ambient_adaptive
        assert not c.yiq_align
        assert c.w_cast == 0.0 and c.w_bound == 0.0
        # The type-head collapse fix is also an intervention of this series:
        # legacy keeps the published one-dimensional bottleneck and no
        # supervision, so the released checkpoint's state dict matches.
        assert c.type_head_features == "none"
        assert c.w_type_sup == 0.0

    def test_legacy_mode_is_recorded_in_the_metadata(self, checkpoint, image):
        import torch
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob["config"]["type_head_features"] != "none":
            pytest.skip("legacy reproduces the published configuration; "
                        "it does not apply to content-head checkpoints")
        m = PhysINRZero(checkpoint=checkpoint, device="cpu", legacy=True)
        m.enhance(image)
        assert m._adapter.cfg.use_spectral_beta is False
