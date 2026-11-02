"""Tests for the deep method adapters and their vendored-code loader.

Each adapter must honour the BaseAlgorithm contract regardless of whether its
official checkpoint is available:

* with weights, it runs the published model and reports ``used_fallback`` as
  false;
* without weights (or without the CUDA kernels a method needs), it runs an
  explicitly labelled analytic fallback and says so in ``run_metadata``.

An unrunnable baseline may never masquerade as the real method -- that is the
property these tests pin down.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hydro_bench.config_defaults import PACKAGE_ROOT
from hydro_bench.core.base_algorithm import (BaseAlgorithm,
                                             discover_algorithm,
                                             registered_algorithms)
from hydro_bench.methods.deep._vendored import (ENV_VAR, import_vendored,
                                                repo_dir, third_party_root)

DEEP_METHODS = ["Water-Net", "FUnIE-GAN", "UIEC2-Net", "Five-A+", "TOPAL",
                "U-Shape-Transformer", "WF-Diff", "WaterMamba"]

# Methods with a vendored repo + released checkpoint in this workspace.
WEIGHTED_METHODS = {
    "Water-Net": "waternet_exported_state_dict-daa0ee.pt",
    "FUnIE-GAN": "funie_generator.pth",
    "TOPAL": "TOPAL_pretrained.pth",
    "U-Shape-Transformer": "UST_generator_795.pth",
    "WaterMamba": "WaterMamba_pretrained.pth",
    "UIEC2-Net": "uiec2_net.pth",
    "Five-A+": "five_a_plus.pth",
}

WORKSPACE = PACKAGE_ROOT.parent
WEIGHTS_DIR = WORKSPACE / "weights"
THIRD_PARTY = WORKSPACE / "third_party"


# Smallest input side a published network accepts. The fixture is 64x64, which
# most architectures handle, but FA+Net averages its pyramid branches over
# unpadded 128-pixel windows and returns an empty tensor below that. The floor
# belongs to the published network, so the test grows the fixture rather than
# the adapter growing the image.
MIN_INPUT_SIDE = {"Five-A+": 128}


def official_weights_available(method: str) -> bool:
    """Are the repo + checkpoint for a weighted method present?"""
    repo = {"Water-Net": "waternet_pytorch", "FUnIE-GAN": "funie_gan",
            "TOPAL": "topal", "U-Shape-Transformer": "ust",
            "WaterMamba": "watermamba", "UIEC2-Net": "uiec2_net",
            "Five-A+": "five_a_plus"}[method]
    # Five A+ commits its weights to the repository itself; weights/ holds a
    # copy, and the adapter accepts either location.
    in_repo = (THIRD_PARTY / repo / "model" / "FAPlusNet-alpha-0.1.pth").is_file()
    return ((WEIGHTS_DIR / WEIGHTED_METHODS[method]).is_file() or in_repo) \
        and (THIRD_PARTY / repo).is_dir()


# ------------------------------------------------------------------ registry
class TestAdapterRegistry:
    def test_every_deep_method_is_discoverable(self):
        registry = registered_algorithms()
        for name in DEEP_METHODS:
            assert name in registry, f"{name} has no registered adapter"
            assert issubclass(registry[name], BaseAlgorithm)

    def test_discover_algorithm_returns_the_named_class(self):
        assert discover_algorithm("FUnIE-GAN").name == "FUnIE-GAN"

    def test_discover_algorithm_rejects_unknown_names(self):
        with pytest.raises(KeyError, match="unknown algorithm"):
            discover_algorithm("Definitely-Not-A-Method")


# ------------------------------------------------------------- fallback mode
class TestFallbackMode:
    """Every adapter must degrade to a labelled fallback, never to a crash."""

    @pytest.mark.parametrize("method", DEEP_METHODS)
    def test_missing_checkpoint_runs_labelled_fallback(self, method, hazy_image,
                                                       monkeypatch, tmp_path):
        # A directory that exists but holds no checkpoint: resolution fails, and
        # the adapter must fall back rather than raise.
        monkeypatch.setenv("HYDROBENCH_MODELS", str(tmp_path))
        cls = discover_algorithm(method)
        adapter = cls(checkpoint=Path("does_not_exist.pth"))
        out = adapter.enhance(hazy_image)

        assert adapter.used_fallback is True
        assert out.shape == hazy_image.shape
        assert out.dtype == hazy_image.dtype
        metadata = adapter.run_metadata()
        assert metadata["used_fallback"] is True
        # The note must flag the substitution; its exact wording varies.
        assert metadata["note"] != "official weights"
        assert "indicative only" in metadata["note"]

    @pytest.mark.parametrize("method", DEEP_METHODS)
    def test_fallback_output_is_valid_rgb(self, method, hazy_image, tmp_path,
                                          monkeypatch):
        monkeypatch.setenv("HYDROBENCH_MODELS", str(tmp_path))
        adapter = discover_algorithm(method)()
        out = adapter.enhance(hazy_image)
        assert out.shape == hazy_image.shape
        assert np.isfinite(out).all()

    def test_fallbacks_do_not_silently_mimic_the_real_method(self, tmp_path,
                                                             monkeypatch):
        monkeypatch.setenv("HYDROBENCH_MODELS", str(tmp_path))
        adapter = discover_algorithm("TOPAL")()
        metadata = adapter.run_metadata()
        assert metadata["note"] != "official weights"
        assert "fallback" in metadata["note"].lower()


# -------------------------------------------------------- official weights
class TestOfficialWeights:
    """Where the released checkpoint and codebase exist, the real model runs."""

    @pytest.mark.parametrize("method", list(WEIGHTED_METHODS))
    def test_official_checkpoint_runs_the_published_model(self, method,
                                                          hazy_image,
                                                          monkeypatch):
        if not official_weights_available(method):
            pytest.skip(f"official weights for {method} are not vendored")
        monkeypatch.setenv("HYDROBENCH_MODELS", str(WEIGHTS_DIR))
        monkeypatch.setenv(ENV_VAR, str(THIRD_PARTY))

        side = max(hazy_image.shape[0], MIN_INPUT_SIDE.get(method, 0))
        image = (hazy_image if side == hazy_image.shape[0]
                 else np.repeat(np.repeat(hazy_image, side // hazy_image.shape[0],
                                          axis=0), side // hazy_image.shape[1],
                                axis=1))

        adapter = discover_algorithm(method)(
            checkpoint=WEIGHTED_METHODS[method])
        if adapter.used_fallback:
            # WaterMamba additionally needs the CUDA-only mamba_ssm kernels.
            pytest.skip(f"{method} weights present but its stack is unrunnable")
        out = adapter.enhance(image)

        assert adapter.used_fallback is False
        assert out.shape == image.shape
        assert out.dtype == image.dtype
        assert adapter.run_metadata()["note"] == "official weights"

    def test_checkpoint_resolved_from_hydrobench_models(self, hazy_image,
                                                        monkeypatch):
        monkeypatch.setenv("HYDROBENCH_MODELS", str(WEIGHTS_DIR))
        monkeypatch.setenv(ENV_VAR, str(THIRD_PARTY))
        if not (WEIGHTS_DIR / "funie_generator.pth").is_file():
            pytest.skip("FUnIE-GAN weights are not vendored")
        adapter = discover_algorithm("FUnIE-GAN")(checkpoint="funie_generator.pth")
        assert adapter.used_fallback is False


# ------------------------------------------------------------- geometry edge
class TestGeometryHandling:
    def test_topal_pad_makes_multiples_of_sixteen(self):
        from hydro_bench.methods.deep.topal.adapter import TOPAL

        image = np.zeros((200, 250, 3), np.uint8)
        padded = TOPAL._pad(image)
        assert padded.shape[0] % 16 == 0
        assert padded.shape[1] % 16 == 0
        # The original pixels sit in the top-left corner, unmodified.
        assert np.array_equal(padded[:200, :250], image)

    def test_topal_pad_is_a_noop_for_aligned_shapes(self):
        from hydro_bench.methods.deep.topal.adapter import TOPAL

        image = np.zeros((64, 128, 3), np.uint8)
        assert TOPAL._pad(image) is image


# ------------------------------------------------------------- vendored loader
class TestVendoredLoader:
    def test_third_party_root_honours_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "vendored"))
        assert third_party_root() == tmp_path / "vendored"

    def test_third_party_root_defaults_next_to_the_package(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert third_party_root() == WORKSPACE / "third_party"

    def test_repo_dir_raises_when_the_repo_is_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="not found"):
            repo_dir("no_such_repo")

    def test_repo_dir_resolves_an_existing_repo(self, monkeypatch, tmp_path):
        (tmp_path / "demo_repo").mkdir()
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert repo_dir("demo_repo").is_dir()

    def test_import_vendored_prefers_the_env_package(self, monkeypatch):
        import types

        fake = types.ModuleType("hydro_demo_pkg.sub")
        monkeypatch.setitem(sys.modules, "hydro_demo_pkg.sub", fake)
        module = import_vendored("sub", env_package="hydro_demo_pkg",
                                 repo="no_such_repo_needed")
        assert module is fake

    def test_import_vendored_requires_a_repo_without_env_package(self):
        with pytest.raises(ValueError, match="no vendored repo name"):
            import_vendored("orphan_module")


class TestOOMFallback:
    """The accelerator-OOM escape that keeps large images scorable.

    A 3 Mpixel UIEB frame pushes TOPAL's fusion stage past the Apple GPU's
    private memory pool. The escape reruns the identical forward on CPU, which
    keeps the published weights' output exact, and then hands the accelerator
    back so the remaining (smaller) images do not slow to a crawl. The pool
    ceiling itself is lifted at startup instead -- see ``TestMPSMemoryPool``,
    which pins down why it cannot be lifted from inside the failing forward.
    """

    @staticmethod
    def _net_on(device, oom_times=1):
        import torch

        class CountedOOMNet(torch.nn.Module):
            """Raises an MPS OOM for the first ``oom_times`` calls."""

            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.oomed = 0
                self.oom_times = oom_times

            def forward(self, x, device_tag):
                assert x.device.type == device_tag
                if (x.device.type == "mps"
                        and self.oomed < self.oom_times):
                    self.oomed += 1
                    raise RuntimeError(
                        "MPS backend out of memory (simulated)")
                return self.lin(x)

        return CountedOOMNet().to(device)

    def test_oom_reruns_on_cpu_and_restores_the_accelerator(self):
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("no MPS device")
        from hydro_bench.methods.deep import _vendored

        net = self._net_on("mps", oom_times=1)
        note = {}

        def forward(device):
            tensor = torch.ones(1, 4, device=device)
            return net(tensor, device).sum().item()

        result = _vendored.forward_with_cpu_fallback(forward, (net,),
                                                     "mps", note)
        assert note["cpu_fallback_images"] == 1
        # The accelerator is handed back for the next image.
        assert next(net.parameters()).device.type == "mps"
        assert isinstance(result, float)

    def test_subsequent_images_reuse_the_accelerator(self):
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("no MPS device")
        from hydro_bench.methods.deep import _vendored

        net = self._net_on("mps", oom_times=1)

        def forward(device):
            tensor = torch.ones(1, 4, device=device)
            return net(tensor, device).sum().item()

        first = _vendored.forward_with_cpu_fallback(forward, (net,), "mps", {})
        second_note = {}
        second = _vendored.forward_with_cpu_fallback(forward, (net,), "mps",
                                                     second_note)
        assert not second_note, "the accelerator should be usable again"
        # The two backends agree to float32 precision; exact bit-equality is
        # not guaranteed between MPS and CPU reductions.
        assert first == pytest.approx(second, abs=1e-5)

    def test_unrelated_errors_are_not_swallowed(self):
        from hydro_bench.methods.deep._vendored import (
            forward_with_cpu_fallback)

        def forward(device):
            raise RuntimeError("unrelated failure")

        with pytest.raises(RuntimeError, match="unrelated failure"):
            forward_with_cpu_fallback(forward, (None,), "cpu", {})

    def test_cpu_device_raises_rather_than_looping(self):
        from hydro_bench.methods.deep._vendored import (
            forward_with_cpu_fallback)

        def forward(device):
            raise RuntimeError("out of memory")

        # Already on CPU: there is nowhere to fall back to.
        with pytest.raises(RuntimeError, match="out of memory"):
            forward_with_cpu_fallback(forward, (None,), "cpu", {})


class TestMPSMemoryPool:
    """The MPS ceiling must be lifted *before* the first allocation.

    PyTorch reads ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` once, when the MPS
    allocator is initialised. A run that hits an OOM and only then sets the
    variable is too late -- the pool's ``max allowed`` is already baked in, so
    the oversized frame is condemned to a slow CPU rerun. That is why the CLI
    lifts it at startup rather than on the failing path.
    """

    def test_prepare_sets_the_variable(self, monkeypatch):
        from hydro_bench.methods.deep import _vendored

        monkeypatch.delenv(_vendored.MPS_HIGH_WATERMARK_VAR, raising=False)
        assert _vendored.prepare_mps_memory_pool(unbounded=True) is True
        assert (os.environ[_vendored.MPS_HIGH_WATERMARK_VAR] == "0")

    def test_prepare_is_idempotent_when_already_lifted(self, monkeypatch):
        from hydro_bench.methods.deep import _vendored

        monkeypatch.setenv(_vendored.MPS_HIGH_WATERMARK_VAR, "0")
        assert _vendored.prepare_mps_memory_pool(unbounded=True) is True
        assert (os.environ[_vendored.MPS_HIGH_WATERMARK_VAR] == "0")

    def test_declining_leaves_the_default_ceiling_alone(self, monkeypatch):
        from hydro_bench.methods.deep import _vendored

        monkeypatch.delenv(_vendored.MPS_HIGH_WATERMARK_VAR, raising=False)
        assert _vendored.prepare_mps_memory_pool(unbounded=False) is False
        assert _vendored.MPS_HIGH_WATERMARK_VAR not in os.environ

    def test_the_ceiling_is_baked_in_before_the_first_allocation(self):
        """The regression this whole design exists to avoid.

        Allocating on MPS and *then* setting the variable must still cap the
        pool, which is why the lift happens at process startup. Untouched
        pages keep the probe cheap; the pool's bookkeeping counts the
        allocation regardless.
        """
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("no MPS device")

        warm = torch.empty(64, 64, device="mps")
        warm.fill_(1.0)
        torch.mps.empty_cache()

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
        blocks = []
        try:
            for _ in range(10):  # 10 GiB, past the 9.07 GiB default ceiling
                blocks.append(torch.empty(256 * 1024 * 1024, device="mps",
                                          dtype=torch.float32))
        except RuntimeError as exc:
            assert "out of memory" in str(exc).lower()
            return
        pytest.fail("the MPS ceiling was lifted after first use, which "
                    "contradicts the startup-time design")
