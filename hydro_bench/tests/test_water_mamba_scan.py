"""Tests for the pure-PyTorch selective scan that replaces mamba_ssm's kernel.

WaterMamba cannot use the CUDA ``selective_scan_fn`` on this host -- there is
no CUDA and no installable wheel -- so HydroBench substitutes the authors'
own pure-PyTorch reference, evaluated in bounded memory by a chunked parallel
scan. The substitution is only honest if it computes the same function, so
these tests pin the chunked implementation to the vendored reference across
every tensor shape the architecture actually produces.

The two cannot be bit-identical: a parallel scan sums in a different order
than a sequential one, and the CUDA kernel differs from the reference the
same way. What is asserted is agreement to float associativity, with a
tolerance set relative to the signal's own scale.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydro_bench.methods.deep.water_mamba.selective_scan_compat import (
    selective_scan_ref)
from hydro_bench.methods.deep.water_mamba.selective_scan_fast import (
    selective_scan_chunked)


def _make_case(L, D=48, N=16, B=1, variable="3d", seed=0):
    torch.manual_seed(seed)
    u = torch.rand(B, D, L) * 2 - 1
    delta = torch.rand(B, D, L)
    A = -torch.rand(D, N) * 0.5
    if variable == "const":
        Bm, C = torch.rand(D, N), torch.rand(D, N)
    elif variable == "3d":
        Bm, C = torch.rand(B, N, L) * 2 - 1, torch.rand(B, N, L) * 2 - 1
    else:  # grouped 4-D, as the VSSM core uses
        groups = 4
        Bm = torch.rand(B, groups, N, L) * 2 - 1
        C = torch.rand(B, groups, N, L) * 2 - 1
    Dv = torch.rand(D)
    z = torch.rand(B, D, L)
    bias = torch.rand(D)
    return u, delta, A, Bm, C, Dv, z, bias


class TestChunkedScanMatchesReference:
    """The chunked scan must reproduce the authors' reference recurrence."""

    @pytest.mark.parametrize("variable", ["const", "3d", "grouped"])
    def test_agrees_to_float_precision(self, variable):
        u, delta, A, B, C, D, z, bias = _make_case(3136, variable=variable)
        with torch.no_grad():
            ref = selective_scan_ref(u, delta, A, B, C, D, z=z,
                                     delta_bias=bias, delta_softplus=True)
            got = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                         delta_bias=bias, delta_softplus=True)
        assert ref.shape == got.shape == (1, 48, 3136)
        scale = ref.abs().max() + 1e-9
        assert (ref - got).abs().max() < 1e-3 * scale

    def test_shape_and_dtype_preserved(self):
        u, delta, A, B, C, D, z, bias = _make_case(777, variable="3d")
        with torch.no_grad():
            got = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                         delta_bias=bias, delta_softplus=True)
        assert got.dtype == u.dtype
        assert got.shape == u.shape

    @pytest.mark.parametrize("L", [1, 2, 63, 1023, 1024, 1025, 4100, 12345])
    def test_lengths_across_chunk_boundaries(self, L):
        """Lengths that split unevenly across chunks must still match."""
        u, delta, A, B, C, D, z, bias = _make_case(L, variable="3d")
        with torch.no_grad():
            ref = selective_scan_ref(u, delta, A, B, C, D, z=z,
                                     delta_bias=bias, delta_softplus=True)
            got = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                         delta_bias=bias, delta_softplus=True)
        scale = ref.abs().max() + 1e-9
        assert (ref - got).abs().max() < 1e-3 * scale

    def test_optional_terms_can_be_omitted(self):
        u, delta, A, B, C = _make_case(2048, variable="3d")[:5]
        with torch.no_grad():
            ref = selective_scan_ref(u, delta, A, B, C)
            got = selective_scan_chunked(u, delta, A, B, C)
        scale = ref.abs().max() + 1e-9
        assert (ref - got).abs().max() < 1e-3 * scale

    def test_batch_dimension_is_independent(self):
        """Each batch element must be scanned without cross-talk."""
        u, delta, A, B, C, D, z, bias = _make_case(512, B=3, variable="3d")
        with torch.no_grad():
            batched = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                             delta_bias=bias,
                                             delta_softplus=True)
            for i in range(3):
                single = selective_scan_chunked(
                    u[i:i + 1], delta[i:i + 1], A, B[i:i + 1], C[i:i + 1],
                    D, z=z[i:i + 1], delta_bias=bias, delta_softplus=True)
                assert torch.allclose(batched[i:i + 1], single, atol=1e-4)

    def test_chunk_size_does_not_change_the_result(self):
        """The chunk length is a memory knob, not a parameter of the math."""
        u, delta, A, B, C, D, z, bias = _make_case(5000, variable="3d")
        with torch.no_grad():
            small = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                           delta_bias=bias, delta_softplus=True,
                                           chunk=256)
            big = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                         delta_bias=bias, delta_softplus=True,
                                         chunk=4096)
        assert (small - big).abs().max() < 1e-4 * (small.abs().max() + 1e-9)

    def test_output_is_finite(self):
        u, delta, A, B, C, D, z, bias = _make_case(8192, variable="3d")
        with torch.no_grad():
            got = selective_scan_chunked(u, delta, A, B, C, D, z=z,
                                         delta_bias=bias, delta_softplus=True)
        assert torch.isfinite(got).all()


class TestParallelScanProperties:
    """The Hillis-Steele core must be a true associative scan."""

    def test_identity_when_all_steps_are_identity(self):
        """a=1, b=0 leaves any start value unchanged."""
        from hydro_bench.methods.deep.water_mamba.selective_scan_fast import (
            _parallel_scan)

        a = torch.ones(1, 4, 16, 1)
        b = torch.zeros(1, 4, 16, 1)
        A, B = _parallel_scan(a, b)
        assert torch.allclose(A, torch.ones_like(A))
        assert torch.allclose(B, torch.zeros_like(B))

    def test_constant_offset_accumulates(self):
        """a=1, b=k yields x_t = x_0 + k*t."""
        from hydro_bench.methods.deep.water_mamba.selective_scan_fast import (
            _parallel_scan)

        L, k = 16, 0.3
        a = torch.ones(1, 2, L, 1)
        b = torch.full((1, 2, L, 1), k)
        A, B = _parallel_scan(a, b)
        steps = torch.arange(1, L + 1, dtype=torch.float32)
        assert torch.allclose(A.squeeze(), torch.ones(L))
        assert torch.allclose(B.squeeze(), k * steps, atol=1e-5)

    def test_geometric_decay(self):
        """a=r, b=0 yields x_t = r^(t+1) * x_0."""
        from hydro_bench.methods.deep.water_mamba.selective_scan_fast import (
            _parallel_scan)

        L, r = 8, 0.5
        a = torch.full((1, 1, L, 1), r)
        b = torch.zeros(1, 1, L, 1)
        A, B = _parallel_scan(a, b)
        steps = torch.arange(1, L + 1, dtype=torch.float32)
        assert torch.allclose(A.squeeze(), r ** steps, atol=1e-6)


class TestTiledForwardCoversEveryPixel:
    """The network is resolution-bound, so frames run as blended 256x256 tiles.

    Stride (224) is smaller than the tile, so a naive grid leaves an uncovered
    strip on the far right and bottom -- the window mass is zero there and the
    output is garbage rather than merely windowed. Reconstructing an identity
    network through the blender pins full coverage.
    """

    WEIGHTS = "WaterMamba_pretrained.pth"

    def _adapter(self):
        from pathlib import Path

        from hydro_bench.methods.deep.water_mamba.adapter import WaterMamba

        weights = Path(__file__).resolve().parent.parent.parent / "weights"
        ckpt = weights / self.WEIGHTS
        if not ckpt.is_file():
            pytest.skip(f"{self.WEIGHTS} not vendored")
        adapter = WaterMamba(checkpoint=ckpt)
        if adapter.used_fallback:
            pytest.skip("checkpoint present but the stack is unrunnable")
        # The blender must reproduce its input exactly when the "network" is
        # the identity: coverage gaps or a wrong window normalisation show up
        # as deviations from the input, not as NaNs.
        import torch

        adapter._model = torch.nn.Identity()
        return adapter

    @pytest.mark.parametrize("hw", [(48, 64), (64, 64), (256, 256), (257, 256),
                                    (300, 300), (450, 800), (512, 512),
                                    (800, 450), (1000, 700)])
    def test_identity_reconstruction(self, hw):
        import torch

        adapter = self._adapter()
        x = torch.rand(1, 3, *hw)
        out = adapter._tiled_forward(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-5), \
            f"{hw}: max |out - x| = {(out - x).abs().max().item():.3e}"
