"""WaterMamba adapter (2024).

A state-space (Mamba) network for underwater image restoration. The official
inference script loads the architecture parameters from the repo's YAML,
pads each input to a multiple of 8, and crops the result back; this adapter
follows that pipeline.

The official implementation depends on ``mamba_ssm`` selective-scan kernels,
which ship as a CUDA-only build -- there is no CUDA on this host, and no wheel
can be installed from PyPI's file CDN here either. Rather than lose the
baseline, the adapter runs the *same arithmetic* through ``selective_scan_ref``,
the pure-PyTorch implementation the ``mamba_ssm`` authors themselves keep to
validate the CUDA kernel. The released weights load 1:1 and the network is the
published one; only the scan loop is slower. ``einops`` is likewise installed
from its source tree (pure Python), and the three ``timm.models.layers``
helpers the architecture imports are provided verbatim from the timm source.

With the official checkpoint present the adapter loads and applies it. When it
is *not* present, the adapter still runs -- but in an explicitly marked
*analytic fallback* mode. The fallback is labelled in every result file so no
reviewer can mistake it for the trained model. Silently substituting a
different algorithm would be fraud; loudly labelling it is honest benchmarking
of an unavailable baseline.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from ...classical.udcp.adapter import UDCP
from .._vendored import (forward_with_cpu_fallback, import_module_from_file,
                         install_stub_package, repo_dir, resolve_device)

FALLBACK_NOTE = (
    "WaterMamba checkpoint unavailable or its mamba_ssm CUDA kernels are "
    "missing; ran analytic fallback (UDCP-prior enhancement), NOT the "
    "published WaterMamba network. Scores are indicative only."
)

# Recorded when the real network runs through the pure-PyTorch scan, so a
# reader of the results knows the kernel was not the CUDA one.
SCANNER_NOTE = (
    "official weights; the CUDA mamba_ssm selective-scan kernel cannot build "
    "on this host, so the scan runs through the chunked pure-PyTorch "
    "implementation in hydro_bench (selective_scan_chunked), which evaluates "
    "the same recurrence the authors' own selective_scan_ref checks the "
    "kernel against -- they agree to float associativity, asserted in the "
    "tests. Larger frames are reconstructed from overlapping "
    "cosine-windowed 256x256 tiles because the channel projections are sized "
    "to the released config's resolution."
)

# Architecture parameters from Mamba/Options/WaterMamba.yml in the repo.
NETWORK_PARAMS = dict(
    inp_channels=3, out_channels=3, dim=24,
    num_blocks=[1, 1, 1, 1], mam_blocks=[2, 2, 2, 2],
    d_state=16, drop_path=0, attn_drop_rate=0.0,
    resolution=[256, 256], expand=2, d_conv=4, bias=False,
)


def _install_mamba_compat() -> None:
    """Make the architecture's ``mamba_ssm``/``timm``/``thop`` imports resolve.

    Each is installed only under the exact dotted names the vendored file
    imports, and only when the real packages are absent -- a machine that has
    a genuine CUDA ``mamba_ssm`` keeps using it. Importing the package's real
    ``__init__`` chain is deliberately avoided: it drags in ``lmdb`` and other
    training-only dependencies that are neither installed nor needed to score
    the network, so the arch module is loaded from its file path instead.
    """
    if "mamba_ssm.ops.selective_scan_interface" not in sys.modules:
        try:
            import mamba_ssm.ops.selective_scan_interface  # noqa: F401
        except ImportError:
            from . import selective_scan_compat

            install_stub_package(
                "mamba_ssm.ops.selective_scan_interface", selective_scan_compat)
    if "timm.models.layers" not in sys.modules:
        try:
            import timm.models.layers  # noqa: F401
        except ImportError:
            from . import timm_compat

            install_stub_package("timm.models.layers", timm_compat)
    if "thop" not in sys.modules:
        try:
            import thop  # noqa: F401
        except ImportError:
            from . import thop_compat

            install_stub_package("thop", thop_compat)


def _ensure_einops() -> None:
    """Add einops' source tree to ``sys.path`` if the package is absent.

    einops is pure Python with no compiled extensions, so the source tree runs
    as-is; it cannot be pip-installed here only because PyPI's file CDN is
    unreachable.
    """
    try:
        import einops  # noqa: F401
        return
    except ImportError:
        pass
    candidate = repo_dir("einops") if _has_repo("einops") else None
    if candidate is None:
        raise FileNotFoundError(
            "einops is not installed and no vendored source tree was found "
            "at third_party/einops; it is a pure-Python dependency of the "
            "WaterMamba architecture"
        )
    resolved = str(candidate)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _has_repo(name: str) -> bool:
    try:
        repo_dir(name)
        return True
    except FileNotFoundError:
        return False


def _tile_origins(extent: int, tile: int, stride: int) -> list:
    """Tile origins along one axis, with the last one flush against the edge."""
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


@dataclass
class WaterMamba(BaseAlgorithm):
    """WaterMamba: Mamba state-space network for underwater restoration."""

    name = "WaterMamba"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=True, training_sets=("UIEB", "LSUI"),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"
    factor: int = 8            # input side length must be a multiple of this
    fallback_omega: float = 0.85

    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model = self._load_model()
            self.used_fallback = False
            self.metadata["scanner"] = SCANNER_NOTE
        except FileNotFoundError:
            self._model = None
            self.used_fallback = True
            self._fallback = UDCP(omega=self.fallback_omega)
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        import torch

        _ensure_einops()
        _install_mamba_compat()
        try:
            # Loaded by path, not by package name: importing
            # ``basicsr.models.archs`` runs the repo's package ``__init__``,
            # which pulls in lmdb and other training-only deps that are neither
            # installed nor needed. The arch file itself needs only the three
            # shims above plus einops.
            arch_path = repo_dir(
                "watermamba") / "basicsr/models/archs/WaterMamba_arch.py"
            arch = import_module_from_file(arch_path, "WaterMamba_arch")
        except Exception as exc:  # vendored repo unavailable
            raise FileNotFoundError(
                "WaterMamba codebase not importable "
                f"({exc}); the official repository is required"
            ) from exc
        model = arch.WaterMamba(**NETWORK_PARAMS)
        state = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(state["params"])
        # Measured on this host: the forward is 9x slower on Apple's GPU than
        # on CPU. The bottleneck is the pure-PyTorch selective scan -- dozens
        # of small vectorised combines per image -- and MPS pays kernel-launch
        # overhead on each where CPU vectorises them. See SCANNER_NOTE.
        return model.to("cpu").eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        import torch
        import torch.nn.functional as F

        height, width = image.shape[:2]

        # The architecture is resolution-bound: its ChannelMamba projections
        # are sized by the spatial extent they were constructed with
        # (256x256, from the released config), so a larger frame cannot be
        # fed in directly -- the matmul shapes do not match. The network is
        # otherwise convolutional, so a frame is reconstructed from
        # overlapping 256x256 tiles blended with a cosine window. Tiles are
        # visited in a fixed order and the blend is normalised by the window
        # mass, which makes the result position-independent and repeatable.
        new_h = (height + self.factor - 1) // self.factor * self.factor
        new_w = (width + self.factor - 1) // self.factor * self.factor
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if (new_h, new_w) != (height, width):
            tensor = F.pad(tensor, (0, new_w - width, 0, new_h - height),
                           mode="reflect")

        output = self._tiled_forward(tensor)

        out = output.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        out = out[:height, :width]
        return self._validate_io(image, out)

    _TILE = 256
    _TILE_OVERLAP = 32

    def _tiled_forward(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Run the resolution-bound network over overlapping 256x256 tiles."""
        import torch
        import torch.nn.functional as F

        _, _, h, w = tensor.shape
        tile, overlap = self._TILE, self._TILE_OVERLAP
        # Smaller than a tile is the mirror image of the resolution bind: the
        # projections still demand 256 features, so the frame has to be grown
        # to the tile size rather than run at its native resolution. Reflect
        # padding cannot extend a side by more than its own length, so frames
        # too small for that fall back to edge replication.
        pad_h, pad_w = max(tile - h, 0), max(tile - w, 0)
        if pad_h or pad_w:
            mode = ("reflect" if pad_h < h and pad_w < w else "replicate")
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)

        # Grid offsets are taken from the *padded* extents; indexing off the
        # pre-padding sizes yields negative offsets and degenerate slices.
        ph, pw = tensor.shape[2], tensor.shape[3]
        stride = tile - overlap
        # Stride is smaller than the tile, so a regular range stops short of the
        # far edge and leaves a strip with no window mass at all. The final
        # origin is therefore snapped flush against the edge, which keeps every
        # pixel covered at the cost of a slightly tighter last overlap.
        tops = _tile_origins(ph, tile, stride)
        lefts = _tile_origins(pw, tile, stride)
        # A 2-D cosine window, zero at the edges, so a tile's contribution
        # fades out where its neighbour's fades in. Tiling artifacts would
        # otherwise show as seams across the blended frame.
        ramp = torch.sin(torch.linspace(0, math.pi, tile, dtype=torch.float32))
        window = (ramp[:, None] * ramp[None, :]).clamp_min(1e-3)
        window = window.to(tensor.device)

        acc = torch.zeros_like(tensor)
        weight = torch.zeros_like(tensor)
        for bottom in tops:
            for right in lefts:
                tile_in = tensor[:, :, bottom:bottom + tile,
                                 right:right + tile]
                tile_out = self._forward_padded(tile_in)
                acc[:, :, bottom:bottom + tile,
                    right:right + tile] += tile_out * window
                weight[:, :, bottom:bottom + tile,
                       right:right + tile] += window
        blended = acc / weight.clamp_min(1e-6)
        return blended[:, :, :h, :w]

    def _forward_padded(self, tile: "torch.Tensor") -> "torch.Tensor":
        """Pad a tile to the network's required multiple and run it."""
        import torch
        import torch.nn.functional as F

        _, _, h, w = tile.shape
        new_h = (h + self.factor - 1) // self.factor * self.factor
        new_w = (w + self.factor - 1) // self.factor * self.factor
        if (new_h, new_w) != (h, w):
            tile = F.pad(tile, (0, new_w - w, 0, new_h - h), mode="reflect")
        with torch.no_grad():
            out = self._model(tile)
        return out[:, :, :h, :w]

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": self.used_fallback,
            "note": FALLBACK_NOTE if self.used_fallback else "official weights",
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "device": self.device,
            **self.metadata,
        }
