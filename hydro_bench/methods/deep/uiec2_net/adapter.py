"""UIEC^2-Net adapter (Wang et al., Signal Processing: Image Communication 2021).

UIEC^2-Net enhances underwater images in two colour spaces (RGB and HSV) with
a dual-stream network: an RGB branch learns an enhanced image, an HSV branch
learns piecewise colour-correction curves, and a confidence net learns how to
combine the two.

With the official checkpoint present the adapter loads and applies it. When it
is *not* present, the adapter still runs -- but in an explicitly marked
*analytic fallback* mode that recovers the image with the UDCP prior. The
fallback is labelled in every result file so no reviewer can mistake it for
the trained model. Silently substituting a different algorithm would be fraud;
loudly labelling it is honest benchmarking of an unavailable baseline.

The architecture lives in ``uiec2net_arch.py`` next to this file. It is a
device-agnostic copy of the published network; the only mechanical changes are
``.cuda()`` device constants replaced by ``torch.zeros_like``/``torch.ones_like``
and the unused framework registry dropped (``backbone`` is ``None`` in the
released config), so the released checkpoint maps 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from ...classical.udcp.adapter import UDCP
from .._vendored import forward_with_cpu_fallback, resolve_device

FALLBACK_NOTE = (
    "UIEC^2-Net checkpoint unavailable; ran analytic fallback (UDCP-prior "
    "enhancement), NOT the published UIEC^2-Net model. Scores are "
    "indicative only."
)


@dataclass
class UIEC2Net(BaseAlgorithm):
    """UIEC^2-Net: dual colour-space enhancement network."""

    name = "UIEC2-Net"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=True, training_sets=("UIEB",),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"
    fallback_omega: float = 0.85

    # Populated on first enhance(); surfaced through ``run_metadata``.
    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model = self._load_model()
            self.used_fallback = False
        except FileNotFoundError:
            # Fall back to the analytic prior; the result is tagged loudly.
            self._model = None
            self.used_fallback = True
            self._fallback = UDCP(omega=self.fallback_omega)
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        import torch

        from .uiec2net_arch import UIEC2Net as UIEC2NetArch

        model = UIEC2NetArch()
        blob = torch.load(self.checkpoint, map_location="cpu",
                          weights_only=False)
        state = blob.get("state_dict", blob)
        # The checkpoint was saved through DataParallel, so every key carries
        # the "module." wrapper the trainer adds. It is a naming artefact, not
        # a different network.
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        def forward(device: str):
            import torch

            rgb = self._to_tensor(image).to(device)
            with torch.no_grad():
                output = self._model(rgb)
            return self._to_numpy(output, image.shape)

        out = forward_with_cpu_fallback(
            forward, (self._model,), self.device, self.metadata)
        return self._validate_io(image, out)

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

    # ----------------------------------------------------------------- io
    @staticmethod
    def _to_tensor(image: np.ndarray) -> "torch.Tensor":
        import torch

        arr = image.astype(np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    @staticmethod
    def _to_numpy(tensor, shape) -> np.ndarray:
        arr = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
