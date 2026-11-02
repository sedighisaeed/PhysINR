"""Water-Net adapter (Li et al., 2019).

Water-Net learns medium transmission and ambient light from the underwater
image formation model. Its published PyTorch implementation takes four inputs
-- the raw image plus white-balance, histogram-equalised and gamma-corrected
copies -- and fuses them through confidence maps; the released checkpoint
expects exactly that pipeline, which this adapter reproduces.

With the official checkpoint present the adapter loads and applies it.
When it is *not* present, the adapter still runs -- but in an explicitly
marked *analytic fallback* mode that reproduces the network's architecture
block diagram (medium-transmission + ambient-light estimation followed by
Jaccard-style radiance recovery) with classical estimators. The fallback is
labelled in every result file so no reviewer can mistake it for the trained
model. Silently substituting a different algorithm would be fraud; loudly
labelling it is honest benchmarking of an unavailable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from ...classical.udcp.adapter import UDCP
from .._vendored import (forward_with_cpu_fallback, import_vendored,
                         resolve_device)

FALLBACK_NOTE = (
    "Water-Net checkpoint unavailable; ran analytic fallback (UDCP-prior "
    "medium-transmission recovery), NOT the trained Water-Net weights. "
    "Scores are indicative only."
)


@dataclass
class WaterNet(BaseAlgorithm):
    """Water-Net: learned medium transmission and ambient light."""

    name = "Water-Net"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=("UIEB",),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"
    fallback_omega: float = 0.9

    # Populated on first enhance(); surfaced through ``run_metadata``.
    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model, self._transform = self._load_model()
            self.used_fallback = False
        except FileNotFoundError:
            # Fall back to the analytic prior; the result is tagged loudly.
            self._model = None
            self.used_fallback = True
            self._fallback = UDCP(omega=self.fallback_omega)
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        import torch

        # The published PyTorch port ships the checkpoint-matching architecture
        # and its preprocessing; HydroBench imports both from the vendored
        # checkout rather than reimplementing them, so weights map 1:1.
        try:
            net = import_vendored("waternet.net", repo="waternet_pytorch")
            data = import_vendored("waternet.data", repo="waternet_pytorch")
        except Exception as exc:  # pragma: no cover - vendored repo only
            raise FileNotFoundError(
                f"Water-Net codebase not importable ({exc}); clone the "
                "official PyTorch port into third_party/waternet_pytorch"
            ) from exc
        model = net.WaterNet()
        state = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(state.get("model", state))
        return model.to(self.device).eval(), data.transform

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        import torch

        # Four-channel preprocessing: white balance, histogram equalisation
        # (CLAHE on luminance) and gamma correction, exactly as the published
        # inference entry point does.
        wb, gc, he = self._transform(image)

        def forward(device: str):
            def on_device(array: np.ndarray):
                return self._to_tensor(array).to(device)

            rgb = on_device(image)
            with torch.no_grad():
                output = self._model(rgb, on_device(wb), on_device(he),
                                     on_device(gc))
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
