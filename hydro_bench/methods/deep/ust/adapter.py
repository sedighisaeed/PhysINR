"""U-Shape Transformer adapter (Peng et al., 2023).

A transformer-driven U-Net generator for underwater image restoration. The
official inference resizes inputs to 256x256 and scales to [0, 1]; the
generator returns a multi-scale list whose last element is the
full-resolution result. This adapter follows that pipeline, restoring the
original geometry afterwards.

With the official checkpoint present the adapter loads and applies the
released generator. Without it the adapter runs an explicitly labelled
analytic fallback and records ``used_fallback: true``, so an unavailable
baseline occupies its table row honestly rather than being quietly swapped
for a different algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from ...classical.udcp.adapter import UDCP
from .._vendored import import_vendored

FALLBACK_NOTE = (
    "U-Shape Transformer checkpoint unavailable; ran analytic fallback "
    "(UDCP-prior enhancement), NOT the published UST generator. Scores are "
    "indicative only."
)


@dataclass
class UShapeTransformer(BaseAlgorithm):
    """U-Shape Transformer: multi-scale transformer generator for UIR."""

    name = "U-Shape-Transformer"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=("UIEB", "LSUI"),
    )

    checkpoint: Optional[Path] = None
    device: str = "cpu"
    input_size: int = 256       # official test resolution
    fallback_omega: float = 0.85

    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model = self._load_model()
            self.used_fallback = False
        except FileNotFoundError:
            self._model = None
            self._fallback = UDCP(omega=self.fallback_omega)
            self.used_fallback = True
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        import torch

        try:
            ust = import_vendored(
                "net.Ushape_Trans", env_package="hydro_ust", repo="ust",
            )
        except Exception as exc:  # pragma: no cover - vendored repo only
            raise FileNotFoundError(
                f"U-Shape Transformer codebase not importable ({exc}); clone "
                "the official repository into third_party/ust"
            ) from exc
        model = ust.Generator()  # paper defaults: img_dim 256, patch 16
        state = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(state.get("state_dict", state))
        return model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        import cv2
        import torch

        height, width = image.shape[:2]
        size = (self.input_size, self.input_size)
        resized = cv2.resize(image, size)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._model(tensor)
        # The generator emits a multi-scale pyramid; index 3 is full scale.
        out = output[3] if isinstance(output, (list, tuple)) else output
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = np.clip(out * 255.0, 0, 255)
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
        return self._validate_io(image, out.astype(np.uint8))

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": self.used_fallback,
            "note": FALLBACK_NOTE if self.used_fallback else "official weights",
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "device": self.device,
            "input_size": self.input_size,
            **self.metadata,
        }
