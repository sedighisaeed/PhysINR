"""Phaseformer adapter (Khan et al., 2025).

Phaseformer restores underwater images with a phase-based attention mechanism:
it operates on the phase of the Fourier spectrum (via ``inv_mag``, which keeps
the phase and discards the magnitude) inside an ECA attention block, on the
premise that underwater degradation distorts phase structure more severely than
amplitude. The backbone is a Restormer-style multi-scale transformer trained
end-to-end on paired underwater data.

The authors publish the trained checkpoints directly in the repository under
``checkpoints/<dataset>/best.pth``. These are *whole pickled modules*, not
state dicts, so loading uses ``weights_only=False``; the class must be
importable at load time, which the vendored ``model_with_eca.py`` provides.
The UIEB checkpoint (1.78 M parameters) is used here, copied to
``weights/phaseformer_uieb_best.pth``.

The published test path resizes the input to 512x512 with bicubic
interpolation, normalises to [-1, 1] with a 0.5 mean/std, and reads the second
of the two returned heads. This adapter reproduces exactly that, then resamples
back to the original geometry -- the resize is a property of the released
inference script, not of the network, and HydroBench's contract forbids
changing the output geometry, since that would pollute the full-reference
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from .._vendored import (forward_with_cpu_fallback, repo_on_syspath,
                         resolve_device)

CHECKPOINT_NAME = "phaseformer_uieb_best.pth"
MODULE = "model_with_eca"
CLASS = "Restormer"
# The released test script hardcodes this square size.
INFERENCE_SIZE = 512


@dataclass
class Phaseformer(BaseAlgorithm):
    """Phase-based attention Restormer for underwater restoration."""

    name = "Phaseformer"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=True, training_sets=("UIEB",),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"

    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.checkpoint is None:
            self.checkpoint = Path(CHECKPOINT_NAME)
        self.checkpoint = load_weights(self.checkpoint, must_exist=True)
        self._model = self._load_model()

    def _load_model(self):
        import torch

        repo_on_syspath("phaseformer")
        try:
            mod = __import__(MODULE, fromlist=[CLASS])
        except Exception as exc:  # pragma: no cover - vendoring failure
            raise FileNotFoundError(
                f"Phaseformer codebase not vendored ({exc})") from exc

        # Whole-module pickle: needs weights_only=False with the class on the
        # import path, exactly as the authors' test.py does.
        model = torch.load(self.checkpoint, map_location="cpu",
                           weights_only=False)
        if not hasattr(model, "parameters"):
            raise TypeError(
                f"expected a pickled nn.Module, got {type(model).__name__}")
        return model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        def forward(device: str):
            import torch
            import torch.nn.functional as F

            x = self._to_tensor(image).to(device)
            # Reproduce the released preprocessing exactly.
            x = F.interpolate(x, size=(INFERENCE_SIZE, INFERENCE_SIZE),
                              mode="bicubic", align_corners=False)
            x = (x - 0.5) / 0.5
            with torch.no_grad():
                out = self._model(x)[1]
            out = (out * 0.5) + 0.5
            # Back to the input geometry, per HydroBench's contract.
            out = F.interpolate(out, size=(h, w), mode="bicubic",
                                align_corners=False)
            return self._to_numpy(out, image.shape)

        out = forward_with_cpu_fallback(
            forward, (self._model,), self.device, self.metadata)
        return self._validate_io(image, out)

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": False,
            "note": "official implementation, authors' UIEB checkpoint; "
                    "released inference resizes to 512x512",
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
