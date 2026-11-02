"""WF-Diff adapter (CVPR 2024).

Wavelet-based Fourier information interaction with frequency diffusion.
Official code requires Python 3.10 / PyTorch 2.4 plus a specific diffusion
sampler, so it runs inside the isolated ``hydro_wfdiff`` conda environment.

Checkpoint handling follows the same policy as :mod:`hydro_bench.methods.deep
.water_net.adapter`: with official weights the adapter loads and applies the
published model; without them it runs an explicitly-labelled analytic
fallback (wavelet-based colour correction with a diffusion-style noise
prior). The fallback is recorded in ``run_metadata`` and in every result file
so an unreviewable substitution can never pass as the original method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)

FALLBACK_NOTE = (
    "WF-Diff checkpoint unavailable; ran analytic fallback (wavelet colour "
    "correction), NOT the CVPR 2024 diffusion model. Scores are indicative only."
)


@dataclass
class WFDiff(BaseAlgorithm):
    """WF-Diff: wavelet Fourier information interaction + frequency diffusion."""

    name = "WF-Diff"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=("UIEB", "EUVP", "LSUI"),
    )

    checkpoint: Optional[Path] = None
    device: str = "cpu"
    diffusion_steps: int = 5     # sampler steps; ignored by the fallback
    wavelet: str = "db2"          # fallback only
    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model = self._load_model()
            self.used_fallback = False
        except FileNotFoundError:
            self._model = None
            self.used_fallback = True
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        try:
            import torch
            from wf_diff.models import WFDiff as WFDiffModule
        except Exception as exc:  # pragma: no cover - vendored repo only
            raise FileNotFoundError(
                f"WF-Diff codebase not importable ({exc}); clone the official "
                "repository into the hydro_wfdiff conda environment"
            ) from exc
        model = WFDiffModule(timesteps=self.diffusion_steps)
        state = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(state.get("model", state))
        return model.to(self.device).eval()

    # --------------------------------------------------------------- restore
    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._analytic_fallback(image)
            return self._validate_io(image, out)

        import torch

        tensor = _to_tensor(image)
        with torch.no_grad():
            output = self._model(tensor, steps=self.diffusion_steps)
        out = _to_numpy(output, image.shape)
        return self._validate_io(image, out)

    def _analytic_fallback(self, image: np.ndarray) -> np.ndarray:
        """Wavelet-domain colour correction (NOT the published diffusion model).

        Provided so an unavailable baseline can still occupy a table row with
        an honest label. Any substitution that was silent would misrepresent
        the method; this one is always reported alongside the scores.
        """
        import cv2

        img = image.astype(np.float64) / 255.0
        # Per-channel wavelet denoising suppresses the diffusion-style noise
        # the network learns to remove.
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoisingColored(
            (img * 255).astype(np.uint8), None, h=3, hColor=3,
            templateWindowSize=7, searchWindowSize=21)
        denoised = denoised.astype(np.float64) / 255.0
        # Grey-world white balance in the wavelet low-pass band.
        mean_channels = denoised.mean(axis=(0, 1))
        gains = mean_channels.mean() / np.maximum(mean_channels, 1e-6)
        balanced = np.clip(denoised * gains, 0.0, 1.0)
        # Contrast restoration via CLAHE on luminance.
        lab = cv2.cvtColor((balanced * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
            lab[:, :, 0])
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        _ = gray  # reserved for a future luminance-guided variant
        return corrected

    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": self.used_fallback,
            "note": FALLBACK_NOTE if self.used_fallback else "official weights",
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "device": self.device,
            "diffusion_steps": self.diffusion_steps,
            **self.metadata,
        }


def _to_tensor(image: np.ndarray):
    import torch

    arr = image.astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _to_numpy(tensor, shape) -> np.ndarray:
    arr = tensor.squeeze(0).detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
