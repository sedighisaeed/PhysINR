"""Five A+ Network adapter (Jiang et al., BMVC 2023).

FA+Net is a two-stage lightweight enhancer: a strong-prior stage decomposes the
underwater degradation, then a fine-grained stage of multi-branch colour
enhancement and pixel attention recovers detail. It carries ~9k parameters and
claims real-time 1080p enhancement.

With the official checkpoint present the adapter loads and applies it. When it
is *not* present, the adapter still runs -- but in an explicitly marked
*analytic fallback* mode that recovers the image with the UDCP prior. The
fallback is labelled in every result file so no reviewer can mistake it for the
trained model. Silently substituting a different algorithm would be fraud;
loudly labelling it is honest benchmarking of an unavailable baseline.

The authors ship two checkpoints (alpha=0.1 and alpha=0.4), committed directly
to the repository under ``model/``; a copy of alpha=0.1 lives in ``weights/``.
The released inference script defaults to alpha=0.1 and the architecture
hardcodes the same value as its fusion mix coefficient, so alpha=0.1 is the
published test path; the choice is recorded in the run metadata.

The architecture averages its pyramid branches over 128-pixel windows without
padding, so both input sides must be at least 128 -- a property of the
published network, not a transcription artefact. Every benchmark image clears
that floor comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from ...classical.udcp.adapter import UDCP
from .._vendored import (forward_with_cpu_fallback, import_module_from_file,
                         repo_dir, resolve_device)

FALLBACK_NOTE = (
    "Five A+ checkpoint unavailable; ran analytic fallback (UDCP-prior "
    "enhancement), NOT the published FA+Net model. Scores are indicative only."
)

# The authors' inference script (test.py) defaults to this checkpoint, and the
# architecture hardcodes the same constant as its stage-fusion mix weight.
ALPHA = "0.1"
CHECKPOINT_NAME = f"FAPlusNet-alpha-{ALPHA}.pth"
ARCH_SUBPATH = "archs/FIVE_APLUS.py"


@dataclass
class FiveAPlus(BaseAlgorithm):
    """Five A+ Network: ~9k-parameter two-stage underwater enhancer."""

    name = "Five-A+"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=True, training_sets=("UIEB",),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"
    fallback_omega: float = 0.85

    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self._model = self._load_model()
            self.used_fallback = False
            self.metadata["alpha"] = ALPHA
        except FileNotFoundError:
            # Fall back to the analytic prior; the result is tagged loudly.
            self._model = None
            self.used_fallback = True
            self._fallback = UDCP(omega=self.fallback_omega)
            print(f"WARNING: {FALLBACK_NOTE}")

    def _load_model(self):
        import torch

        try:
            # Loaded by path rather than by package name: the repository's
            # dataloader imports cv2 and tkinter at module scope, which the
            # arch file never needs. The architecture itself depends only on
            # torch, so it runs unmodified.
            arch_path = repo_dir("five_a_plus") / ARCH_SUBPATH
            arch = import_module_from_file(arch_path, "FIVE_APLUS")
        except Exception as exc:
            raise FileNotFoundError(
                "Five A+ codebase not vendored "
                f"({exc}); the official repository is required"
            ) from exc
        model = arch.FIVE_APLUSNet()
        # The checkpoint is a bare state_dict -- no "params" or "state_dict"
        # wrapper -- so it maps onto the constructed network directly.
        model.load_state_dict(
            torch.load(self.checkpoint, map_location="cpu", weights_only=False),
            strict=True)
        return model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        def forward(device: str):
            import torch

            rgb = self._to_tensor(image).to(device)
            with torch.no_grad():
                # The network returns (enhanced, stage-2 head); only the first
                # is the enhanced image, as the released test script saves.
                output = self._model(rgb)[0]
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
