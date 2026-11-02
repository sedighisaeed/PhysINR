"""Zero-DCE adapter (Guo et al., CVPR 2020).

Zero-Reference Deep Curve Estimation estimates a set of pixel-wise, high-order
tone-mapping curves from a single image with *no reference*: the network is
trained with a bundle of non-reference losses (spatial consistency, exposure
control, colour-constancy, illumination-smoothness) on unpaired low-light
photographs. It is included here as the canonical zero-reference baseline: it
is the strongest published claim that a network can enhance an image without
ever seeing a ground truth, which is exactly the axis this work competes on.

The released checkpoint (``Epoch99.pth``, 100 epochs on the authors' low-light
training set) is committed to the official repository under
``Zero-DCE_code/snapshots/``; a copy lives in ``weights/``. It is trained on
low-light imagery, not underwater, so this is a *cross-domain transfer* run of
the published weights -- reported as such, never silently relabelled. The
network is fully convolutional and small (79k parameters), so it runs in
process on CPU or MPS with no isolation.

The repository's ``model.py`` imports ``cv2``-backed helpers only behind
comments, and the architecture itself depends on torch alone, so it loads
unmodified. The published test script feeds the raw [0,1] image with no mean
normalisation and reads the second of the three returned tensors, which is the
fully-enhanced image after all eight curve applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from .._vendored import (forward_with_cpu_fallback, import_module_from_file,
                         repo_dir, resolve_device)

CHECKPOINT_NAME = "zero_dce_epoch99.pth"
ARCH_SUBPATH = "Zero-DCE_code/model.py"

# The authors train on low-light datasets (DICM/ExDark/NPE/LIME), never on
# underwater imagery. Every HydroBench leakage check keys off this tuple.
TRAINING_SETS: tuple = ()


@dataclass
class ZeroDCE(BaseAlgorithm):
    """Zero-Reference Deep Curve Estimation, published weights, transferred."""

    name = "Zero-DCE"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=TRAINING_SETS,
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"

    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.checkpoint is None:
            # Default to the copy staged in weights/ (HYDROBENCH_MODELS).
            self.checkpoint = Path(CHECKPOINT_NAME)
        self.checkpoint = load_weights(self.checkpoint, must_exist=True)
        self._model = self._load_model()
        self.metadata["transfer"] = (
            "checkpoint trained on low-light data (authors' set), applied "
            "zero-shot to underwater imagery"
        )

    def _load_model(self):
        import torch

        arch_path = repo_dir("zero_dce") / ARCH_SUBPATH
        arch = import_module_from_file(arch_path, "zero_dce_model")
        model = arch.enhance_net_nopool()
        state = torch.load(self.checkpoint, map_location="cpu",
                           weights_only=False)
        # The snapshot stores a bare state_dict under the training scheduler's
        # naming; the released test script loads it straight onto the net.
        model.load_state_dict(state, strict=True)
        return model.to(self.device).eval()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        def forward(device: str):
            import torch

            rgb = self._to_tensor(image).to(device)
            with torch.no_grad():
                # (stage-4 output, final 8-curve output, curve maps). The
                # released test script saves the final enhanced image.
                _, enhanced, _ = self._model(rgb)
            return self._to_numpy(enhanced, image.shape)

        out = forward_with_cpu_fallback(
            forward, (self._model,), self.device, self.metadata)
        return self._validate_io(image, out)

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": False,
            "note": "official implementation, published checkpoint; "
                    "low-light -> underwater transfer",
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
