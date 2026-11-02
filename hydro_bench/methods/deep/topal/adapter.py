"""TOPAL adapter (Liu et al., 2022).

TOPAL is a three-stage pipeline: a DeepWB network performs task-oriented
white balance, an MSBDN-RDFF network performs colour correction, and a
WaterFusion network fuses the two estimates. The official ``main.py`` wires
exactly these three models; this adapter reproduces that pipeline, loading
each stage's released checkpoint so no stage is replaced by a stand-in.

The colour-correction checkpoint is the one fetched into the shared weights
directory; the white-balance and fusion checkpoints ship inside the official
repository and are resolved from the vendored checkout.

Without *any* of the three the adapter runs an explicitly labelled analytic
fallback and records ``used_fallback: true``, so an unavailable baseline
occupies its table row honestly rather than being quietly swapped for a
different algorithm.
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
                        import_vendored, repo_dir, repo_on_syspath,
                        resolve_device)

FALLBACK_NOTE = (
    "TOPAL pipeline checkpoints unavailable; ran analytic fallback "
    "(UDCP-prior enhancement), NOT the published TOPAL pipeline. Scores are "
    "indicative only."
)

# Default checkpoints shipped inside the official repository.
DEFAULT_AWB_CHECKPOINT = "DeepWB/models/net_awb.pth"
DEFAULT_FUSION_CHECKPOINT = "logs/gan_300_20_1_0.1/checkpoint_model.pt"


@dataclass
class TOPAL(BaseAlgorithm):
    """TOPAL: task-oriented prior learning for underwater enhancement."""

    name = "TOPAL"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=("UIEB", "LSUI"),
    )

    # Colour-correction stage (MSBDN-RDFF); resolved via load_weights.
    checkpoint: Optional[Path] = None
    # White-balance (DeepWB) and fusion (WaterFusion) stages; resolved inside
    # the vendored repository unless an absolute path is given.
    awb_checkpoint: Optional[Path] = None
    fusion_checkpoint: Optional[Path] = None
    device: str = "auto"
    fallback_omega: float = 0.85

    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            self.checkpoint = load_weights(self.checkpoint, must_exist=True)
            self.awb_checkpoint = self._resolve_repo_weight(
                self.awb_checkpoint, DEFAULT_AWB_CHECKPOINT)
            self.fusion_checkpoint = self._resolve_repo_weight(
                self.fusion_checkpoint, DEFAULT_FUSION_CHECKPOINT)
            self._awb, self._color, self._fusion = self._load_pipeline()
            self.used_fallback = False
        except FileNotFoundError:
            self._model = None
            self._fallback = UDCP(omega=self.fallback_omega)
            self.used_fallback = True
            print(f"WARNING: {FALLBACK_NOTE}")

    @staticmethod
    def _resolve_repo_weight(path: Optional[Path],
                             default: str) -> Optional[Path]:
        """Resolve a stage checkpoint, falling back to the repo's own copy."""
        if path is not None:
            resolved = Path(path).expanduser()
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(f"TOPAL stage checkpoint not found: {path}")
        shipped = repo_dir("topal") / default
        if not shipped.is_file():
            raise FileNotFoundError(
                f"TOPAL stage checkpoint not shipped at {shipped}"
            )
        return shipped

    def _load_pipeline(self):
        import torch

        try:
            repo_on_syspath("topal")  # model.py and networks.base_networks
            from DeepWB import deep_wb_single_task
            from model import WaterFusion
            # The released colour-correction checkpoint is the MSBDN-DFF v1.1
            # variant; its file name is not a legal identifier, hence the
            # path-based import.
            msbdn = import_module_from_file(
                repo_dir("topal") / "networks" / "MSBDN-DFF-v1-1.py",
                "hydro_topal_msbdn_dff",
            )
        except Exception as exc:  # pragma: no cover - vendored repo only
            raise FileNotFoundError(
                f"TOPAL codebase not importable ({exc}); clone the official "
                "repository into third_party/topal"
            ) from exc

        awb = deep_wb_single_task.deepWBnet()
        awb.load_state_dict(
            torch.load(self.awb_checkpoint, map_location=self.device))

        color = msbdn.Net()
        state = torch.load(self.checkpoint, map_location=self.device)
        color.load_state_dict(state.get("state_dict", state))

        fusion = WaterFusion(in_channels=3, out_channels=3,
                             num_features=64, growthrate=32)
        fusion_state = torch.load(self.fusion_checkpoint,
                                  map_location=self.device)
        fusion.load_state_dict(fusion_state["state_dict"])

        for net in (awb, color, fusion):
            net.to(self.device).eval()
        return awb, color, fusion

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if self.used_fallback:
            out = self._fallback.enhance(image)
            return self._validate_io(image, out)

        import torch

        # DeepWB is a UNet: pad to a multiple of 16, then crop back.
        height, width = image.shape[:2]
        padded = self._pad(image)

        def forward(device: str):
            tensor = torch.from_numpy(padded.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                balanced = self._awb(tensor)
                corrected = self._color(tensor)
                output = self._fusion(balanced, corrected)
            out = output.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)[:height, :width]

        out = forward_with_cpu_fallback(
            forward, (self._awb, self._color, self._fusion), self.device,
            self.metadata)
        return self._validate_io(image, out)

    @staticmethod
    def _pad(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        new_h = (height + 15) // 16 * 16
        new_w = (width + 15) // 16 * 16
        if (new_h, new_w) == (height, width):
            return image
        import cv2

        return cv2.copyMakeBorder(image, 0, new_h - height, 0, new_w - width,
                                  cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": self.used_fallback,
            "note": FALLBACK_NOTE if self.used_fallback else "official weights",
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "awb_checkpoint": str(self.awb_checkpoint) if self.checkpoint else None,
            "fusion_checkpoint": str(self.fusion_checkpoint) if self.checkpoint else None,
            "device": self.device,
            **self.metadata,
        }
