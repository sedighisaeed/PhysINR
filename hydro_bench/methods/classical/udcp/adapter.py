"""Underwater Dark Channel Prior (UDCP).

Re-implementation of Drews, Cadena, Ceballos & Neira, "Towards Deep Learning
from a Single View for Underwater Depth Estimation" / the UDCP formulation of
Carlevaris-Bianco, Mohan & Eustice, "Initial Results in Underwater Single
Image Dehazing" (ICRA 2010). This is a classical, learning-free baseline: it
is deliberately kept dependency-free (NumPy + OpenCV only) so it runs
everywhere and serves as the reference implementation other adapters are
compared against.

Algorithm
---------
1. Estimate the *underwater* dark channel, which uses the maximum of the B and
   G channels rather than the minimum over all channels -- red is absorbed
   fastest by water, so a naive grey-world dark channel is invalid here.
2. Estimate the medium transmission from the dark channel via a morphological
   opening and a soft matting-free refinement.
3. Recover the scene radiance with the atmospheric scattering model, using the
   estimated background light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)


@dataclass
class UDCP(BaseAlgorithm):
    """Underwater Dark Channel Prior restoration."""

    name = "UDCP"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=(),
    )

    patch_size: int = 15          # dark-channel window
    omega: float = 0.75           # scattering coefficient (depth haze retained)
    background_light_ratio: float = 0.95  # percentile used for background light
    min_transmission: float = 0.1  # guard against division blow-ups
    guided_filter_radius: int = 40
    guided_filter_eps: float = 1e-3
    gamma: float = 1.0            # optional post-gamma; 1.0 disables
    checkpoint: Optional[Path] = None

    def __post_init__(self) -> None:
        # UDCP is learning-free; a checkpoint path would be meaningless.
        load_weights(self.checkpoint, must_exist=False)

    # ------------------------------------------------------------- dark priors
    @staticmethod
    def _underwater_dark_channel(image: np.ndarray, patch: int) -> np.ndarray:
        """min(max(B, G)) over a patch -- the UDCP-specific dark channel.

        Red light is absorbed most strongly underwater, so the *minimum* over
        RGB is dominated by the red channel and is uninformative. The
        underwater dark channel instead takes the minimum over the patch of
        the maximum of the green and blue channels.
        """
        import cv2

        # Work in 0-1 float for numerical stability.
        img = image.astype(np.float64) / 255.0
        bg_max = np.maximum(img[:, :, 1], img[:, :, 2])
        if patch <= 1:
            return bg_max
        kernel = np.ones((patch, patch), np.uint8)
        return cv2.erode(bg_max, kernel)

    def _estimate_transmission(self, image: np.ndarray) -> np.ndarray:
        import cv2

        img = image.astype(np.float64) / 255.0
        dark = self._underwater_dark_channel(image, self.patch_size)

        # Background light: the brightest percentile of the dark channel.
        b = self.background_light_ratio
        bg_value = float(np.percentile(dark, b * 100.0))
        bg_value = max(bg_value, 1e-3)

        transmission = 1.0 - self.omega * (dark / bg_value)
        transmission = np.clip(transmission, self.min_transmission, 1.0)

        # Refine edges with a guided filter (fast joint-bilateral surrogate
        # for the original soft matting).
        if self.guided_filter_radius > 0 and hasattr(cv2, "ximgproc"):
            guide = cv2.ximgproc.guidedFilter(
                guide=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                src=transmission.astype(np.float32),
                radius=self.guided_filter_radius,
                eps=self.guided_filter_eps,
            )
            transmission = guide.astype(np.float64)
            transmission = np.clip(transmission, self.min_transmission, 1.0)
        elif self.guided_filter_radius > 0:
            # No opencv-contrib: fall back to a bilateral filter refinement.
            refined = cv2.bilateralFilter(transmission.astype(np.float32),
                                          d=self.guided_filter_radius,
                                          sigmaColor=0.05, sigmaSpace=10.0)
            transmission = refined.astype(np.float64)
        return np.clip(transmission, self.min_transmission, 1.0)

    def _background_light(self, image: np.ndarray) -> np.ndarray:
        """The ambient (backscatter) light, one value per channel."""
        img = image.astype(np.float64) / 255.0
        dark = self._underwater_dark_channel(image, self.patch_size)
        flat_dark = dark.ravel()
        k = max(1, int(round(flat_dark.size * 0.001)))
        brightest = np.argpartition(flat_dark, -k)[-k:]
        # Among the brightest dark-channel pixels take the mean colour.
        ys, xs = np.unravel_index(brightest, dark.shape)
        return img[ys, xs, :].mean(axis=0)

    # --------------------------------------------------------------- restore
    def enhance(self, image: np.ndarray) -> np.ndarray:
        original = image
        transmission = self._estimate_transmission(original)
        b_light = self._background_light(original)
        img = original.astype(np.float64) / 255.0

        # J(x) = (I(x) - B) / t(x) + B   (atmospheric scattering model)
        radiance = np.empty_like(img)
        for c in range(3):
            radiance[:, :, c] = (img[:, :, c] - b_light[c]) / transmission + b_light[c]
        radiance = np.clip(radiance, 0.0, 1.0)
        if self.gamma != 1.0:
            radiance = np.power(radiance, 1.0 / self.gamma)
        out = (radiance * 255.0).round().astype(np.uint8)
        return self._validate_io(original, out)
