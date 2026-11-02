"""Multi-Scale Retinex (MSR) with colour restoration.

Classical baseline. Implements the Rahman, Jobson & Woodell single-scale
Retinex and its multi-scale extension, plus the colour restoration factor that
keeps MSR from desaturating underwater imagery (a well-known failure mode:
unweighted MSR gains dynamic range but shifts hue toward grey).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)


@dataclass
class Retinex(BaseAlgorithm):
    """Multi-Scale Retinex with colour restoration."""

    name = "Retinex"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False, training_sets=(),
    )

    scales: Sequence[float] = (15.0, 80.0, 250.0)   # Gaussian sigmas
    weights: Optional[Sequence[float]] = None        # defaults to uniform
    sigma: float = 30.0
    gain: float = 128.0          # output gain (classic Retinex default)
    offset: float = 128.0
    color_gain: float = 125.0    # colour restoration gain
    color_alpha: float = 125.0
    color_beta: float = 46.0
    clip: float = 2.0            # percentile used for auto-rescaling
    checkpoint: Optional[Path] = None

    def __post_init__(self) -> None:
        load_weights(self.checkpoint, must_exist=False)
        if not self.scales:
            raise ValueError("Retinex requires at least one scale")
        if self.weights is None:
            self.weights = [1.0 / len(self.scales)] * len(self.scales)
        if len(self.weights) != len(self.scales):
            raise ValueError("weights and scales must have equal length")
        total = float(sum(self.weights))
        self.weights = [w / total for w in self.weights]

    # ------------------------------------------------------------- internals
    @staticmethod
    def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
        import cv2

        ksize = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _single_scale_retinex(self, image: np.ndarray, sigma: float) -> np.ndarray:
        # log(I) - log(I * G), computed in a numerically safe way.
        eps = 1.0
        img = image.astype(np.float64) + eps
        illumination = self._gaussian_blur(img, sigma)
        return np.log(img) - np.log(np.maximum(illumination, 1.0))

    # --------------------------------------------------------------- restore
    def enhance(self, image: np.ndarray) -> np.ndarray:
        original = image
        img = original.astype(np.float64) + 1.0

        # Multi-scale Retinex: weighted sum of single-scale outputs.
        msr = np.zeros_like(img)
        for scale, weight in zip(self.scales, self.weights):
            msr += weight * self._single_scale_retinex(original, scale)

        # Colour restoration factor (Jobson, Rahman & Woodell 1997). MSR is a
        # difference of logs, so it discards absolute channel energy; CRF
        # restores the relative colour balance that MSR would otherwise
        # desaturate. The intensity is the *mean* over channels -- using the
        # sum makes the log term large and negative, which inverts the result.
        intensity = img.mean(axis=2, keepdims=True)
        crf = self.color_beta * (np.log(self.color_alpha * img)
                                 - np.log(intensity))
        # CRF must stay strictly positive to act as a multiplier.
        crf = np.maximum(crf, 1e-3)

        out = self.gain * (msr * crf) + self.offset

        # Rescale each channel to [0, 1] with robust percentiles so a few
        # outlier pixels cannot crush the whole image, then to uint8.
        result = np.empty_like(out)
        lo_q, hi_q = self.clip, 100.0 - self.clip
        for c in range(3):
            channel = out[:, :, c]
            lo, hi = np.percentile(channel, [lo_q, hi_q])
            if hi - lo < 1e-8:
                channel = np.zeros_like(channel)
            else:
                channel = (channel - lo) / (hi - lo)
            result[:, :, c] = channel
        result = np.clip(result, 0.0, 1.0)
        uint8 = (result * 255.0).round().astype(np.uint8)
        return self._validate_io(original, uint8)
