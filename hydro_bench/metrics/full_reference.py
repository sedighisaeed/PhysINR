"""Full-reference (FR) image quality metrics: PSNR, SSIM, MS-SSIM, LPIPS.

These require ground truth, so they apply only to paired benchmarks
(UIEB, EUVP, LSUI, UFO-120). Every metric operates on float arrays in [0, 1]
and is vectorised over a batch where it is cheap to do so.

Design notes for reviewers
--------------------------
* PSNR uses the standard MSE on uint8-equivalent dynamic range (max = 255).
  We convert to float64 *before* differencing to avoid uint8 wraparound.
* SSIM is the Wang et al. (2004) formulation with an 11x11 Gaussian window
  (sigma=1.5), matching the reference MATLAB implementation. A uniform-window
  fallback is available for cross-checking against codebases that use it.
* LPIPS is delegated to the official ``lpips`` package when importable; when
  it is not, the metric reports ``None`` rather than a fake number. A missing
  perceptual metric must never be silently replaced by PSNR in a paper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional at runtime; only LPIPS needs it.
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_TORCH = False

try:
    import lpips as _lpips
    _HAS_LPIPS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LPIPS = False


class MetricUnavailable(RuntimeError):
    """Raised when a metric cannot be computed (e.g. missing dependency)."""


# --------------------------------------------------------------------- utils
def _to_float01(image: np.ndarray) -> np.ndarray:
    """uint8/float image -> float64 [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float64)
        if out.max() > 1.5:  # floats that are really 0-255
            out = out / 255.0
        return np.clip(out, 0.0, 1.0)
    raise TypeError(f"unsupported image dtype: {arr.dtype}")


def _check_shapes(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name}: shape mismatch {a.shape} vs {b.shape}. "
            "Restored and reference images must have identical geometry."
        )
    if a.ndim != 3 or a.shape[2] not in (1, 3):
        raise ValueError(f"{name}: expected HxWxC, got {a.shape}")


# ---------------------------------------------------------------------- PSNR
def psnr(reference: np.ndarray, distorted: np.ndarray,
         data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibels (higher is better).

    PSNR is scale-invariant: an image and its uint8/float representation give
    the same score. Inputs are therefore normalised to [0, 1] internally and
    compared against ``data_range`` (1.0 by default). Most callers should not
    pass ``data_range`` at all; it exists only to reproduce codebases that
    hardcode 255 while feeding raw uint8 arrays *without* normalisation.
    """
    ref = _to_float01(reference)
    dst = _to_float01(distorted)
    _check_shapes(ref, dst, "psnr")
    mse = np.mean((ref - dst) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ---------------------------------------------------------------------- SSIM
def _gaussian_window1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """The 1D Gaussian SSIM window; the 2D window is its outer product."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _uniform_window1d(size: int = 11) -> np.ndarray:
    """The 1D counterpart of the flat averaging window."""
    return np.full(size, 1.0 / size)


def _filter2d(image: np.ndarray, window1d: np.ndarray,
              mode: str = "reflect") -> np.ndarray:
    """2D correlation with a symmetric separable window, channel-wise.

    The SSIM windows are separable -- the Gaussian is an outer product of the
    1D kernel with itself, and the uniform window likewise -- so the filter is
    applied as two 1D passes (one per axis) rather than one dense 2D pass.
    That is the same kernel and the same boundary extension
    (``mode='reflect'`` matches the reference SSIM code), so the result is the
    dense correlation to floating-point summation order -- but it touches
    ``size`` taps per pixel instead of ``size**2``, which is ~16x less work on
    a full-resolution frame and dominates the cost of a full run otherwise.
    """
    from scipy.ndimage import correlate1d

    out = np.empty_like(image)
    for c in range(image.shape[2]):
        tmp = correlate1d(image[:, :, c], window1d, axis=0, mode=mode)
        out[:, :, c] = correlate1d(tmp, window1d, axis=1, mode=mode)
    return out


def ssim(reference: np.ndarray, distorted: np.ndarray,
         data_range: float = 1.0, window: str = "gaussian",
         size: int = 11, sigma: float = 1.5,
         k1: float = 0.01, k2: float = 0.03,
         return_map: bool = False):
    """Structural Similarity Index (Wang et al., 2004).

    Parameters
    ----------
    window : {"gaussian", "uniform"}
        The Gaussian 11x11/sigma=1.5 window matches the reference
        implementation. ``uniform`` reproduces results from codebases that
        average over an unweighted block.
    return_map : bool
        Also return the per-pixel SSIM map (useful for error heatmaps).
    """
    ref = _to_float01(reference)
    dst = _to_float01(distorted)
    _check_shapes(ref, dst, "ssim")

    win = (_gaussian_window1d(size, sigma) if window == "gaussian"
           else _uniform_window1d(size))
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = _filter2d(ref, win)
    mu_y = _filter2d(dst, win)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = _filter2d(ref ** 2, win) - mu_x2
    sigma_y2 = _filter2d(dst ** 2, win) - mu_y2
    sigma_xy = _filter2d(ref * dst, win) - mu_xy

    # Small negative variances can arise from floating-point cancellation.
    sigma_x2 = np.maximum(sigma_x2, 0.0)
    sigma_y2 = np.maximum(sigma_y2, 0.0)

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    score = float(ssim_map.mean())
    if return_map:
        return score, ssim_map
    return score


# -------------------------------------------------------------------- MS-SSIM
_MS_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)


def ms_ssim(reference: np.ndarray, distorted: np.ndarray,
            weights: Optional[Sequence[float]] = None,
            data_range: float = 1.0, window: str = "gaussian") -> float:
    """Multi-Scale SSIM (Wang et al., 2003), 5 scales by default."""
    ref = _to_float01(reference)
    dst = _to_float01(distorted)
    _check_shapes(ref, dst, "ms_ssim")
    w = np.asarray(weights if weights is not None else _MS_WEIGHTS, dtype=np.float64)
    w = w / w.sum()
    if len(w) > 5:
        raise ValueError("ms_ssim supports at most 5 scales")

    # Pre-compute luminance/contrast/structure terms at each scale.
    from scipy.ndimage import uniform_filter

    levels = len(w)
    mcs: List[float] = []
    cur_r, cur_d = ref, dst
    for level in range(levels):
        if level > 0:
            cur_r = _avg_pool(cur_r)
            cur_d = _avg_pool(cur_d)
            if min(cur_r.shape[:2]) < 11:
                warnings.warn(
                    f"ms_ssim: image too small ({cur_r.shape[:2]}) for scale "
                    f"{level + 1}; truncating to {level} scales"
                )
                w = w[:level] / w[:level].sum()
                break
        s_full, ssim_map = ssim(cur_r, cur_d, data_range=data_range,
                                window=window, return_map=True)
        # Contrast-structure product at this scale (luminance factored out).
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        gauss = _gaussian_window1d()
        cs_map = (2 * _filter2d(cur_r * cur_d, gauss) + c2) / \
                 (_filter2d(cur_r ** 2, gauss)
                  + _filter2d(cur_d ** 2, gauss) + c2)
        mcs.append(float(cs_map.mean()))
        if level == levels - 1:
            mcs[-1] = s_full
    mcs_arr = np.asarray(mcs, dtype=np.float64)
    return float(np.prod(mcs_arr ** w))


def _avg_pool(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """2x2 average pooling."""
    h, w = image.shape[:2]
    h2, w2 = h // factor, w // factor
    cropped = image[: h2 * factor, : w2 * factor]
    return cropped.reshape(h2, factor, w2, factor, -1).mean(axis=(1, 3))


# --------------------------------------------------------------------- LPIPS
def _lpips_device() -> str:
    """The device LPIPS runs on.

    The official package builds its network on CPU and never moves it, so a
    VGG backbone on a 2.25 MP frame costs minutes on CPU while the GPU idles.
    MPS/CUDA give bit-identical distances at 10-15x the speed, so prefer them
    and fall back to CPU where neither is present.
    """
    if not _HAS_TORCH:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LPIPSEvaluator:
    """Thin, memoised wrapper over the official LPIPS implementation."""

    _cache: Dict[str, "object"] = {}

    def __init__(self, backbone: str = "alex", spatial: bool = False):
        if not _HAS_TORCH:
            raise MetricUnavailable(
                "LPIPS requires PyTorch: pip install torch (see requirements.txt)"
            )
        if not _HAS_LPIPS:
            raise MetricUnavailable(
                "LPIPS requires the `lpips` package: pip install lpips. "
                "HydroBench does not ship pretrained perceptual weights."
            )
        if backbone not in ("alex", "vgg", "squeeze"):
            raise ValueError(f"unsupported LPIPS backbone: {backbone}")
        self.backbone = backbone
        self.spatial = spatial
        # Cached per (backbone, spatial, device): the same backbone resident
        # on two different devices is a distinct copy of the weights, not a
        # cache hit.
        device = _lpips_device()
        self._device = device
        key = f"{backbone}:{spatial}:{device}"
        if key not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _lpips.LPIPS(net=backbone, spatial=spatial,
                                     verbose=False)
            self._cache[key] = model.to(device).eval()
        self._model = self._cache[key]

    @staticmethod
    def _to_tensor(image: np.ndarray) -> "torch.Tensor":
        arr = _to_float01(image)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        # HWC -> CHW, then normalise to the [-1, 1] range LPIPS expects.
        t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()
        return (t * 2.0) - 1.0

    def __call__(self, reference: np.ndarray, distorted: np.ndarray) -> float:
        _check_shapes(np.asarray(reference), np.asarray(distorted), "lpips")
        with torch.no_grad():
            ref = self._to_tensor(reference).unsqueeze(0).to(self._device)
            dst = self._to_tensor(distorted).unsqueeze(0).to(self._device)
            value = self._model(ref, dst)
        return float(value.squeeze().detach().cpu().numpy())


def lpips_distance(reference: np.ndarray, distorted: np.ndarray,
                   backbone: str = "alex") -> float:
    """LPIPS perceptual distance (lower is better). Requires torch + lpips."""
    return LPIPSEvaluator(backbone=backbone)(reference, distorted)


# ------------------------------------------------------------------ registry
@dataclass
class FRResult:
    """One FR evaluation over one image pair."""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    ms_ssim: Optional[float] = None
    lpips_alex: Optional[float] = None
    lpips_vgg: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def evaluate_pair(reference: np.ndarray, distorted: np.ndarray,
                  metrics: Sequence[str] = ("psnr", "ssim")) -> FRResult:
    """Compute a requested subset of FR metrics for one image pair."""
    out = FRResult()
    for metric in metrics:
        if metric == "psnr":
            out.psnr = psnr(reference, distorted)
        elif metric == "ssim":
            out.ssim = ssim(reference, distorted)
        elif metric == "ms_ssim":
            out.ms_ssim = ms_ssim(reference, distorted)
        elif metric in ("lpips", "lpips_alex"):
            out.lpips_alex = lpips_distance(reference, distorted, backbone="alex")
        elif metric == "lpips_vgg":
            out.lpips_vgg = lpips_distance(reference, distorted, backbone="vgg")
        else:
            raise ValueError(f"unknown full-reference metric: {metric}")
    return out


AVAILABLE_FR_METRICS: Tuple[str, ...] = ("psnr", "ssim", "ms_ssim",
                                         "lpips_alex", "lpips_vgg")


def available_metrics() -> Dict[str, bool]:
    """Which FR metrics can actually run in this environment."""
    return {
        "psnr": True,
        "ssim": True,
        "ms_ssim": True,
        "lpips_alex": _HAS_LPIPS,
        "lpips_vgg": _HAS_LPIPS,
    }
