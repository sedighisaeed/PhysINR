"""No-reference (NR) image quality metrics for underwater imagery.

Underwater-specific NR metrics are what make a UIE benchmark meaningful on
unpaired data (RUIE, UCCS, and the 60 UIEB challenge images): there is no
ground truth, so the score has to come from colour-science priors about what
"clear water" looks like.

Implemented here
----------------
* :func:`uciqe` -- Underwater Colour Image Quality Evaluation
  (Yang & Sowmya, 2015). Weighted combination of saturation, luminance
  contrast, and chroma contrast in LAB space.
* :func:`uiqm` -- Underwater Image Quality Measure (Panetta et al., 2016);
  a weighted sum of the colourfulness measure (ICM), the sharpness measure
  (SM), and the contrast measure (ConM).
* :func:`ccf` -- Colour Cast Factor, a diagnostic for residual colour bias.
* :func:`brisque` -- BRISQUE SVR score, via OpenCV's quality module when the
  ``opencv-contrib`` BRISQUE model file is available. Reported as ``None``
  otherwise; never faked.
* :func:`musiq` -- placeholder interface for the multi-scale IQA model;
  requires external weights, so it is registered as unavailable unless
  explicitly configured.

All NR scores are computed on the enhanced image only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Coefficients published in the UCIQE paper (Yang & Sowmya 2015).
UCIQE_WEIGHTS = (0.4680, 0.2745, 0.2576)

# Coefficients published in the UIQM paper (Panetta et al. 2016).
UIQM_WEIGHTS = (0.0282, 0.2953, 3.5755)


@dataclass
class NRResult:
    uciqe: Optional[float] = None
    uiqm: Optional[float] = None
    uicm: Optional[float] = None
    uism: Optional[float] = None
    uiconm: Optional[float] = None
    ccf: Optional[float] = None
    brisque: Optional[float] = None
    musiq: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        base = {k: v for k, v in self.__dict__.items()
                if k != "extra" and v is not None}
        base.update(self.extra)
        return base


# --------------------------------------------------------------------- utils
def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    raise TypeError(f"unsupported image dtype: {arr.dtype}")


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """CIE LAB (D65) from an sRGB uint8 image.

    Convention: L in [0, 100] and a, b in [-128, 127] -- the range used by
    MATLAB's ``rgb2lab`` and ``skimage.color.rgb2lab``, which is what the
    published UCIQE reference implementation assumes. (OpenCV's own LAB
    encoding squashes L into [0, 255]; using it silently would change every
    UCIQE value by a factor of ~2.55.)
    """
    import cv2

    rgb = _to_uint8(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = (lab[:, :, 0] / 255.0) * 100.0   # [0,255] -> [0,100]
    lab[:, :, 1:] -= 128.0
    return lab


# -------------------------------------------------------------------- UCIQE
def uciqe(image: np.ndarray, weights: Tuple[float, float, float] = UCIQE_WEIGHTS,
          saturation_luminance: float = 85.16) -> float:
    """Underwater Colour Image Quality Evaluation (higher is better).

    .. math:: \\mathrm{UCIQE} = w_1 \\sigma_c + w_2\\,\\mathrm{con_l} + w_3\\mu_s

    where :math:`\\sigma_c` is the chroma standard deviation, ``con_l`` the
    1%/99% luminance contrast, and :math:`\\mu_s` the mean saturation over
    pixels whose LAB luminance falls below ``saturation_luminance`` (the
    threshold of the original reference implementation, rescaled to the
    [0, 100] LAB luminance convention).
    """
    lab = _rgb_to_lab(image)
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)

    import cv2

    hsv = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2HSV).astype(np.float64)
    saturation = hsv[:, :, 1] / 255.0
    mask = lab[:, :, 0] < saturation_luminance
    if not np.any(mask):
        mask = np.ones_like(mask, dtype=bool)
    mu_s = float(np.mean(saturation[mask]))

    sigma_c = float(np.std(chroma))
    # 1% / 99% percentile contrast of luminance, matching common practice.
    lum = lab[:, :, 0]
    lo, hi = np.percentile(lum, 1), np.percentile(lum, 99)
    con_l = float(hi - lo)

    w1, w2, w3 = weights
    return float(w1 * sigma_c + w2 * con_l + w3 * mu_s)


# --------------------------------------------------------------------- UIQM
def _uicm(image: np.ndarray) -> float:
    """Underwater Image Colourfulness Measure."""
    rgb = _to_uint8(image).astype(np.float64)
    # The reference uses the mean of the green and blue channels' deviation
    # from red in the log-chromaticity plane.
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    alpha = (g - b) / (g + b + 1e-9)
    beta = (r - g) / (r + g + 1e-9)
    return float(np.sqrt(np.var(alpha) + np.var(beta)))


def _uism(image: np.ndarray) -> float:
    """Underwater Image Sharpness Measure.

    The published definition is edge energy; HydroBench uses the mean Sobel
    gradient magnitude over all three channels, normalised by the uint8 dynamic
    range so the score stays in [0, 1] and is comparable across codebases.
    """
    import cv2

    rgb = _to_uint8(image)
    sharpness = []
    for c in range(3):
        channel = rgb[:, :, c].astype(np.float64)
        gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(gx ** 2 + gy ** 2)
        sharpness.append(edges.mean() / 255.0)
    return float(np.mean(sharpness))


def _uiconm(image: np.ndarray, block: int = 8, contrast: str = "michelson") -> float:
    """Underwater Image Contrast Measure.

    Two definitions are provided:

    ``"michelson"`` (default)
        Mean Michelson block contrast ``(max - min) / (max + min)``. This is
        strictly increasing in block contrast, positive, and numerically
        stable -- haze drives it towards zero, which is the behaviour a
        contrast measure must have.

    ``"amee"``
        The log-AMEE variant transcribed in some reimplementations. It is
        *retained for legacy comparability only*: as ``max`` approaches
        ``min`` the ratio ``(max+min)/(max-min)`` diverges, so low-contrast
        (hazy) blocks score *higher* than high-contrast ones and the sign
        convention differs between sources. Do not use it for new work.
    """
    gray = _to_uint8(image)
    import cv2

    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape
    if h < block or w < block:
        block = max(1, min(block, h, w))
    values = []
    for i in range(0, h - block + 1, block):
        for j in range(0, w - block + 1, block):
            patch = gray[i:i + block, j:j + block]
            lo, hi = float(patch.min()), float(patch.max())
            if contrast == "amee":
                if hi - lo < 1e-8 or hi + lo < 1e-8:
                    continue
                ratio = (hi - lo) / (hi + lo)
                if ratio <= 0 or ratio >= 1:
                    continue
                values.append(((hi + lo) / (hi - lo)) * np.log(1.0 / ratio))
            elif contrast == "michelson":
                if hi + lo < 1e-8:
                    continue
                values.append((hi - lo) / (hi + lo))
            else:
                raise ValueError(f"unknown contrast definition: {contrast}")
    if not values:
        return 0.0
    return float(np.mean(values))


def uiqm(image: np.ndarray, weights: Tuple[float, float, float] = UIQM_WEIGHTS,
         components: bool = False, contrast: str = "michelson"):
    """Underwater Image Quality Measure (higher is better).

    .. math:: \\mathrm{UIQM} = c_1\\,\\mathrm{ICM} + c_2\\,\\mathrm{SM} + c_3\\,\\mathrm{ConM}

    Pass ``components=True`` to also get the three sub-scores; reporting them
    alongside the aggregate lets a reviewer audit how a method wins.
    """
    icm = _uicm(image)
    sm = _uism(image)
    conm = _uiconm(image, contrast=contrast)
    c1, c2, c3 = weights
    score = float(c1 * icm + c2 * sm + c3 * conm)
    if components:
        return score, {"uicm": icm, "uism": sm, "uiconm": conm}
    return score


# ---------------------------------------------------------------------- CCF
def ccf(image: np.ndarray, gray_bins: int = 100) -> float:
    """Colour Cast Factor (lower = less residual colour cast).

    A high CCF indicates the image is still dominated by a single hue -- the
    failure mode underwater enhancement is supposed to fix. The value is
    computed from the saturation histogram as the fraction of pixels whose
    saturation is far from grey, normalised by the grey-pixel mass.
    """
    import cv2

    hsv = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float64) / 255.0
    v = hsv[:, :, 2].astype(np.float64) / 255.0
    grey_mask = (s < 0.10) & (v > 0.20)
    n_grey = float(grey_mask.sum())
    if n_grey < 1:
        # No grey reference pixels at all: report the mean saturation instead,
        # which is a monotone proxy for cast strength.
        return float(s.mean())
    n_total = float(s.size)
    return float((n_total - n_grey) / n_total)


# ------------------------------------------------------------------- BRISQUE
_BRISQUE_MODEL_ENV = "HYDROBENCH_BRISQUE_MODEL"
_DEFAULT_BRISQUE_MODEL = os.path.expanduser(
    "~/.hydro_bench/models/brisque_svm.yml")


def brisque(image: np.ndarray, model_path: Optional[str] = None) -> Optional[float]:
    """BRISQUE SVR quality score (lower is better), or ``None`` if unavailable.

    Requires the trained BRISQUE model file. HydroBench will not download it;
    obtain ``brisque_svm.yml`` (and the range file if your build needs it)
    from your OpenCV distribution and point ``model_path`` or the
    ``HYDROBENCH_BRISQUE_MODEL`` environment variable at it.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        return None
    # cv2.quality lives in opencv-contrib-python; absent from -headless builds.
    quality = getattr(cv2, "quality", None)
    if quality is None:
        return None
    path = model_path or os.environ.get(_BRISQUE_MODEL_ENV) or _DEFAULT_BRISQUE_MODEL
    if not Path(path).is_file():
        return None
    try:
        brisque_obj = quality.QualityBRISQUE_create(path)
        score = brisque_obj.compute(_to_uint8(image))
        if score is None:
            return None
        # compute() returns a 3-vector (one per channel) for RGB.
        return float(np.mean(np.asarray(score).flatten()))
    except Exception:
        return None


# -------------------------------------------------------------------- MUSIQ
def musiq(image: np.ndarray, model_path: Optional[str] = None) -> Optional[float]:
    """MUSIQ multi-scale image quality score, or ``None`` if unavailable.

    MUSIQ is a learned metric whose weights are distributed separately from
    HydroBench. Point ``model_path`` (or ``HYDROBENCH_MUSIQ_MODEL``) at a
    TorchScript checkpoint to enable it; otherwise it is reported as missing.
    """
    path = model_path or os.environ.get("HYDROBENCH_MUSIQ_MODEL")
    if not path or not Path(path).is_file():
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        model = torch.jit.load(path, map_location="cpu").eval()
        import cv2

        rgb = _to_uint8(image)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            value = model(tensor)
        return float(np.mean(np.asarray(value).flatten()))
    except Exception as exc:  # do not let a bad checkpoint kill the run
        print(f"WARNING: MUSIQ evaluation failed ({exc}); reporting as missing")
        return None


# ------------------------------------------------------------------ registry
def evaluate_nr(image: np.ndarray, metrics: Tuple[str, ...] = ("uciqe", "uiqm",
                                                               "ccf")) -> NRResult:
    """Compute the requested NR metrics for one enhanced image."""
    out = NRResult()
    for metric in metrics:
        if metric == "uciqe":
            out.uciqe = uciqe(image)
        elif metric == "uiqm":
            out.uiqm = uiqm(image)
        elif metric == "uicm":
            out.uicm = _uicm(image)
        elif metric == "uism":
            out.uism = _uism(image)
        elif metric == "uiconm":
            out.uiconm = _uiconm(image)
        elif metric == "ccf":
            out.ccf = ccf(image)
        elif metric == "brisque":
            out.brisque = brisque(image)
        elif metric == "musiq":
            out.musiq = musiq(image)
        else:
            raise ValueError(f"unknown no-reference metric: {metric}")
    return out


AVAILABLE_NR_METRICS: Tuple[str, ...] = ("uciqe", "uiqm", "uicm", "uism",
                                         "uiconm", "ccf", "brisque", "musiq")

# Direction of "better" for each NR metric: +1 means higher is better.
NR_BETTER_DIRECTION: Dict[str, int] = {
    "uciqe": +1,
    "uiqm": +1,
    "uicm": +1,
    "uism": +1,
    "uiconm": +1,
    "ccf": -1,
    "brisque": -1,
    "musiq": +1,
}


def available_nr_metrics() -> Dict[str, bool]:
    """Which NR metrics can actually run in this environment."""
    import cv2

    return {
        "uciqe": True,
        "uiqm": True,
        "uicm": True,
        "uism": True,
        "uiconm": True,
        "ccf": True,
        "brisque": getattr(cv2, "quality", None) is not None
                   and Path(os.environ.get(_BRISQUE_MODEL_ENV)
                            or _DEFAULT_BRISQUE_MODEL).is_file(),
        "musiq": bool(os.environ.get("HYDROBENCH_MUSIQ_MODEL")),
    }
