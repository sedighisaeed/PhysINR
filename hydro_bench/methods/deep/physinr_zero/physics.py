"""Physical layer of PhysINR-Zero: Jerlov water-type priors, the underwater
image-formation model, and the sRGB -> CIELAB transform used by the colour
loss.

Everything in this file is optics, not learning.  Two things deserve scrutiny:

1.  ``JERLOV_TABLE`` holds published diffuse attenuation coefficients.  The
    provenance is recorded in ``JERLOV_SOURCE`` and *must* be carried into the
    provenance record of every run, because the soft water-type prior is the
    paper's central physical assumption.  If you replace the table, replace the
    citation too -- a table without a source is a fabrication.

2.  The RGB channels of a camera do not respond to a single wavelength, so
    ``jerlov_beta_matrix`` integrates the table against a Gaussian spectral
    response per channel instead of reading three wavelengths.  The
    single-wavelength interpolation it replaces is retained as
    ``spectral=False`` (the Assumption A3 reading of the table), because the
    ablation study has to be able to reproduce the pre-refactoring numbers.

3.  The attenuation ordering ``beta_R > beta_G >= beta_B`` holds for clear
    oceanic water and *fails* for the turbid coastal types, where blue
    attenuates most (CDOM and phytoplankton absorb short wavelengths first --
    which is exactly why the light that survives is green).  Any constraint
    written on that ordering must therefore take its margins from the table
    itself; ``losses.py`` does.  See ``_ORDERING_CAVEAT`` below.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# ------------------------------------------------------------------ Jerlov --
# Diffuse attenuation coefficient K_d (m^-1) per Jerlov water type and
# wavelength (nm), from Jerlov (1976) Table XXVII, as transcribed in the
# provenance-audited `jerlov` package (Ishibashi, Zenodo 10.5281/zenodo.22855939,
# released 2026-09-20), which cross-checks the Dstl figshare transcription
# (10.6084/m9.figshare.21710252) against three independent transcriptions of
# the primary source, including 143 cells matching a scan of Table XXVII
# exactly.
#
# Edition matters: this is the 1976 classification.  Solonenko & Mobley's
# widely reproduced `Kd0` column is the *1968* edition and differs from these
# values by up to 34% (Jerlov revised 600 nm in 1976; type I rose 45%).  The
# two editions are not interchangeable -- see
# paper/07_provenance/sources.md.
JERLOV_WAVELENGTHS: Tuple[int, ...] = (400, 450, 475, 500, 550, 600, 650, 700)

JERLOV_TABLE: Dict[str, Tuple[float, ...]] = {
    "I":  (0.0280, 0.0190, 0.0180, 0.0270, 0.0630, 0.2350, 0.3600, 0.5600),
    "IA": (0.0380, 0.0260, 0.0250, 0.0320, 0.0670, 0.2400, 0.3700, 0.5700),
    "IB": (0.0510, 0.0360, 0.0330, 0.0420, 0.0720, 0.2450, 0.3750, 0.5800),
    "II": (0.0960, 0.0680, 0.0620, 0.0700, 0.0890, 0.2600, 0.4000, 0.6100),
    "III": (0.1850, 0.1350, 0.1160, 0.1150, 0.1200, 0.2950, 0.4450, 0.6600),
    "1":  (0.5100, 0.2500, 0.1700, 0.1400, 0.1200, 0.3000, 0.4500, 0.6500),
    "3":  (0.7800, 0.3900, 0.2900, 0.2200, 0.1900, 0.3300, 0.4600, 0.7100),
    "5":  (1.1000, 0.5600, 0.4300, 0.3600, 0.3000, 0.4000, 0.5400, 0.8000),
    "7":  (1.6000, 0.8900, 0.7100, 0.5800, 0.4600, 0.4800, 0.6300, 0.9200),
    "9":  (2.4000, 1.6000, 1.2300, 0.9900, 0.6300, 0.6000, 0.7600, 1.1000),
}

# Jerlov's type I falls below the attenuation of *pure sea water* at 9 of 15
# wavelengths, which Austin & Petzold (1986) document and correct: "some of
# the attenuation values for type I oceanic water as given by Jerlov are less
# than the values we suggest for K_w ... We recommend, therefore, that the
# values of K as published by Jerlov be replaced by those in Table VI."  Their
# Table VI for type I (pure-seawater K_w, m^-1), substituted by default:
AUSTIN_PETZOLD_TYPE_I: Tuple[float, ...] = (0.0217, 0.0176, 0.0184, 0.0280,
                                            0.0640, 0.2408, 0.3559, 0.6513)
USE_AUSTIN_PETZOLD_TYPE_I: bool = True

# Order of the keys; ``num_water_types`` in the config must match this length.
JERLOV_TYPES: Tuple[str, ...] = tuple(JERLOV_TABLE.keys())

# Citation for the table above.  Written into the provenance record of every
# run, since the soft water-type prior is the paper's central physical
# assumption.
#
# ``RGB_SPECTRAL_SIGMA_NM`` is defined here because the citation mentions it;
# the Gaussian channel response itself is documented where it is used, below.
RGB_SPECTRAL_SIGMA_NM: float = 30.0

# The ordering caveat recorded in the module docstring, as a string so it can be
# written into a run's provenance next to the table citation.  A constraint that
# assumes red attenuates most is wrong for the coastal water types, and those
# are the ones whose surviving light is green.
_ORDERING_CAVEAT = (
    "The attenuation ordering beta_R > beta_G >= beta_B holds for the clear "
    "oceanic types (I, IA, IB, II) but is violated by the published table for "
    "oceanic III and every coastal type (1, 3, 5, 7, 9), where blue attenuates "
    "more than green.  That is the optical reason turbid coastal water transmits "
    "green light, so a universal red-attenuates-most constraint would fight the "
    "physics on precisely the images the colour-cast correction targets.  "
    "Constraints on the ordering therefore take their margins from the table's "
    "own values rather than imposing a single universal ordering."
)

JERLOV_SOURCE = (
    "Jerlov, N.G. (1976), Marine Optics, Elsevier Oceanography Series 14, "
    "Table XXVII (1976 classification of the diffuse attenuation coefficient), "
    "as transcribed with cell-level provenance in the `jerlov` package "
    "(Ishibashi, Zenodo, doi:10.5281/zenodo.22855939) and the Dstl figshare "
    "dataset doi:10.6084/m9.figshare.21710252, cross-checked against three "
    "independent transcriptions of the primary source.  Oceanic type I uses "
    "the Austin & Petzold (1986, Opt. Eng. 25(3) 471-479) Table VI "
    "pure-seawater substitution, because Jerlov's own type I values lie below "
    "pure-water attenuation at 9 of 15 wavelengths.  See "
    "paper/07_provenance/sources.md for the verification record and the "
    "edition-difference caveat.  Channel attenuation integrates the table "
    "against a Gaussian spectral response (sigma = "
    f"{RGB_SPECTRAL_SIGMA_NM:g} nm) rather than reading three wavelengths.  "
    "Ordering caveat: " + _ORDERING_CAVEAT
)

# Representative wavelength (nm) for each camera channel, computed from the
# CIE 1931 2-deg colour-matching functions and the IEC 61966-2-1 sRGB
# primaries under D65: the peak of each channel's spectral response.  Reading
# K_d at three wavelengths is an acknowledged approximation of the full
# spectrum; it is stated as Assumption A3 in the methodology resolution.
RGB_WAVELENGTHS: Dict[str, int] = {"r": 608, "g": 545, "b": 455}
CHANNELS: Tuple[str, ...] = ("r", "g", "b")

# Effective half-width (nm) of each sRGB primary's spectral response.  A camera
# channel integrates light over a band, not a line, so the attenuation a channel
# *sees* is the table integrated against that band rather than read at one
# wavelength.  30 nm is the half-width of a typical sRGB primary; the Gaussian
# is an approximation of the measured response, which is mildly asymmetric, but
# a symmetric Gaussian is the honest choice when the response is not measured.
# (Defined near ``JERLOV_SOURCE`` above, which cites it.)


def channel_spectral_weights(center_nm: float, sigma_nm: float = RGB_SPECTRAL_SIGMA_NM
                             ) -> np.ndarray:
    """Normalised Gaussian weights over the tabulated wavelengths.

    A channel's effective attenuation is ``sum_l w_l * K_d(lambda_l)`` with
    these weights -- the table integrated against the channel's spectral
    response, at the resolution the table offers.  As ``sigma_nm -> 0`` the
    weights collapse onto the nearest tabulated wavelength, which is the
    behaviour the spectral-parameterisation test pins.
    """
    lam = np.asarray(JERLOV_WAVELENGTHS, dtype=np.float64)
    w = np.exp(-0.5 * ((lam - center_nm) / sigma_nm) ** 2)
    return w / w.sum()


def jerlov_beta_matrix(austin_petzold: bool = USE_AUSTIN_PETZOLD_TYPE_I,
                       spectral: bool = True,
                       sigma_nm: float = RGB_SPECTRAL_SIGMA_NM) -> np.ndarray:
    """(K, 3) matrix of per-channel attenuation per water type, in m^-1.

    Row ``k`` is the (beta_r, beta_g, beta_b) triplet of water type ``k``.

    With ``spectral=True`` (default) each triplet is the Jerlov curve
    integrated against a Gaussian spectral response of width ``sigma_nm``
    centred on the channel's peak wavelength.  With ``spectral=False`` it is
    the curve interpolated at the peak wavelength alone -- the single-
    wavelength reading this replaces, kept so the ablation can reproduce the
    pre-refactoring numbers exactly.
    """
    waves = np.asarray(JERLOV_WAVELENGTHS, dtype=np.float64)
    out = np.zeros((len(JERLOV_TYPES), 3), dtype=np.float64)
    for c, channel in enumerate(CHANNELS):
        lam = RGB_WAVELENGTHS[channel]
        if spectral:
            wt = channel_spectral_weights(lam, sigma_nm)
        else:
            wt = None                                   # interpolate instead
        for k, wtype in enumerate(JERLOV_TYPES):
            curve = JERLOV_TABLE[wtype]
            if austin_petzold and wtype == "I":
                curve = AUSTIN_PETZOLD_TYPE_I
            if spectral:
                out[k, c] = float(np.dot(wt, np.asarray(curve, dtype=np.float64)))
            else:
                out[k, c] = float(np.interp(lam, waves, curve))
    return out


# --------------------------------------------------------------- Lab colour --
# D65 illuminant.
_XYZ_WHITE = torch.tensor([0.95047, 1.0, 1.08883])

# sRGB (linear) -> XYZ, BT.709 primaries.
_RGB2XYZ = torch.tensor([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
])


def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> linear RGB, elementwise."""
    return torch.where(srgb <= 0.04045,
                       srgb / 12.92,
                       ((srgb + 0.055) / 1.055).clamp(min=0.0) ** 2.4)


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    # The cube-root derivative (1/3) t^(-2/3) diverges at zero, which makes the
    # backward pass emit NaN gradients even where the forward value is finite.
    # Flooring the radicand at delta**3 bounds the gradient to 1/(3 delta^2),
    # and values below the floor take the linear branch of the where anyway.
    safe = t.clamp(min=delta ** 3)
    return torch.where(t > delta ** 3, safe ** (1.0 / 3.0),
                       t / (3.0 * delta * delta) + 4.0 / 29.0)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB image to CIELAB.

    Parameters
    ----------
    rgb : Tensor
        (..., 3) in [0, 1], sRGB, channel-last.

    Returns
    -------
    Tensor
        (..., 3) with L* in [0, 100] and a*, b* roughly in [-128, 128].
    """
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected channel-last RGB, got shape {tuple(rgb.shape)}")
    lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    xyz = lin @ _RGB2XYZ.to(device=rgb.device, dtype=rgb.dtype).T
    white = _XYZ_WHITE.to(device=rgb.device, dtype=rgb.dtype)
    f = _f_lab(xyz / white)
    return torch.stack([
        116.0 * f[..., 1] - 16.0,
        500.0 * (f[..., 0] - f[..., 1]),
        200.0 * (f[..., 1] - f[..., 2]),
    ], dim=-1)


# --------------------------------------------------------- image formation --
def transmission_from_tau(tau: torch.Tensor) -> torch.Tensor:
    """t_c(x) = exp(-tau_c(x)); method §5."""
    return torch.exp(-tau.clamp(min=0.0))


def physical_inverse(image: torch.Tensor, ambient: torch.Tensor,
                     transmission: torch.Tensor, eps: float = 1e-4,
                     t_min: float = 0.0) -> torch.Tensor:
    """J~_c = (I_c - A_c (1 - t_c)) / (t_c + eps); method §8.

    All arguments are channel-last (..., 3).

    ``t_min`` floors the transmission used for the division.  The inverse
    amplifies by 1/t, so an over-estimated optical thickness does not merely
    mis-correct a pixel -- it explodes it: at t = 0.33 a mid-tone channel is
    scaled by 3 and most pixels saturate, which is the neon-wash failure the
    v1 outputs show.  A floor of 0.25 caps the amplification at 4x, the
    limit beyond which the signal is dominated by backscatter and nothing
    recoverable remains.  Pass the SAME floored transmission to ``regrade``
    so the cycle stays consistent.
    """
    if t_min > 0.0:
        transmission = transmission.clamp(min=t_min)
    return (image - ambient * (1.0 - transmission)) / (transmission + eps)


def estimate_transmission_level(image: torch.Tensor, p_low: float = 0.01,
                                p_high: float = 0.99,
                                range_floor: float = 0.25) -> torch.Tensor:
    """Per-channel transmission level t_c estimated from the image itself.

    The image formation I = J t + A (1 - t) is affine in J, so the ambient
    wash shifts but does not compress: ``range(I_c) = t_c * range(J_c)``.
    Taking the least-attenuated channel's range as the scene reference gives

        t~_c = range_c / max_c' range_c',   clamped to [range_floor -> 1]

    with the range measured between the ``p_low``/``p_high`` percentiles so
    outliers and noise do not dominate.  A channel whose contrast is a
    quarter of the best channel's is read as t = 0.25 -- attenuated 4x --
    which is exactly the cue the red channel's early extinction carries in
    real water.  The estimator needs no training and no water type: it is
    the data-driven half of the v2 prior, and the Jerlov table is demoted to
    shaping the spectral ratio rather than setting the level.

    ``image`` is (B, H, W, 3) in [0, 1]; returns a detached (B, 3) tensor.
    """
    if not 0.0 <= p_low < p_high <= 1.0:
        raise ValueError(f"need 0 <= p_low < p_high <= 1, got {p_low}, {p_high}")
    b, h, w, _ = image.shape
    flat = image.permute(0, 3, 1, 2).reshape(b, 3, h * w)
    n = flat.shape[-1]
    ordered = torch.sort(flat, dim=-1).values
    lo = ordered[:, :, max(0, int(round(p_low * (n - 1))))]
    hi = ordered[:, :, min(n - 1, int(round(p_high * (n - 1))))]
    spread = (hi - lo).clamp(min=0.0)                    # (B, 3)
    ref = spread.max(dim=1).values                       # (B,)
    # A frame with no contrast at all carries no attenuation signal either --
    # reading it as "maximum attenuation" would make the prior amplify noise.
    has_signal = ref > 1e-3
    t = spread / ref.clamp(min=range_floor).unsqueeze(1)
    return torch.where(has_signal.unsqueeze(1), t, torch.ones_like(t)) \
        .clamp(0.0, 1.0).detach()


def bounded_nonlinear(v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Bounded monotone residual curve of §8.

    ``g(u) = u (1 - u) (2u - 1)`` vanishes at u in {0, 0.5, 1} and is largest
    in the mid-tones, so the correction touches contrast without clipping
    highlights or shadows.
    """
    g = v * (1.0 - v) * (2.0 * v - 1.0)
    return (v + alpha * g).clamp(0.0, 1.0)


def regrade(restored: torch.Tensor, transmission: torch.Tensor,
            ambient: torch.Tensor) -> torch.Tensor:
    """I_c = J_c t_c + A_c (1 - t_c); the re-degradation operator of §9."""
    return restored * transmission + ambient * (1.0 - transmission)


def channel_luminance(image: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance of a channel-last RGB image."""
    if image.shape[-1] != 3:
        raise ValueError(f"expected (..., 3), got {tuple(image.shape)}")
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2])


def gradient_pair(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along each axis, zero-padded to the input shape.

    Returns ``(dy, dx)``: the first element is the difference between
    consecutive rows (vertical), the second the difference between consecutive
    columns (horizontal).  Both carry the input's shape, so they broadcast
    against masks and against each other.
    """
    dy = torch.zeros_like(tensor)
    dx = torch.zeros_like(tensor)
    dy[..., 1:, :] = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx[..., :, 1:] = tensor[..., :, 1:] - tensor[..., :, :-1]
    return dy, dx


# ------------------------------------------------- colour cast diagnostics --
# The cast the refactor targets is a *global* chroma bias, so the quantities
# below are means and spreads of the chroma channels, not per-pixel penalties.
# Measuring them is what makes "the green cast got better" a reportable number
# instead of an eyeball claim.

# sRGB (gamma-encoded) -> YIQ.  YIQ is used rather than YCbCr because its
# chroma axes separate the opponent directions that matter for a green/cyan
# cast: I is the red-cyan opponent, Q the yellow-blue/magenta-green opponent.
_RGB2YIQ = torch.tensor([
    [0.299, 0.587, 0.114],
    [0.5959, -0.2746, -0.3213],
    [0.2115, -0.5227, 0.3112],
])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def rgb_to_yiq(rgb: torch.Tensor) -> torch.Tensor:
    """Channel-last sRGB -> YIQ. Y carries luminance, I and Q the chroma."""
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected (..., 3), got {tuple(rgb.shape)}")
    m = _RGB2YIQ.to(device=rgb.device, dtype=rgb.dtype)
    return rgb @ m.T


def yiq_to_rgb(yiq: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_yiq`."""
    if yiq.shape[-1] != 3:
        raise ValueError(f"expected (..., 3), got {tuple(yiq.shape)}")
    m = _YIQ2RGB.to(device=yiq.device, dtype=yiq.dtype)
    return yiq @ m.T


def yiq_chroma_stats(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-image mean and spread of the I and Q chroma channels.

    Returns two (B, 2) tensors: ``(mu, sigma)`` of (I, Q).  A green/cyan cast
    shows up as a strongly negative mean on both axes; the spread is the
    image's own colourfulness, which is what distinguishes a cast from a
    genuinely colourful scene.
    """
    yiq = rgb_to_yiq(image.clamp(0.0, 1.0))
    chroma = yiq[..., 1:]                                # (B,H,W,2)
    mu = chroma.flatten(1, -2).mean(dim=1)               # (B,2)
    sigma = chroma.flatten(1, -2).std(dim=1)             # (B,2)
    return mu, sigma


def yiq_chroma_gains(image: torch.Tensor, strength: float, floor: float,
                     sigma_ref: float = 0.05,
                     eps: float = 1e-6) -> torch.Tensor:
    """Saturation-aware attenuation gains for the I and Q channels.

    A green/cyan cast is a *bias* of the chroma distribution, so the gain that
    removes it is one that shrinks the chroma channels.  Shrinking uniformly
    would also desaturate legitimately colourful content, so the shrink is
    scaled by how much of the chroma is bias rather than spread::

        g_c = clamp(1 - strength * |mu_c| / max(sigma_c, sigma_ref), floor, 1)

    A scene whose chroma spread exceeds its chroma bias (a coral reef, a red
    fish) keeps its gains near 1; a scene whose chroma is dominated by a bias
    (greenish water) has its gains driven toward ``floor``.  ``sigma_ref`` is
    the spread below which a chroma channel is treated as bias-dominated.

    Returns a (B, 2) tensor of gains for (I, Q), each in ``[floor, 1]``.
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    mu, sigma = yiq_chroma_stats(image)
    divisor = torch.maximum(sigma, torch.full_like(sigma, sigma_ref))
    gains = 1.0 - strength * mu.abs() / (divisor + eps)
    return gains.clamp(floor, 1.0).detach()


def yiq_chroma_align(image: torch.Tensor, gain_i: torch.Tensor,
                     gain_q: torch.Tensor) -> torch.Tensor:
    """Scale the I and Q chroma amplitudes, leaving luminance Y untouched.

    ``gain_i``/``gain_q`` are scalars or per-batch (B,) tensors.  The map is a
    linear scaling of the two chroma axes about the Y axis, so Y is preserved
    exactly and the transform is exactly invertible -- properties a hue-rotation
    or a channel remix cannot offer.  Because gains are <= 1 and the RGB cube is
    convex, every output lies on the segment between the input and its Y-axis
    projection and therefore stays in [0, 1].
    """
    yiq = rgb_to_yiq(image)
    chroma = yiq[..., 1:]
    g = torch.stack([gain_i, gain_q], dim=-1)
    # Broadcast the gains against whatever leading dims the image carries:
    # the restored tensor inside ``PhysINR.forward`` is flattened to (B, N, 3)
    # while a test may hand it (B, H, W, 3).
    while g.dim() < chroma.dim():
        g = g.unsqueeze(-2)
    yiq = torch.cat([yiq[..., :1], chroma * g], dim=-1)
    return yiq_to_rgb(yiq).clamp(0.0, 1.0)


# ----------------------------------------------------- display normalisation --
def display_normalize(image: torch.Tensor, p_low: float = 0.01,
                      p_high: float = 0.99, strength: float = 1.0,
                      eps_span: float = 1e-3) -> torch.Tensor:
    """Per-channel percentile stretch of the rendered output.

    The physical inverse recovers the scene radiance J, but its estimate is
    conservative: the transmission floor and the bounded residual curve cap
    the amplification, so the rendered frame keeps much of its low contrast
    and the blue-green water tint sits in the middle of the histogram instead
    of spanning it.  Supervised methods (UIEC2-Net, Water-Net) effectively
    learn a white balance + contrast stretch on top of their restoration; this
    stage gives the zero-shot pipeline the same *display* transform without a
    reference: each channel's ``[p_low, p_high]`` percentile range is mapped
    linearly to [0, 1], with the percentiles computed per image and per
    channel from the rendered frame itself.

    It is applied at RENDER time only (the final full-resolution grid in
    ``adapt.py``), never inside the adaptation loss and never inside the
    cycle: the reference-free objective and the physics stay exactly as they
    were, and an existing meta-initialisation needs no re-training.  The
    percentiles are detached statistics like the YIQ gains; the map itself is
    an affine per-channel rescale, so a ``strength`` blend stays inside
    [0, 1] for inputs in [0, 1].

    Parameters
    ----------
    image : Tensor
        (H, W, 3), (B, H, W, 3) or (B, N, 3), channel-last, in [0, 1].
    p_low, p_high : float
        The percentiles mapped to 0 and 1 (calibrated on the UIEB
        visual-check frames -- experiments/visual_check/prototype_display.py:
        +1.6 dB mean PSNR and frame-100 contrast 0.19 -> 0.28, the reference's
        own level).
    strength : float
        Blend between the input (0.0) and the full stretch (1.0).
    eps_span : float
        Floor on the percentile range, so a constant channel maps to 0.5
        instead of exploding.
    """
    if image.shape[-1] != 3:
        raise ValueError(f"expected channel-last RGB, got {tuple(image.shape)}")
    if not 0.0 <= p_low < p_high <= 1.0:
        raise ValueError(f"need 0 <= p_low < p_high <= 1, got {p_low}, {p_high}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    single = image.dim() == 3
    batch = image.unsqueeze(0) if single else image
    b = batch.shape[0]
    flat = batch.reshape(b, -1, 3)
    n = flat.shape[1]
    ordered = torch.sort(flat, dim=1).values
    lo = ordered[:, max(0, int(round(p_low * (n - 1)))), :].detach()
    hi = ordered[:, min(n - 1, int(round(p_high * (n - 1)))), :].detach()
    span = (hi - lo).clamp(min=eps_span)
    stretched = ((flat - lo.unsqueeze(1)) / span.unsqueeze(1)).clamp(0.0, 1.0)
    out = (1.0 - strength) * batch + strength * stretched.reshape(batch.shape)
    return out.squeeze(0) if single else out


# ------------------------------------------------- adaptive background light --
def estimate_background_light(image: torch.Tensor, dark_fraction: float = 0.10,
                              percentile: float = 0.90) -> torch.Tensor:
    """Per-channel background light B_inf from the deep-scene region.

    The deep region of an underwater frame is where the water column dominates
    the radiance: the direct signal has expired and what remains is backscatter
    and upwelling, so ``I_c(x) -> A_c`` there.  The region is located by the
    *red* channel, which is the first channel to be extinguished with range in
    most natural waters -- the darkest ``dark_fraction`` of red pixels marks
    water that is optically deep.  Each channel's background light is then a
    high percentile of that channel over the region, which is an estimator of
    the asymptote rather than of the noisy minimum.

    The estimate is a *prior*, not a measurement: it is detached and used to
    bracket the learned ambient field, so the network may still depart from it
    where the evidence in the image disagrees.

    Parameters
    ----------
    image : Tensor
        (B, H, W, 3) in [0, 1], channel-last.
    dark_fraction : float
        Fraction of pixels, by red value, that count as the deep region.
    percentile : float
        Per-channel percentile taken over the deep region.

    Returns
    -------
    Tensor
        (B, 3) background light, detached from the autograd graph.
    """
    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError(f"expected (B,H,W,3), got {tuple(image.shape)}")
    if not 0.0 < dark_fraction < 1.0:
        raise ValueError(f"dark_fraction must be in (0, 1), got {dark_fraction}")
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")

    b, h, w, _ = image.shape
    flat = image.reshape(b, h * w, 3)
    red = flat[..., 0]
    k = max(1, int(round(dark_fraction * h * w)))
    # ``kthvalue`` is unimplemented on MPS, so the deep region is selected by
    # sorting red ascending and taking the first k indices -- a fixed-size
    # gather that ``quantile`` (MPS-supported) can then reduce per channel.
    order = red.argsort(dim=1)
    deep = order[:, :k].unsqueeze(-1).expand(-1, -1, 3)
    region = torch.gather(flat, 1, deep)                 # (B, k, 3)
    return region.quantile(percentile, dim=1).detach()   # (B, 3)


# ------------------------------------------------------- CDOM / gelbstoff --
# Coloured dissolved organic matter absorbs short wavelengths preferentially,
# following an approximately exponential decay with wavelength (Bricaud et al.
# 1981).  In coastal water this is what makes the transmitted light green:
# blue is removed first, red was already removed by the water itself, and green
# is what survives.  Adding it to the synthesiser gives meta-training tasks
# whose degradation has the same spectral signature as the images the cast
# correction must fix.
CDOM_SLOPE_NM: float = 0.015       # spectral slope S, nm^-1


def cdom_absorption(wavelengths, a440: float,
                    slope: float = CDOM_SLOPE_NM) -> np.ndarray:
    """a_cdom(lambda) = a440 * exp(-S (lambda - 440)), in m^-1.

    ``a440`` is the absorption at 440 nm, the conventional reference wavelength
    for gelbstoff.  The curve is monotonically decreasing in wavelength, so it
    always attenuates blue more than red.
    """
    lam = np.asarray(wavelengths, dtype=np.float64)
    return a440 * np.exp(-slope * (lam - 440.0))


__all__ = [
    "JERLOV_TABLE", "JERLOV_TYPES", "JERLOV_WAVELENGTHS", "JERLOV_SOURCE",
    "RGB_WAVELENGTHS", "RGB_SPECTRAL_SIGMA_NM", "CHANNELS",
    "channel_spectral_weights", "jerlov_beta_matrix",
    "rgb_to_lab", "transmission_from_tau", "physical_inverse",
    "bounded_nonlinear", "regrade", "channel_luminance", "gradient_pair",
    "rgb_to_yiq", "yiq_to_rgb", "yiq_chroma_stats", "yiq_chroma_gains",
    "yiq_chroma_align", "estimate_background_light",
    "estimate_transmission_level",
    "cdom_absorption", "CDOM_SLOPE_NM",
    "display_normalize",
]
