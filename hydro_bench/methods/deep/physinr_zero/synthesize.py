"""Physically sampled synthetic underwater degradations for meta-training.

Methodology §18 lists the factors a task is drawn from: Jerlov water type,
channel-dependent attenuation, effective depth, ambient illumination, colour
shift, noise and scattering, local optical-thickness variation, contrast
degradation and blur.  This module turns that list into a sampler.

The generator is deliberately a *sampler*, not a hand-tuned pipeline: every
factor is drawn from a distribution, so the meta-initialisation sees a broad
family rather than one synthetic look.  Crucially, only the *clean* source
image is real; the degradation, and therefore the training signal, is fully
synthetic.  No real underwater image ever enters meta-training.

Two factors are weighted toward the failure this refactor targets.  The
coastal Jerlov types (1,3,5,7,9) -- heavy backscattering and absorption from
phytoplankton and organic matter -- are sampled more often than a uniform draw
over the ten types would, and a per-task CDOM absorption models the gelbstoff
that removes blue light first and leaves the surviving radiance green.  Both
are configured, so the pre-refactoring task distribution stays reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from . import physics
from .config import PhysINRConfig


@dataclass
class TaskParams:
    """One sampled degradation task."""

    water_type: str
    beta: Tuple[float, float, float]      # per-channel attenuation, m^-1
    depth_scale: float                    # effective depth, m
    ambient: Tuple[float, float, float]   # asymptotic ambient light A_inf
    color_gain: Tuple[float, float, float]
    color_bias: Tuple[float, float, float]
    noise_sigma: float
    blur_sigma: float
    local_tau: float                      # amplitude of spatial thickness variation
    # CDOM (gelbstoff) absorption at 440 nm, m^-1.  Defaults to zero so a task
    # constructed by hand describes water without it.
    cdom_a440: float = 0.0
    cdom_slope: float = 0.015
    seed: int = 0
    # Index of the drawn type into the synthesiser's type list, and the soft
    # mixture the effective attenuation was built from.  Meta-training knows
    # both, so the outer loop can supervise the water-type head against them;
    # defaults describe a hand-built task and keep ``replace()`` working.
    water_type_idx: int = 0
    type_mix: Optional[Tuple[float, ...]] = None


def smooth_field(shape: Tuple[int, int], rng: np.random.Generator,
                 octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """A smooth random scalar field in [0, 1] built from low-pass noise.

    Used as the effective-depth proxy: real bathymetry is smooth at image
    scale, so a band-limited field is the honest stand-in for the unknown
    depth map d(x).
    """
    h, w = shape
    field = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    for octave in range(octaves):
        size = max(2, 2 ** octave)
        low = rng.random((max(2, h // size), max(2, w // size)),
                         dtype=np.float32)
        # Bilinear upsample to full resolution: cheap low-pass filter.
        yy = np.linspace(0, low.shape[0] - 1, h, dtype=np.float32)
        xx = np.linspace(0, low.shape[1] - 1, w, dtype=np.float32)
        y0 = np.floor(yy).astype(int)
        x0 = np.floor(xx).astype(int)
        y1 = np.minimum(y0 + 1, low.shape[0] - 1)
        x1 = np.minimum(x0 + 1, low.shape[1] - 1)
        fy = (yy - y0)[:, None]
        fx = (xx - x0)[None, :]
        up = (low[y0][:, x0] * (1 - fy) * (1 - fx)
              + low[y1][:, x0] * fy * (1 - fx)
              + low[y0][:, x1] * (1 - fy) * fx
              + low[y1][:, x1] * fy * fx)
        field += amplitude * up
        total += amplitude
        amplitude *= persistence
    field /= max(total, 1e-12)
    return (field - field.min()) / max(field.max() - field.min(), 1e-6)


class SyntheticDegradation:
    """Samples task parameters and applies them to clean images."""

    def __init__(self, cfg: PhysINRConfig) -> None:
        self.cfg = cfg
        self.beta_table = physics.jerlov_beta_matrix(
            spectral=cfg.use_spectral_beta,
            sigma_nm=cfg.spectral_sigma_nm)      # (K, 3)
        self.types = list(physics.JERLOV_TYPES)
        # The coastal types (1,3,5,7,9) are where the green/cyan cast comes
        # from, and a uniform draw over the ten sees them 50% of the time.
        # Weighting them up is what makes the meta-initialisation practise the
        # degradation family it is being corrected for.
        coastal = {"1", "3", "5", "7", "9"}
        w = np.array([1.0 + cfg.synth_coastal_weight if t in coastal else 1.0
                      for t in self.types])
        self.type_weights = w / w.sum()

    # ------------------------------------------------------------- sampling
    def sample_task(self, rng: np.random.Generator, shape: Tuple[int, int],
                    image_stats: np.ndarray) -> TaskParams:
        """Draw one task.  ``image_stats`` is the per-channel mean of the clean
        image; the ambient light is drawn relative to it so the synthetic
        degradation is not trivially invertible from global statistics."""
        cfg = self.cfg
        k = int(rng.choice(len(self.types), p=self.type_weights))
        beta = np.asarray(self.beta_table[k], dtype=np.float64).copy()

        # Channel-dependent attenuation curves: jitter the Jerlov triplet
        # multiplicatively, preserving the (weak) ordering structure.
        jitter = np.exp(rng.normal(0.0, cfg.synth_beta_jitter, size=3))
        beta = beta * jitter
        # Water type may also mix: sample a soft mixture around the drawn type.
        mix = np.exp(rng.normal(0.0, 0.5, size=len(self.types)))
        mix[k] += 2.0
        mix /= mix.sum()
        beta = 0.5 * beta + 0.5 * (mix[:, None] * self.beta_table).sum(0)

        # CDOM / gelbstoff: coloured dissolved organic matter from phytoplankton
        # and decaying organic matter absorbs short wavelengths preferentially
        # (a(lambda) = a440 exp(-S(lambda-440))).  Added to the attenuation per
        # channel it removes blue first, which is the spectral signature of the
        # green cast -- the degradation family the meta-initialisation must
        # learn to undo.  Drawn per task, so the pool spans clear water (a440=0)
        # to turbid coastal water.
        a440 = float(rng.uniform(*cfg.synth_cdom_a440))
        cdom = physics.cdom_absorption(
            [physics.RGB_WAVELENGTHS[c] for c in physics.CHANNELS],
            a440, cfg.synth_cdom_slope)
        beta = beta + cdom

        depth = float(rng.uniform(*cfg.synth_depth_range))
        a_lo, a_hi = cfg.synth_ambient_range
        # Ambient light: asymptotic value of the backscatter+upwelling term.
        # Physically its colour is the water's own transmitted colour -- the
        # reason deep coastal water is green and deep oceanic water blue -- so
        # with ``synth_ambient_tied`` the per-channel colour is derived from
        # the drawn type's attenuation and only the brightness is random.  A
        # uniform per-channel draw instead decorrelates the water colour from
        # the type, which makes the type unidentifiable from the image and
        # collapses any type head to the sampling marginal.
        if cfg.synth_ambient_tied:
            colour = np.exp(-beta * cfg.synth_ambient_depth_ref)
            colour = colour / colour.sum()
            jitter = np.exp(rng.normal(0.0, cfg.synth_ambient_jitter, size=3))
            brightness = float(rng.uniform(a_lo, a_hi))
            ambient = brightness * 3.0 * colour * jitter
        else:
            green_bias = float(rng.uniform(0.0, 0.25)) * (1.0 / (1.0 + beta[1]))
            ambient = np.array([
                float(rng.uniform(a_lo, a_hi)) * (1.0 - green_bias),
                float(rng.uniform(a_lo, a_hi)) * (1.0 + green_bias),
                float(rng.uniform(a_lo, a_hi)) * (1.0 - 0.5 * green_bias),
            ])
        ambient = np.clip(ambient, 0.0, 1.0)
        gain = 1.0 + rng.normal(0.0, cfg.synth_color_shift, size=3)
        bias = rng.normal(0.0, cfg.synth_color_shift, size=3)
        return TaskParams(
            water_type=self.types[k],
            water_type_idx=k,
            type_mix=tuple(float(x) for x in mix),
            beta=tuple(float(x) for x in beta),
            depth_scale=depth,
            ambient=tuple(float(x) for x in ambient),
            color_gain=tuple(float(x) for x in gain),
            color_bias=tuple(float(x) for x in bias),
            noise_sigma=float(rng.uniform(*cfg.synth_noise_sigma)),
            blur_sigma=float(rng.uniform(*cfg.synth_blur_sigma)),
            local_tau=float(rng.uniform(0.0, cfg.synth_local_tau)),
            cdom_a440=a440,
            cdom_slope=cfg.synth_cdom_slope,
            seed=int(rng.integers(0, 2 ** 31 - 1)),
        )

    # ------------------------------------------------------------- degrade
    def degrade(self, clean: torch.Tensor, task: TaskParams,
                rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a task to a clean image.

        Parameters
        ----------
        clean : Tensor
            (B, 3, H, W) in [0, 1].

        Returns
        -------
        degraded, clean : Tensor, Tensor
            The synthetic underwater observation and its (colour-shifted)
            clean counterpart, which the meta-loss supervises against.
        """
        b, _, h, w = clean.shape
        device, dtype = clean.device, clean.dtype

        # Colour shift applied to the clean image first: the "clear" image a
        # camera would have taken is not identical to the source photograph.
        gain = torch.tensor(task.color_gain, device=device, dtype=dtype)
        bias = torch.tensor(task.color_bias, device=device, dtype=dtype)
        clean_shift = (clean * gain.view(1, 3, 1, 1) + bias.view(1, 3, 1, 1)
                       ).clamp(0.0, 1.0)

        # Optical thickness: global component plus a smooth spatial variation.
        beta = torch.tensor(task.beta, device=device, dtype=dtype)
        field = torch.from_numpy(
            smooth_field((h, w), rng)).to(device=device, dtype=dtype)
        tau = (beta.view(1, 3, 1, 1)
               * (task.depth_scale * (1.0 - task.local_tau
                                      + 2.0 * task.local_tau * field))
               .view(1, 1, h, w))
        tau = tau.clamp(0.0, 10.0)
        t = torch.exp(-tau)

        # Ambient light grows toward its asymptote as the range increases,
        # which is the physical behaviour the image-formation model encodes.
        a_inf = torch.tensor(task.ambient, device=device, dtype=dtype)
        ambient = a_inf.view(1, 3, 1, 1) * (1.0 - t)

        degraded = clean_shift * t + ambient * (1.0 - t)

        if task.blur_sigma > 1e-3:
            # Separable Gaussian: two 1-D passes, which is k+multiply-adds
            # instead of k^2.  The two passes need differently-shaped kernels,
            # so the vector is re-viewed rather than expanded into place.
            k = int(2 * round(3 * task.blur_sigma) + 1)
            g = torch.arange(k, device=device, dtype=dtype) - k // 2
            g = torch.exp(-(g ** 2) / (2 * task.blur_sigma ** 2))
            g = g / g.sum()
            kh = g.view(1, 1, 1, k).expand(3, 1, 1, k)
            kv = g.view(1, 1, k, 1).expand(3, 1, k, 1)
            degraded = torch.nn.functional.conv2d(
                degraded, kh, padding=(0, k // 2), groups=3)
            degraded = torch.nn.functional.conv2d(
                degraded, kv, padding=(k // 2, 0), groups=3)

        if task.noise_sigma > 1e-4:
            noise = torch.randn_like(degraded) * task.noise_sigma
            degraded = degraded + noise

        return degraded.clamp(0.0, 1.0), clean_shift


__all__ = ["SyntheticDegradation", "TaskParams", "smooth_field"]
