"""The PhysINR network: a lightweight local feature extractor followed by a
coordinate-based neural field that outputs the physical parameter fields.

Data flow (methodology §2):

    I -> E_theta(I) -> F -> bilinear sample f(x)
                     -> z(x) = [f(x), gamma(x)] -> MLP -> o(x)
                     -> {tau_c(x), A_c(x), alpha_c(x), w}
                     -> J  -> I_hat

Everything is channel-last inside the field so the physical equations of
``physics.py`` read exactly as they are written in the methodology; the only
NCHW moment is the convolution stack, which is wrapped at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import physics
from .config import PhysINRConfig


# ------------------------------------------------------------------ blocks --
class DepthwiseSeparableBlock(nn.Module):
    """Block(z) = LeakyReLU(Norm(PWConv(DWConv(z)))); methodology §3."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 leaky_slope: float) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        # Instance normalisation: the network is applied to single images with
        # no batch statistics to borrow, so batch norm would be meaningless.
        self.norm = nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(leaky_slope, inplace=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(z))))


class CoordinateMLP(nn.Module):
    """Compact two-hidden-layer MLP; methodology §4."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(z))
        h = self.act(self.fc2(h))
        return self.fc3(h)


# --------------------------------------------------------------- the output --
class PhysOutputs(NamedTuple):
    """Physical fields, all channel-last (B, H, W, 3) unless noted.

    ``tau`` is the optical thickness, ``transmission`` is exp(-tau),
    ``ambient`` the ambient-light field, ``alpha`` the bounded nonlinear
    correction, ``restored`` the enhanced image J, ``regraded`` the
    re-degraded I_hat, ``regraded_plus`` the residual-corrected I_hat^+ and
    ``water_weights`` the soft Jerlov assignment (B, K).
    """

    tau: torch.Tensor
    transmission: torch.Tensor
    tau_jerlov: torch.Tensor          # (B, 3), global Jerlov component
    ambient: torch.Tensor
    alpha: torch.Tensor
    restored: torch.Tensor
    regraded: torch.Tensor
    regraded_plus: torch.Tensor
    water_weights: torch.Tensor       # (B, K)
    tau_delta: torch.Tensor           # local deviation from the Jerlov prior
    ambient_estimate: torch.Tensor    # (B, 3), data-driven background light
    tau_level: Optional[torch.Tensor] = None    # (B,), v3 best-channel tau


# ------------------------------------------------------------------- model --
class PhysINR(nn.Module):
    """Feature extractor + continuous neural field."""

    def __init__(self, cfg: PhysINRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.feature_channels
        s1, s2, s3 = cfg.feature_strides
        self.feature_extractor = nn.Sequential(
            DepthwiseSeparableBlock(3, c1, s1, cfg.leaky_slope),
            DepthwiseSeparableBlock(c1, c2, s2, cfg.leaky_slope),
            DepthwiseSeparableBlock(c2, c3, s3, cfg.leaky_slope),
        )
        self.feature_channels_out = c3

        # Registered buffer, not a parameter: the Jerlov table is a physical
        # constant, not something the network is allowed to learn.
        beta = physics.jerlov_beta_matrix()
        self.register_buffer("jerlov_beta", torch.tensor(beta, dtype=torch.float32))

        # Fourier dimension: sin/cos of both coordinates at L levels.
        self.fourier_dim = 2 * 2 * cfg.fourier_levels
        in_dim = c3 + self.fourier_dim
        # Output: q (3, tau residual), u (3, alpha), a (3, ambient), r (1,
        # water-type logit field) -- ten scalars, as in §4.
        self.out_dim = 10
        self.mlp = CoordinateMLP(in_dim, cfg.mlp_hidden, self.out_dim)
        # The type head's input is the collapse fix: "none" keeps the published
        # one-dimensional bottleneck (the spatial mean of the water-logit
        # field), which a content-blind training run fills with the marginal
        # type distribution; "physics" appends the content descriptor built in
        # forward -- channel means, their log-ratios, the estimated per-channel
        # background light, and the pooled extractor features -- so the
        # assignment is a function of the image.
        self.type_head_features = cfg.type_head_features
        if cfg.type_head_features == "none":
            head_in = 1
        elif cfg.type_head_features == "physics":
            head_in = 1 + 3 + 2 + 3 + c3
        else:
            raise ValueError(
                f"type_head_features must be 'none' or 'physics', "
                f"got {cfg.type_head_features!r}")
        self.jerlov_head = nn.Linear(head_in, cfg.num_water_types)
        # s_d of §6: the effective-depth scale.  The published architecture
        # holds a single learnable logit; the content-dependent one maps the
        # same descriptor the type head reads to a per-image depth, because a
        # constant prior cannot span the tau = beta * d range the tasks cover.
        # Zero-initialised, so training starts at the published semantics
        # (sigmoid(0) = 0.5).
        if cfg.type_head_features == "physics":
            self.depth_head = nn.Linear(head_in, 1)
            nn.init.zeros_(self.depth_head.weight)
            nn.init.zeros_(self.depth_head.bias)
            # No standalone depth_logit in this mode: FOMAML differentiates
            # the outer loss against every parameter, so the model must not
            # carry one the forward never reads.
            self.depth_logit = None
        else:
            self.depth_head = None
            self.depth_logit = nn.Parameter(torch.tensor(0.0))
        # v3: the transmission LEVEL.  The range estimator normalises the best
        # channel to t = 1; real water attenuates even that channel, so the
        # prior under-corrects.  This head predicts the best channel's own
        # transmission from the same content descriptor, supervised in the
        # outer loop on the synthetic task's t_best.  Zero-init so training
        # starts from the v2 behaviour (sigmoid(0) = 0.5) and the head must
        # earn its keep.
        self.level_head = (nn.Linear(head_in, 1)
                           if cfg.level_head and cfg.type_head_features == "physics"
                           else None)
        if self.level_head is not None:
            nn.init.zeros_(self.level_head.weight)
            nn.init.zeros_(self.level_head.bias)

        self._check_param_budget()

    # --------------------------------------------------------------- helpers
    def _check_param_budget(self) -> None:
        n = sum(p.numel() for p in self.parameters())
        if n >= self.cfg.param_budget:
            raise ValueError(
                f"PhysINR has {n} trainable parameters, exceeding the budget of "
                f"{self.cfg.param_budget} imposed by methodology §4/§22. "
                "Reduce mlp_hidden or the feature channels."
            )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_groups(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Two learning-rate levels, methodology §19."""
        cfg = self.cfg
        field_names, extractor_names = cfg.parameter_groups()
        field_params = [p for n, p in self.named_parameters()
                        if any(n.startswith(pre) for pre in field_names)]
        extractor_params = [p for n, p in self.named_parameters()
                            if any(n.startswith(pre) for pre in extractor_names)]
        missing = [n for n, _ in self.named_parameters()
                   if not (any(n.startswith(p) for p in field_names)
                           or any(n.startswith(p) for p in extractor_names))]
        if missing or not field_params or not extractor_params:
            raise RuntimeError(
                f"parameter split is incomplete; unassigned={missing}"
            )
        return ({"params": field_params, "lr": cfg.lr_field},
                {"params": extractor_params, "lr": cfg.lr_extractor})

    # ------------------------------------------------------------- internals
    @staticmethod
    def _grid(h_full: int, w_full: int, stride: int, device,
              dtype) -> Tuple[torch.Tensor, int, int]:
        """Normalised coordinates of the pixels the field is evaluated at.

        Pixel row ``i * stride`` is assigned the coordinate
        ``-1 + 2*i*stride/(H-1)`` -- i.e. the *same* normalisation the
        stride-1 grid would have given it.  A strided forward is therefore the
        restriction of one continuous field to a subset of the pixels, not a
        reparameterised field, which is what makes it legitimate as the
        "sample normalised image coordinates x" of inference step 4a.

        Returns the (h, w, 2) grid and the number of rows/columns it covers.
        """
        h = (h_full + stride - 1) // stride
        w = (w_full + stride - 1) // stride
        iy = torch.arange(h, device=device, dtype=dtype) * stride
        ix = torch.arange(w, device=device, dtype=dtype) * stride
        ys = -1.0 + 2.0 * iy / max(1, h_full - 1)
        xs = -1.0 + 2.0 * ix / max(1, w_full - 1)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1), h, w

    def _fourier(self, coords: torch.Tensor, step: int) -> torch.Tensor:
        """gamma(x) with progressive octave activation; methodology §4.

        Higher frequencies are switched on one octave at a time so the field
        cannot lock onto high-frequency oscillations before the coarse
        solution is established.  ``step`` is the adaptation index.
        """
        ramp = max(1, self.cfg.fourier_ramp)
        feats = []
        for level in range(self.cfg.fourier_levels):
            # An octave is fully on once (level + 1) * ramp steps have elapsed.
            gate = min(max((step - level * ramp) / ramp, 0.0), 1.0)
            freq = float(2 ** level) * torch.pi
            s = torch.sin(freq * coords) * gate
            c = torch.cos(freq * coords) * gate
            feats.append(s)
            feats.append(c)
        return torch.cat(feats, dim=-1)

    @staticmethod
    def _sample(features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Evaluate the feature field on the (h, w) output grid.

        The field is coordinate-based in principle, but the output grid is
        regular and aligned, so evaluating ``f(x)`` at every output pixel is
        exactly a bilinear resampling of the feature map to (h, w).
        ``F.interpolate`` is used rather than ``grid_sample`` because the
        sampler's backward pass is not implemented for MPS, while bilinear
        interpolation is; on CPU the two are numerically identical for a
        regular grid.

        Returns (B, h, w, C): the spatial axes are kept so the caller can index
        the field at the sampled coordinates, which is what makes a strided
        forward the *same* field on a subset of the pixels.
        """
        b, c, _, _ = features.shape
        up = F.interpolate(features, size=(h, w), mode="bilinear",
                           align_corners=False)
        return up.permute(0, 2, 3, 1)

    # ------------------------------------------------------------------ fwd
    def forward(self, image: torch.Tensor, step: int = 0,
                coord_stride: int = 1) -> PhysOutputs:
        """Run the full pipeline on a batch of images.

        Parameters
        ----------
        image : Tensor
            (B, 3, H, W) in [0, 1], NCHW.
        step : int
            Adaptation step, used only for the Fourier and loss schedules.
        coord_stride : int
            Evaluate the field on a strided coordinate grid, per inference
            step 4a ("sample normalised image coordinates x").  The field is
            continuous, so a strided evaluation is an approximation of the
            same function, not a different one; the final image is always
            rendered at stride 1 by the adaptation loop.

        Returns
        -------
        PhysOutputs
        """
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W), got {tuple(image.shape)}")
        if coord_stride < 1:
            raise ValueError(f"coord_stride must be >= 1, got {coord_stride}")

        b, _, h_full, w_full = image.shape
        cfg = self.cfg
        coords, h, w = self._grid(h_full, w_full, coord_stride,
                                  image.device, image.dtype)

        # ---- local features ------------------------------------------------
        # The extractor always runs at full resolution: it is a cheap conv
        # stack whose cost does not depend on the coordinate stride.  Its
        # output is resampled to the full grid once and then *indexed* at the
        # sampled coordinates, so f(x) is a genuine function of position -- a
        # strided forward evaluates the same field on a subset of the pixels.
        feats = self.feature_extractor(image)                  # (B, C, h', w')
        f_full = self._sample(feats, h_full, w_full)           # (B, H, W, C)
        # Pooled extractor features for the type head, taken from the
        # full-resolution grid: global statistics, so like the YIQ gains they
        # are identical at every stride and a strided forward remains the same
        # field on the shared pixels.
        f_pool = f_full.mean(dim=(1, 2))                       # (B, C)
        if coord_stride > 1:
            f_hw = f_full[:, ::coord_stride, ::coord_stride]
        else:
            f_hw = f_full
        f_xy = f_hw.reshape(b, h * w, -1)

        # ---- coordinates + Fourier features --------------------------------
        coords = coords.reshape(1, h * w, 2).expand(b, -1, -1)
        gamma = self._fourier(coords, step)                    # (B, h*w, g)
        z = torch.cat([f_xy, gamma], dim=-1)

        # ---- neural field ---------------------------------------------------
        o = self.mlp(z)                                        # (B, h*w, 10)
        q, u, a, r_field = o.split([3, 3, 3, 1], dim=-1)

        # ---- input statistics (§8/B2, needed before the type head) ---------
        # The input image is strided to match the coordinate grid.
        img_cl = image.permute(0, 2, 3, 1)[:, ::coord_stride, ::coord_stride]
        img_hw = img_cl.reshape(b, h * w, 3)

        # Background light B_inf, estimated per channel from the deep region of
        # the *full-resolution* frame.  Detached: it is a measurement the field
        # is bracketed around and the type head reads, not a quantity the
        # forward pass differentiates through.  Global statistics are computed
        # on the full frame so they do not depend on the coordinate stride.
        with torch.no_grad():
            ambient_estimate = physics.estimate_background_light(
                image.permute(0, 2, 3, 1), cfg.ambient_dark_fraction,
                cfg.ambient_percentile)
            # v2: the prior's LEVEL from the image itself.  range(I_c) =
            # t_c * range(J_c), so the per-channel percentile range gives the
            # transmission directly; the Jerlov table below only shapes the
            # spectral ratio on top of it.  Detached, full-resolution, so it
            # is stride-independent like the other global statistics.
            t_data = physics.estimate_transmission_level(
                image.permute(0, 2, 3, 1))

        # ---- soft Jerlov assignment (§6) -----------------------------------
        r_bar = r_field.mean(dim=1)                            # (B, 1)
        if self.type_head_features == "physics":
            # Content descriptor: pooled water logit, full-resolution channel
            # means, their log-ratios (the attenuation signature), the
            # estimated background light, and the pooled extractor features.
            # The assignment becomes a function of the image instead of a
            # checkpoint constant.
            m = image.permute(0, 2, 3, 1).mean(dim=(1, 2))     # (B, 3)
            log_rg = torch.log((m[:, 0:1] + 1e-4) / (m[:, 1:2] + 1e-4))
            log_gb = torch.log((m[:, 1:2] + 1e-4) / (m[:, 2:3] + 1e-4))
            head_in = torch.cat([r_bar, m, log_rg, log_gb,
                                 ambient_estimate, f_pool], dim=-1)
        else:
            head_in = r_bar
        water = F.softmax(self.jerlov_head(head_in), dim=-1)   # (B, K)
        beta_jerlov = water @ self.jerlov_beta.to(o.dtype)     # (B, 3)
        if self.depth_head is not None:
            d_logit = self.depth_head(head_in)                 # (B, 1)
        else:
            d_logit = self.depth_logit.expand(b, 1)
        d_bar = cfg.depth_max * torch.sigmoid(d_logit)         # (B, 1)
        tau_jerlov = beta_jerlov * d_bar                       # (B, 3)
        # The learned component contributes its spectral shape about the
        # channel mean: its DC is the synthetic-tied offset the supervision
        # trained it on (the reference channel's attenuation, invisible to the
        # range estimator), and carrying that DC onto real frames is what
        # over-amplifies clear water.  Centering is a no-op for the bound
        # margins, which are inter-channel differences.
        if cfg.tau_jerlov_centered and self.type_head_features == "physics":
            tau_jerlov_eff = tau_jerlov - tau_jerlov.mean(dim=1, keepdim=True)
        else:
            tau_jerlov_eff = tau_jerlov
        # v3: rescale the data-driven ratio by the learned level.  The range
        # estimator normalises the best channel to t = 1; the level head
        # predicts that channel's own transmission, restoring the absolute
        # scale the estimator cannot see.  Applied as a multiplicative factor
        # on t (additive on tau) so it moves the level without touching the
        # spectral ratio the estimator already gets right.  Zero-init sigmoid
        # gives 0.5 at start; the head must beat that to change anything.
        # The clamp is deliberately WIDE (not t_min): censoring the level at
        # t_min would truncate the synthetic target -- its mean t_best is 0.26
        # and its low decile is 0.002 -- and bias the head toward the
        # saturation region.  Amplification safety is enforced downstream by
        # the t_min floor on the transmission entering the inverse.
        if self.level_head is not None:
            t_best_prior = torch.sigmoid(self.level_head(head_in)).squeeze(-1)
            level = t_best_prior.clamp(min=1e-3, max=1.0)          # (B,)
            tau_level = -torch.log(level)                          # (B,)
        else:
            tau_level = torch.zeros(b, device=o.device)
        # The prior: data-driven ratio, learned level, learned spectral shape.
        tau_prior = (cfg.tau_data_weight * (-torch.log(t_data.clamp(min=cfg.t_min)))
                     + tau_level.unsqueeze(1) + tau_jerlov_eff)    # (B, 3)

        # ---- optical thickness (§5) -----------------------------------------
        # The prior supplies the physically admissible base; the spatial field
        # may only deviate from it by a bounded amount.
        tau_delta = cfg.delta_max * torch.tanh(q)              # (B, H*W, 3)
        tau = tau_prior.unsqueeze(1) + tau_delta
        tau = tau.clamp(0.0, cfg.tau_max)
        transmission = physics.transmission_from_tau(tau)

        # ---- ambient light (§7) --------------------------------------------
        # The bracket the learned field is sigmoid-mapped into is per channel
        # and centred on the estimate, instead of the fixed global
        # [ambient_min, ambient_max].  The estimate is a *prior*: the window is
        # wide and clipped to the physical range, so the field keeps most of
        # its freedom and the evidence in the image can still overrule it.
        a_global = a.mean(dim=1, keepdim=True)                 # (B, 1, 3)
        if cfg.ambient_adaptive:
            half = cfg.ambient_window
            est = ambient_estimate.unsqueeze(1)                # (B, 1, 3)
            # The bracket's lower clamp is ``ambient_floor_adaptive``, not the
            # fixed-bracket ``ambient_min``: the estimator routinely reads a
            # red background light well below 0.10 on deep coastal frames, and
            # clamping it up to 0.10 injects an ambient the frame does not
            # have -- the inverse then drives the red channel negative
            # wherever the transmission floor binds, deleting red and leaving
            # the cast in place (the visual-check failure).
            floor = getattr(cfg, "ambient_floor_adaptive", cfg.ambient_min)
            lo = (est - half).clamp(floor, cfg.ambient_max)
            hi = (est + half).clamp(floor, cfg.ambient_max)
            hi = torch.maximum(hi, lo + 1e-3)      # never a degenerate window
        else:
            lo = a_global.new_full((b, 1, 3), cfg.ambient_min)
            hi = a_global.new_full((b, 1, 3), cfg.ambient_max)
        ambient_global = lo + (hi - lo) * torch.sigmoid(a_global)
        ambient = ambient_global + cfg.kappa_ambient * torch.tanh(a)

        # The transmission used for the division is floored at cfg.t_min: the
        # inverse amplifies by 1/t, and an over-estimated tau must not be able
        # to saturate the frame (v1's neon wash).  The SAME floored t goes to
        # regrade so the cycle stays exactly consistent.
        t_inv = transmission.clamp(min=cfg.t_min) if cfg.t_min > 0.0 \
            else transmission
        raw = physics.physical_inverse(img_hw, ambient, t_inv, cfg.eps)
        v = raw.clamp(0.0, 1.0)
        alpha = cfg.alpha_max * torch.tanh(u)
        restored_phys = physics.bounded_nonlinear(v, alpha)

        # ---- YIQ chroma alignment (§D) --------------------------------------
        # A parameter-free final stage: the I/Q amplitudes are scaled by gains
        # derived from chroma statistics, which neutralises a residual cast
        # without touching luminance.  The gains are computed on the
        # full-resolution frame -- they are global statistics, which converge
        # far faster than a strided sample of them, so the map stays a pointwise
        # function of position and a strided forward remains the same field on
        # the shared pixels.  Which frame supplies the statistics is a config
        # choice: the input frame carries the cast the water added, the restored
        # frame also carries any cast the model itself introduced.
        if cfg.yiq_align:
            if cfg.yiq_gains_source == "input":
                gain_source = image.permute(0, 2, 3, 1)
            elif cfg.yiq_gains_source == "restored":
                gain_source = restored_phys
            else:
                raise ValueError(
                    f"yiq_gains_source must be 'input' or 'restored', "
                    f"got {cfg.yiq_gains_source!r}")
            gains = physics.yiq_chroma_gains(
                gain_source, cfg.yiq_strength,
                cfg.yiq_chroma_floor, cfg.yiq_sigma_ref)
            restored = physics.yiq_chroma_align(restored_phys,
                                                gains[:, 0], gains[:, 1])
        else:
            restored = restored_phys

        # ---- re-degradation (§9) --------------------------------------------
        # The methodology lists exactly ten field outputs (3+3+3+1), all of
        # which are accounted for above; the residual r~_c of §9 has no
        # dedicated output slot.  We therefore take r~_c = q_c -- the same
        # pre-activation that drives the local departure from the Jerlov prior
        # -- at no extra parameters.  The two are physically coupled: the
        # residual exists to absorb discrepancies the physical model cannot
        # explain, which are largest exactly where the optical-thickness field
        # is being pulled away from the prior.  r_max bounds the coupling.
        # See Assumption A6 in docs/METHODOLOGY_RESOLUTION.md.
        regraded = physics.regrade(restored, t_inv, ambient)
        residual = cfg.residual_max * torch.tanh(q)
        regraded_plus = (regraded + residual).clamp(0.0, 1.0)

        shape = (b, h, w, 3)
        return PhysOutputs(
            tau=tau.reshape(shape),
            transmission=transmission.reshape(shape),
            tau_jerlov=tau_jerlov,
            ambient=ambient.reshape(shape),
            alpha=alpha.reshape(shape),
            restored=restored.reshape(shape),
            regraded=regraded.reshape(shape),
            regraded_plus=regraded_plus.reshape(shape),
            water_weights=water,
            tau_delta=tau_delta.reshape(shape),
            ambient_estimate=ambient_estimate,
            tau_level=tau_level,
        )


def strided_image(image: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """(B,3,H,W) -> (B,h,w,3) channel-last, sampled at the coordinate stride.

    ``PhysINR.forward(..., coord_stride=s)`` evaluates its fields at exactly
    these pixels, so a loss handed this tensor and the model output sees two
    arrays that line up pixel for pixel.
    """
    img = image.permute(0, 2, 3, 1)
    return img if stride <= 1 else img[:, ::stride, ::stride]


__all__ = ["PhysINR", "PhysOutputs", "DepthwiseSeparableBlock", "CoordinateMLP",
           "strided_image"]
