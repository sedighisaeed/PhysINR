"""The PhysINR-Zero objective: seven loss groups and the schedule that ramps
them in.

Every term implements a numbered section of the methodology.  The module
returns a *dict* of individual terms alongside the weighted total, because the
ablation study needs per-term numbers and a manuscript claim must be
traceable to the quantity that produced it.

Naming follows the methodology: ``l_recon`` is L_recon, ``w_phys`` is
lambda_phys, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from . import physics
from .config import PhysINRConfig
from .model import PhysOutputs


# --------------------------------------------------------------- utilities --
def _as_2d_batch(t: torch.Tensor) -> torch.Tensor:
    """(B, H, W) -> (B, 1, H, W) for conv2d."""
    if t.dim() != 3:
        raise ValueError(f"expected (B,H,W), got {tuple(t.shape)}")
    return t.unsqueeze(1)


def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _box_window(size: int, device, dtype) -> torch.Tensor:
    """Normalised box filter, usable as a conv kernel for local means."""
    k = torch.full((size, size), 1.0 / (size * size), device=device, dtype=dtype)
    return k


def _local_mean(x: torch.Tensor, size: int) -> torch.Tensor:
    """Mean of (B,H,W) over a size x size window (border: zero padding)."""
    xb = _as_2d_batch(x)
    k = _box_window(size, x.device, x.dtype)
    return F.conv2d(xb, k.unsqueeze(0).unsqueeze(0), padding=size // 2)[:, 0]


def _local_std(x: torch.Tensor, size: int) -> torch.Tensor:
    m = _local_mean(x, size)
    m2 = _local_mean(x * x, size)
    return (m2 - m * m).clamp(min=0.0).sqrt()


def _ssim(x: torch.Tensor, y: torch.Tensor, size: int = 11,
          sigma: float = 1.5) -> torch.Tensor:
    """Wang et al. SSIM with a Gaussian window, on (B,H,W) luminance.

    Used as a *loss term* only.  Reported SSIM numbers come from
    HydroBench's metric implementation, which is a separate, tested code path.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    g = _gaussian_kernel(size, sigma, x.device, x.dtype)
    win = torch.outer(g, g).unsqueeze(0).unsqueeze(0)

    def blur(t):
        return F.conv2d(_as_2d_batch(t), win, padding=size // 2)[:, 0]

    mx, my = blur(x), blur(y)
    mxx, myy, mxy = blur(x * x), blur(y * y), blur(x * y)
    cov = mxy - mx * my
    # Variances must be taken about each image's own local mean. Using
    # mxx + myy - 2*mxy instead collapses to zero for identical inputs (it is
    # a covariance, not a sum of variances), which zeroes the denominator's
    # c2 term and sends SSIM to ~186 on a plain image instead of 1.
    var_x = mxx - mx * mx
    var_y = myy - my * my
    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx * mx + my * my + c1) * (var_x + var_y + c2)
    return (num / den.clamp(min=1e-12)).mean()


def _soft_histogram_entropy(image: torch.Tensor, bins: int, sigma: float,
                            window: int) -> torch.Tensor:
    """Local Shannon entropy of luminance via a Gaussian-bin soft histogram.

    The soft one-hot encoding of every pixel, ``P(x,i) = exp(-(u(x)-c_i)^2 /
    2 sigma^2)`` normalised over bins, is smoothed over a ``window`` x
    ``window`` neighbourhood to give a differentiable local histogram.  The
    entropy of that histogram is H_loc.  Spatially smoothing a per-pixel soft
    one-hot is a local histogram in exactly the sense §14.3 needs, and it costs
    one grouped convolution instead of an unfold.
    """
    centers = torch.linspace(1.0 / (2 * bins), 1.0 - 1.0 / (2 * bins), bins,
                             device=image.device, dtype=image.dtype)
    u = image.clamp(0.0, 1.0).unsqueeze(-1)                    # (B,H,W,1)
    resp = torch.exp(-((centers - u) ** 2) / (2 * sigma ** 2))
    probs = resp / resp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # (B, bins, H, W) for a grouped box convolution over space.
    pb = probs.permute(0, 3, 1, 2)
    k = _box_window(window, image.device, image.dtype)
    hist = F.conv2d(pb, k.unsqueeze(0).unsqueeze(0).expand(bins, 1, window, window),
                    groups=bins, padding=window // 2)
    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-12)
    entropy = -(hist * (hist + 1e-12).log()).sum(dim=1)        # (B,H,W)
    return entropy


# ---------------------------------------------------------------- the loss --
@dataclass
class LossBreakdown:
    """Named terms of L_total, for logging and ablation.

    ``total`` carries the autograd graph; the ``terms`` dict holds detached
    floats.  Keeping both in one object means there is exactly one place where
    the objective is defined -- callers can log the numbers and backpropagate
    the same expression, never two parallel copies.
    """

    total: torch.Tensor
    terms: Dict[str, float]
    weights: Dict[str, float]


class PhysINRLoss:
    """Computes L_total(image, model outputs, step)."""

    # Loss groups whose weights follow the cosine warm-up of §20.
    _SCHEDULED = ("color", "struct", "spatial", "jerlov", "cast", "bound")

    # The nine groups, in the order the total assembles them.  ``cast`` and
    # ``bound`` are the colour-cast refactor's additions; both ramp with the §20
    # schedule like the other regularisers, so they do not fight the physical
    # reconstruction while it is still forming.
    GROUPS = ("phys", "cycle", "color", "struct", "spatial", "jerlov",
              "prior", "cast", "bound")

    def __init__(self, cfg: PhysINRConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------- schedule
    def group_weight(self, group: str, step: int, total_steps: int) -> float:
        """lambda_k^(s) of §20.  ``phys`` and ``cycle`` are held at maximum."""
        w = {
            "phys": self.cfg.w_phys,
            "cycle": self.cfg.w_cycle,
            "color": self.cfg.w_color,
            "struct": self.cfg.w_struct,
            "spatial": self.cfg.w_spatial,
            "jerlov": self.cfg.w_jerlov,
            "prior": self.cfg.w_prior,
            "cast": self.cfg.w_cast,
            "bound": self.cfg.w_bound,
        }[group]
        if group not in self._SCHEDULED or total_steps <= 0:
            return w
        warm = max(1, self.cfg.warmup_steps)
        s = min(step, total_steps - 1)
        # The ramp starts at a small positive floor rather than zero: a zero
        # group weight would silently mask a divergent gradient to NaN
        # (0 * NaN = NaN) and let a numerical fault pass unnoticed.
        ramp = 0.05 + 0.95 * (1.0 - torch.cos(
            torch.tensor(torch.pi * s / (2.0 * warm))).item())
        return w * ramp

    def _water_weight(self, step: int, total_steps: int) -> float:
        """§16: the water-type entropy weight starts small and grows."""
        s = min(step, max(0, total_steps - 1))
        return self.cfg.w_water * (s + 1.0) / max(1, total_steps)

    # ---------------------------------------------------------------- terms
    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps).mean()

    def _range_loss(self, tau: torch.Tensor) -> torch.Tensor:
        """L_range of §11, evaluated on the (already clamped) thickness field.

        The model clamps tau by construction; the penalty is retained because
        it also shapes the pre-clamp field in the region where clamping is
        active, which is where the optimiser would otherwise pile up mass.
        """
        lo = F.relu(-tau)
        hi = F.relu(tau - self.cfg.tau_max)
        return (lo + hi).mean()

    def _grad_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L_grad of §12 on channel-last images."""
        total = pred.new_zeros(())
        for c in range(3):
            for dim in (-2, -1):
                gp = torch.diff(pred[..., c], dim=dim)
                gt = torch.diff(target[..., c], dim=dim)
                total = total + (gp - gt).abs().mean()
        return total / 6.0

    def _edge_loss(self, out: PhysOutputs, image: torch.Tensor) -> torch.Tensor:
        """L_edge of §14.2: align edges where the input edges are reliable."""
        y_j = physics.channel_luminance(out.restored)
        y_i = physics.channel_luminance(image)
        gj, _ = physics.gradient_pair(y_j)
        gi, _ = physics.gradient_pair(y_i)
        q95 = torch.quantile(gi.abs().reshape(-1), self.cfg.edge_quantile).clamp(min=1e-6)
        mask = (gi.abs() / q95).clamp(0.0, 1.0).detach()
        return (mask * (gj - gi).abs()).mean()

    def _entropy_term(self, out: PhysOutputs, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """§14.3: controlled local entropy plus the noise penalty."""
        cfg = self.cfg
        y_j = physics.channel_luminance(out.restored)
        y_i = physics.channel_luminance(image)
        gi_dy, gi_dx = physics.gradient_pair(y_i)

        sigma_loc = _local_std(y_i, cfg.flat_window)
        thresh = torch.quantile(sigma_loc.reshape(-1), 0.75).clamp(min=1e-6)
        # Low-contrast regions get entropy maximisation; saturated ones do not.
        # Both masks are (B, h, w, 1) so they broadcast against the channel-last
        # entropy map; ``sat`` needs the trailing unsqueeze because a mean over
        # the last two axes of a (B,h,w) tensor leaves it 3-D, not 4-D.
        sat = ((y_i < 0.02) | (y_i > 0.98)).float().mean(
            dim=(-2, -1), keepdim=True).unsqueeze(-1)
        low_contrast = (1.0 - (sigma_loc / thresh).clamp(0.0, 1.0)).unsqueeze(-1)
        mask_h = (low_contrast * (1.0 - sat)).detach()

        h_loc = _soft_histogram_entropy(y_j, cfg.entropy_bins,
                                        cfg.entropy_bin_sigma, cfg.entropy_window)
        l_entropy = -(mask_h * h_loc.unsqueeze(-1)).mean()

        # Noise penalty: high-frequency variance may not grow in flat regions.
        # ``gradient_pair`` returns zero-padded full-size differences, so the
        # flat mask broadcasts against them directly.
        gj_dy, gj_dx = physics.gradient_pair(y_j)
        flat = (sigma_loc < torch.quantile(sigma_loc.reshape(-1), 0.25)).float().detach()

        def flat_var(g):
            mu = (flat * g).sum() / flat.sum().clamp(min=1.0)
            return (flat * (g - mu) ** 2).sum() / flat.sum().clamp(min=1.0)

        l_noise = ((flat_var(gj_dy) + flat_var(gj_dx))
                   - (flat_var(gi_dy) + flat_var(gi_dx))).clamp(min=0.0)
        return {"entropy": l_entropy, "noise": l_noise}

    def _tv_edge_aware(self, field: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Edge-aware total variation of §15.

        The weighting is applied per gradient direction, since the two
        difference operators have different shapes; ``e^{-kappa |grad I|}`` is
        evaluated on the matching directional derivative of the input
        luminance.
        """
        kappa = self.cfg.tv_edge_kappa
        y_i = physics.channel_luminance(image)
        wi_dy, wi_dx = physics.gradient_pair(y_i)
        fy, fx = physics.gradient_pair(field)
        return (torch.exp(-kappa * wi_dy.abs()) * fy.abs()).mean() \
            + (torch.exp(-kappa * wi_dx.abs()) * fx.abs()).mean()

    def _curvature(self, field: torch.Tensor) -> torch.Tensor:
        """Weak second-order smoothness, §15."""
        gx, gy = physics.gradient_pair(field)
        gxx, _ = physics.gradient_pair(gx)
        gyy, _ = physics.gradient_pair(gy)
        return (gxx.abs() + gyy.abs()).mean()

    # --------------------------------------------- colour-cast refactor terms
    # The three regularisers the refactor adds, plus the ambient prior.  All
    # four are properties of the *restored* image or the *predicted* physics,
    # never of a reference: they have to be, since adaptation is reference-free.
    # They are documented against the failure mode each one closes rather than
    # against a methodology section, because the methodology predates them.

    def _lab_cast_loss(self, lab: torch.Tensor) -> torch.Tensor:
        """L_lab: penalise a global chroma bias in CIELAB.

        Parameters
        ----------
        lab : Tensor
            (..., 3) CIELAB of the restored image; only the a*/b* channels are
            used, so the caller passes the tensor it already computed for the
            §13 colour group instead of converting a second time.

        Notes
        -----
        ``lab_mode = "symmetric"`` is |mu(a*)| + |mu(b*)|: it penalises a cast
        in any direction, red included.  ``"signed"`` penalises only the
        green/blueward half of each opponent axis -- a* < 0 is green, b* < 0 is
        blue, so a green/cyan cast drives both negative and a warm cast is left
        alone.  The signed mode is the default because a neon-green shift is the
        failure this term exists for; the symmetric mode is a config flip.

        Both modes are means over the image, so a mean chroma that is small
        because the scene is genuinely colourful is penalised as much as a cast
        of the same magnitude.  The weight is therefore deliberately small and
        the YIQ alignment stage -- which is saturation-aware -- is what spares
        colourful content.
        """
        mu_a = lab[..., 1].mean()
        mu_b = lab[..., 2].mean()
        mode = self.cfg.lab_mode
        if mode == "symmetric":
            return mu_a.abs() + mu_b.abs()
        if mode == "signed":
            # relu(-x) = max(0, -x): the green/blueward direction only.
            return torch.relu(-mu_a) + torch.relu(-mu_b)
        raise ValueError(f"lab_mode must be 'symmetric' or 'signed', got {mode!r}")

    def _color_constancy_loss(self, restored: torch.Tensor) -> torch.Tensor:
        """L_cc: gray-world channel balance of the restored image.

        sum over channel pairs of |mu(I^p) - mu(I^q)|, per image, averaged over
        the batch.  The gray-world assumption is not physically true
        underwater -- the water column is spectrally selective by definition --
        but the *restored* image is the scene as it would have been photographed
        through little water, where the illuminant is broadband, so a weak
        balance prior on J is defensible.  Weak is the operative word: the
        weight is small, and the term is a tie-breaker between solutions the
        reconstruction loss alone cannot separate.
        """
        mu = restored.mean(dim=(1, 2))                       # (B, 3)
        total = mu.new_zeros(())
        for p, q in ((0, 1), (0, 2), (1, 2)):
            total = total + (mu[:, p] - mu[:, q]).abs().mean()
        return total

    def _attenuation_bound_loss(self, out: PhysOutputs) -> torch.Tensor:
        """L_bound: constrain the channel ordering of the optical thickness.

        The freedom to violate a physical ordering lives in ``tau_delta``, the
        per-pixel departure from the Jerlov prior, not in ``tau_jerlov`` --
        that is a convex combination of table rows and cannot depart from the
        table at all.  The penalty is therefore applied to the effective
        per-pixel thickness ``tau``.

        With ``bound_margin_from_table`` the constraint follows the ordering the
        *assigned water type* itself exhibits.  Optical thickness scales with
        range (``tau_c = beta_c * d``), so the physically meaningful invariant is
        the *sign* of the inter-channel gap, not its magnitude: a pixel is in
        violation when ``tau_p - tau_q`` falls on the opposite side of zero from
        ``beta_p - beta_q``.  The hinge is
        ``relu(-sign(m) * gap - |m|)`` -- the ordering may be reversed by up to
        the table's own gap at unit optical depth, and no more.  For the turbid
        coastal types that is the *opposite* ordering to the clear-water one,
        which is the point: the term enforces the physics of the water the model
        assigned, not a guess about it.  With the flag off the margins vanish and
        the term reduces to the universal clear-water ordering
        ``tau_r > tau_g >= tau_b``.
        """
        tau_r, tau_g, tau_b = out.tau[..., 0], out.tau[..., 1], out.tau[..., 2]
        b = out.tau.shape[0]
        # Detached: the margin is a target set by the prior, not a quantity the
        # loss may minimise by moving the water-type assignment.
        if self.cfg.bound_margin_from_table:
            tj = out.tau_jerlov.detach()
            margins = (tj[:, 0] - tj[:, 1], tj[:, 0] - tj[:, 2], tj[:, 1] - tj[:, 2])
        else:
            zero = out.tau.new_zeros(b, 3)
            margins = (zero[:, 0], zero[:, 1], zero[:, 2])
        gaps = (tau_r - tau_g, tau_r - tau_b, tau_g - tau_b)
        # The written formula in the design spec lists only the two R comparisons;
        # the third pair carries the "G >= B" half of the stated ordering.
        total = tau_r.new_zeros(())
        for gap, margin in zip(gaps, margins):
            sign = (torch.ones_like(margin) if not self.cfg.bound_margin_from_table
                    else torch.sign(margin))
            tol = margin.abs().view(b, 1, 1)
            total = total + torch.relu(-sign.view(b, 1, 1) * gap - tol).mean()
        return total

    def _ambient_prior_loss(self, out: PhysOutputs) -> torch.Tensor:
        """Pull the learned ambient field toward the estimated background light.

        The estimate is computed in the model from the red-deepest region of
        the frame and carried in ``out.ambient_estimate``; reading it here
        (rather than re-estimating) is what keeps the prior the loss pulls
        toward identical to the bracket the field is mapped into, so the two
        cannot disagree about what the background light is.
        """
        mu_ambient = out.ambient.mean(dim=(1, 2))            # (B, 3)
        return (mu_ambient - out.ambient_estimate).abs().mean()

    # ------------------------------------------------------------------ call
    def __call__(self, image: torch.Tensor, out: PhysOutputs, step: int,
                 total_steps: int,
                 weights: Optional[Dict[str, float]] = None) -> LossBreakdown:
        """Weighted total loss.

        Parameters
        ----------
        image : Tensor
            (B,H,W,3) in [0,1], channel-last.
        out : PhysOutputs
            The model's physical outputs, at the same resolution as ``image``.
        step, total_steps : int
            Position in the adaptation schedule; only used to pick the group
            weights.
        weights : dict, optional
            Override the group weights outright.  Supplying one fixed dict at
            several checkpoints makes the loss comparable across steps or
            across configurations, which the schedule otherwise forbids: the
            ramp deliberately changes the objective as adaptation proceeds.
        """
        cfg = self.cfg
        if image.shape != out.restored.shape:
            raise ValueError("image and restored output must match in shape")

        terms: Dict[str, torch.Tensor] = {}
        w: Dict[str, float] = {}

        # ---- §11 physical model loss ---------------------------------------
        l_recon = self._charbonnier(out.regraded_plus - image, cfg.charbonnier_eps)
        l_range = self._range_loss(out.tau)
        terms["recon"] = l_recon
        terms["range"] = l_range
        l_phys = l_recon + cfg.w_range * l_range

        # ---- §12 cycle consistency ------------------------------------------
        l_char = self._charbonnier(out.regraded - image, cfg.charbonnier_eps)
        l_grad = self._grad_loss(out.regraded, image)
        terms["char"] = l_char
        terms["grad"] = l_grad
        l_cycle = l_char + cfg.w_grad * l_grad

        # ---- §13 colour balance ---------------------------------------------
        lab_j = physics.rgb_to_lab(out.restored)
        lab_i = physics.rgb_to_lab(image)
        # Adaptive prior: chroma statistics of the input, reweighted toward the
        # high-transmission regions where the water column contributes least.
        # The mask derives from the predicted transmission, hence from the soft
        # Jerlov assignment; it is detached so the prior is a target, not a
        # quantity the loss can minimise by moving the mask.
        t = out.transmission.mean(dim=-1)
        thresh_t = torch.quantile(t.reshape(-1), 0.90).clamp(min=1e-6)
        mask_t = (t >= thresh_t).float().detach()
        n_t = mask_t.sum().clamp(min=1.0)
        ht_a = (mask_t * lab_i[..., 1]).sum() / n_t
        ht_b = (mask_t * lab_i[..., 2]).sum() / n_t
        blend = 0.7
        mu_a_prior = blend * ht_a + (1 - blend) * lab_i[..., 1].mean()
        mu_b_prior = blend * ht_b + (1 - blend) * lab_i[..., 2].mean()
        mu_a_prior = mu_a_prior.detach()
        mu_b_prior = mu_b_prior.detach()

        l_mean = ((lab_j[..., 1].mean() - mu_a_prior).abs()
                  + (lab_j[..., 2].mean() - mu_b_prior).abs())
        l_std = ((lab_j[..., 1].std() - lab_i[..., 1].std()).abs()
                 + (lab_j[..., 2].std() - lab_i[..., 2].std()).abs())
        terms["color_mean"] = l_mean
        terms["color_std"] = l_std
        l_color = l_mean + cfg.w_std * l_std

        # ---- §14 structure --------------------------------------------------
        y_j = physics.channel_luminance(out.restored)
        y_i = physics.channel_luminance(image)
        l_ssim = 1.0 - _ssim(y_j, y_i)
        l_edge = self._edge_loss(out, image)
        ent = self._entropy_term(out, image)
        terms["ssim"] = l_ssim
        terms["edge"] = l_edge
        terms["entropy"] = ent["entropy"]
        terms["noise"] = ent["noise"]
        l_struct = (cfg.w_ssim * l_ssim + cfg.w_edge * l_edge
                    + cfg.w_entropy * ent["entropy"]
                    + cfg.w_noise * ent["noise"])

        # ---- §15 spatial smoothness ----------------------------------------
        tv_tau = sum(self._tv_edge_aware(out.tau[..., c], image) for c in range(3)) / 3.0
        tv_ambient = sum(self._tv_edge_aware(out.ambient[..., c], image)
                         for c in range(3)) / 3.0
        tv_alpha = sum(self._tv_edge_aware(out.alpha[..., c], image)
                       for c in range(3)) / 3.0
        curv = sum(self._curvature(out.tau[..., c]) for c in range(3)) / 3.0
        terms["tv_tau"] = tv_tau
        terms["tv_ambient"] = tv_ambient
        terms["tv_alpha"] = tv_alpha
        terms["curv"] = curv
        l_spatial = tv_tau + tv_ambient + cfg.w_alpha_tv * tv_alpha + cfg.w_curv * curv

        # ---- §16 Jerlov prior ----------------------------------------------
        tau_bar = out.tau.mean(dim=(1, 2))                  # (B, 3)
        l_base = (tau_bar - out.tau_jerlov).abs().mean()
        l_dev = out.tau_delta.abs().mean()
        ww = self._water_weight(step, total_steps)
        l_water = -(out.water_weights * (out.water_weights + 1e-12).log()).sum(-1).mean()
        terms["jerlov_base"] = l_base
        terms["jerlov_dev"] = l_dev
        terms["jerlov_water"] = l_water
        l_jerlov = l_base + cfg.w_dev * l_dev + ww * l_water

        # ---- §17 constraints against trivial solutions ---------------------
        l_identity = torch.relu(cfg.identity_delta - (y_j - y_i).abs().mean())
        # Margins come from the Jerlov prior itself (detached), so the ordering
        # is imposed by the optics of the assigned water type, not a constant.
        margin_rg = (out.tau_jerlov[:, 0] - out.tau_jerlov[:, 1]).mean().detach()
        margin_gb = (out.tau_jerlov[:, 1] - out.tau_jerlov[:, 2]).mean().detach()
        tau_r, tau_g, tau_b = out.tau[..., 0], out.tau[..., 1], out.tau[..., 2]
        l_ratio = (torch.relu(tau_r - tau_g + margin_rg).mean()
                   + torch.relu(tau_g - tau_b + margin_gb).mean())
        l_curve = sum(out.alpha[..., c].abs().mean() for c in range(3)) / 3.0
        l_magnitude = (out.alpha ** 2).mean()
        terms["identity"] = l_identity
        terms["ratio"] = l_ratio
        terms["curve"] = l_curve
        terms["magnitude"] = l_magnitude
        l_prior = (l_identity + cfg.w_ratio * l_ratio
                   + cfg.w_curve * l_curve + cfg.w_magnitude * l_magnitude)

        # ---- colour-cast refactor: the cast and bound groups ----------------
        # L_lab reuses the CIELAB conversion of §13 rather than converting the
        # restored image a second time; both groups read the same output.
        l_lab = self._lab_cast_loss(lab_j)
        l_cc = self._color_constancy_loss(out.restored)
        terms["lab_cast"] = l_lab
        terms["cc"] = l_cc
        l_cast = cfg.w_lab * l_lab + cfg.w_cc * l_cc

        l_order = self._attenuation_bound_loss(out)
        l_ambient = self._ambient_prior_loss(out)
        terms["bound_order"] = l_order
        terms["ambient_prior"] = l_ambient
        l_bound = (cfg.w_bound_order * l_order
                   + cfg.w_ambient_prior * l_ambient)

        # ---- §10 weighted total with the §20 schedule ----------------------
        if weights is not None:
            w = {g: float(weights[g]) for g in self.GROUPS}
        else:
            w = {g: self.group_weight(g, step, total_steps) for g in self.GROUPS}
        total = (w["phys"] * l_phys + w["cycle"] * l_cycle
                 + w["color"] * l_color + w["struct"] * l_struct
                 + w["spatial"] * l_spatial + w["jerlov"] * l_jerlov
                 + w["prior"] * l_prior + w["cast"] * l_cast
                 + w["bound"] * l_bound)

        return LossBreakdown(
            total=total,
            terms={k: float(v.detach()) for k, v in terms.items()},
            weights=w,
        )


__all__ = ["PhysINRLoss", "LossBreakdown"]
