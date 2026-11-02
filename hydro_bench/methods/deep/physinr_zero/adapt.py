"""Single-image test-time adaptation (methodology §19).

Each test image starts from the meta-initialised parameters, is adapted with
the *unsupervised* L_total, and the adapted parameters are then discarded.
There is no reference image, no batch of other test images, and no carry-over
between images -- the reset at the end of this module is what makes the method
zero-shot rather than fine-tuned.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import physics
from .config import PhysINRConfig
from .losses import LossBreakdown, PhysINRLoss
from .model import PhysINR, PhysOutputs, strided_image
from .utils import resolve_device, seed_everything


@dataclass
class AdaptTrace:
    """Per-image record, kept so the manuscript's diagnostics are real."""

    losses: List[float]
    terms: List[Dict[str, float]]
    final_loss: float
    steps: int
    water_weights: List[float]
    mean_tau: List[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "losses": self.losses,
            "terms": self.terms,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "water_weights": self.water_weights,
            "mean_tau": self.mean_tau,
        }


def load_meta_checkpoint(model: PhysINR, checkpoint: Path) -> Dict[str, object]:
    """Restore meta-initialised parameters into ``model``."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob.get("model", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {checkpoint} does not match the model: "
            f"missing={missing} unexpected={unexpected}"
        )
    return blob


class TestTimeAdapter:
    """Runs the §19 adaptation loop on one image at a time."""

    def __init__(self, cfg: PhysINRConfig, model: PhysINR,
                 device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.device = resolve_device(device if device is not None else cfg.device)
        self.loss_fn = PhysINRLoss(cfg)

    # ------------------------------------------------------------------ API
    def adapt(self, image: np.ndarray, steps: Optional[int] = None,
              return_trace: bool = False) -> Tuple[np.ndarray, Optional[AdaptTrace]]:
        """Enhance a single HxWx3 image in [0, 1].

        Parameters
        ----------
        image : np.ndarray
            HxWx3 float32 in [0, 1].
        steps : int, optional
            Override ``cfg.adapt_steps`` (the ablation study does this).
        return_trace : bool
            Also return the per-step loss trace.
        """
        cfg = self.cfg
        s = steps if steps is not None else cfg.adapt_steps
        # Adaptation samples coordinates at this stride (§21 step 4a); the
        # enhanced image is rendered on the full grid below.
        stride = max(1, cfg.loss_stride)

        # Per-image reset: work on a pristine copy of the meta-initialisation.
        model = copy.deepcopy(self.model).to(self.device)
        model.train()

        groups = model.parameter_groups()
        opt = torch.optim.Adam(groups, lr=0.0)
        for g in groups:
            g["lr"] = cfg.lr_field if g is groups[0] else cfg.lr_extractor

        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # float64 inputs (common when an image is built from numpy meshes)
        # would otherwise collide with the float32 parameters.
        x = x.to(torch.float32).unsqueeze(0).to(self.device)
        # The loss must see the image at the same pixels the field was
        # evaluated at; the two arrays then line up exactly.
        x_loss = strided_image(x, stride)

        trace = AdaptTrace([], [], 0.0, s, [], [])
        for step in range(s):
            opt.zero_grad(set_to_none=True)
            out = model(x, step=step, coord_stride=stride)
            breakdown = self.loss_fn(x_loss, out, step, s)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            self._project(model)

            if return_trace:
                trace.losses.append(float(breakdown.total.detach()))
                trace.terms.append(breakdown.terms)
                with torch.no_grad():
                    trace.water_weights.extend(out.water_weights.mean(0).tolist())
                    trace.mean_tau.append(float(out.tau.mean()))

        # Final rendering on the full coordinate grid (stride 1), so the
        # returned image has exactly the input geometry.
        with torch.no_grad():
            out = model(x, step=s)
            enhanced = out.restored.permute(0, 3, 1, 2).clamp(0.0, 1.0)
            # §E display normalisation: a reference-free per-channel percentile
            # stretch of the RENDERED frame only.  The physical inverse is
            # deliberately conservative (the t_min floor caps the
            # amplification), so the raw render keeps a compressed histogram;
            # supervised competitors learn an equivalent display transform.
            # Applied here -- not inside forward -- so the adaptation loss,
            # the re-degradation cycle, and the physics are untouched and an
            # existing meta-initialisation needs no re-training.  Note the
            # tensor is (B, 3, H, W): transpose to channel-last for the
            # percentile sort, then back.
            if cfg.display_norm:
                enhanced = physics.display_normalize(
                    enhanced.permute(0, 2, 3, 1), cfg.display_p_low,
                    cfg.display_p_high, cfg.display_strength) \
                    .permute(0, 3, 1, 2)

        arr = enhanced[0].float().cpu().numpy().transpose(1, 2, 0)
        trace.final_loss = trace.losses[-1] if trace.losses else float("nan")
        return arr, (trace if return_trace else None)

    # --------------------------------------------------------------- internals
    def _project(self, model: PhysINR) -> None:
        """§19 step 4i: keep the physical parameters inside their valid ranges.

        Every field is bounded by construction (tanh, sigmoid, clamps); what
        is *not* bounded by construction is the raw pre-activation magnitude,
        which can saturate the activations and stall the optimiser.  Clipping
        the buffers keeps the adaptation numerically well behaved without
        altering the parameterisation.
        """
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("depth_logit"):
                    p.clamp_(-6.0, 6.0)
                else:
                    p.clamp_(-8.0, 8.0)


__all__ = ["TestTimeAdapter", "AdaptTrace", "load_meta_checkpoint"]
