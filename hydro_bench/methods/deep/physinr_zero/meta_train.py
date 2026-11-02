"""Meta-initialisation (methodology §18).

The meta-parameter theta_0 is trained so that a handful of unsupervised
adaptation steps on a *new* image already land near a good solution.  Tasks
are synthetic: a clean photograph is degraded by a physically sampled model,
the inner loop adapts on the degraded image with the reference-free L_total,
and the outer loss scores the result against the synthetic clean image.

Only the clean photographs are real.  No real underwater image, paired or
unpaired, is used at any point -- the degradation and the supervision are both
synthetic by construction.  The outer loop is first-order MAML (FOMAML): the
inner adaptation path is detached, which is the standard approximation when
the inner loop takes optimiser steps rather than plain gradients.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import PhysINRConfig
from .losses import PhysINRLoss
from .model import PhysINR, strided_image
from .synthesize import SyntheticDegradation
from .utils import (get_logger, load_image, resolve_device, seed_everything,
                    to_tensor, write_provenance)


@dataclass
class MetaRecord:
    """Provenance for one meta-training run."""

    config: Dict[str, object]
    param_count: int
    tasks_completed: int
    duration_seconds: float
    seed: int
    source_images: int
    dedup: Dict[str, object] = field(default_factory=dict)
    final_meta_loss: float = float("nan")
    # Training-time top-1 accuracy of the type head against the drawn type,
    # over the last log window.  Chance is ~1/K; because type and depth are
    # confounded in a single frame, even a good head tops out well below 1.0
    # -- the tau error next to it is the primary health signal.
    type_accuracy: float = float("nan")
    # Mean absolute residual of log tau_jerlov against the task's true
    # tau = beta * depth.  0.0 means the prior nails the murkiness level;
    # a value stuck at its initial magnitude means the prior is still a
    # constant and the collapse fix has failed.
    tau_error: float = float("nan")
    # Per-task outer-loop loss. Written to the provenance file so the
    # convergence curve in the paper is one file read away, not a re-run.
    history: List[Dict[str, object]] = field(default_factory=list)
    jerlov_source: str = ""
    device: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "param_count": self.param_count,
            "tasks_completed": self.tasks_completed,
            "duration_seconds": round(self.duration_seconds, 2),
            "seed": self.seed,
            "source_images": self.source_images,
            "dedup": self.dedup,
            "final_meta_loss": self.final_meta_loss,
            "type_accuracy": self.type_accuracy,
            "tau_error": self.tau_error,
            "history": self.history,
            "jerlov_source": self.jerlov_source,
            "device": self.device,
            "config": self.config,
        }


class MetaTrainer:
    """FOMAML meta-training on synthetic tasks."""

    # Weights of L_meta (§18); the methodology fixes the *form* of the loss
    # but not its coefficients.  Reconstruction dominates, structural
    # agreement is half as important, and the physical-model term is a weak
    # tie-breaker so the meta-loss cannot be minimised by a physically
    # implausible solution.
    META_L1 = 1.0
    META_SSIM = 0.5
    META_PHYS = 0.1

    def __init__(self, cfg: PhysINRConfig, image_paths: Sequence[Path],
                 device: Optional[str] = None,
                 log_dir: Optional[Path] = None,
                 logger=None) -> None:
        self.cfg = cfg
        self.device = resolve_device(device if device is not None else cfg.device)
        self.image_paths = [Path(p) for p in image_paths]
        if not self.image_paths:
            raise ValueError("meta-training requires at least one clean source image")
        self.logger = logger or get_logger("physinr.meta", log_dir)
        self.model = PhysINR(cfg).to(self.device)
        self.loss_fn = PhysINRLoss(cfg)
        self.synth = SyntheticDegradation(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        # Warm the inner-loop optimisers once; the state is reset per task.
        self._cache: List[np.ndarray] = []

    # --------------------------------------------------------------- caching
    def preload(self, max_images: int = 1_500, max_side: Optional[int] = None
                ) -> None:
        """Decode a random subset of the source pool once and keep it in RAM.

        Meta-training samples tasks with replacement over many epochs, so
        decoding every image on the fly would spend most of the run in the
        image loader.  A fixed in-RAM pool is the standard remedy; the pool is
        drawn *after* deduplication, so it contains no evaluation content.

        ``max_side`` defaults to ``2 * meta_crop``. A source only needs to be
        large enough to yield one ``meta_crop`` patch, and the widest-aspect
        image keeps a short side of about half its long side, so loading at
        ``2 * meta_crop`` is what makes a landscape or portrait photo eligible
        at all. Loading smaller silently discards most of the pool: DIV2K at
        ``max_side=256`` leaves only 120 of 900 images usable for a 192 crop,
        because a 2040x1356 frame scales to 256x170 and 170 < 192.
        """
        if max_side is None:
            max_side = 2 * self.cfg.meta_crop
        idx = self.rng.permutation(len(self.image_paths))[:max_images]
        self._cache = []
        n_skipped = 0
        for i in idx:
            try:
                img = load_image(self.image_paths[i], max_side)
                if img.shape[0] < self.cfg.meta_crop or img.shape[1] < self.cfg.meta_crop:
                    n_skipped += 1
                    continue
                self._cache.append((img * 255.0).round().astype(np.uint8))
            except Exception as exc:
                self.logger.warning(f"skipping unreadable source {self.image_paths[i]}: {exc}")
        self.logger.info(f"preloaded {len(self._cache)} clean source images "
                         f"of {len(self.image_paths)} candidates "
                         f"(load max_side={max_side}, "
                         f"{n_skipped} too small for a {self.cfg.meta_crop}px crop)")

    def _sample_batch(self) -> List[np.ndarray]:
        """Random crops at the meta-training resolution."""
        size = self.cfg.meta_crop
        out = []
        if not self._cache:
            self.preload()
        for _ in range(self.cfg.meta_batch):
            img = self._cache[int(self.rng.integers(0, len(self._cache)))]
            h, w = img.shape[:2]
            if h < size or w < size:
                out.append(img)
                continue
            y = int(self.rng.integers(0, h - size + 1))
            x = int(self.rng.integers(0, w - size + 1))
            out.append(img[y:y + size, x:x + size].astype(np.float32) / 255.0)
        return out

    # ---------------------------------------------------------------- train
    def train(self, tasks: Optional[int] = None, out_path: Optional[Path] = None,
              provenance: Optional[Dict[str, object]] = None,
              log_every: int = 50) -> MetaRecord:
        """Run the meta-loop and optionally checkpoint theta_0."""
        cfg = self.cfg
        seed_everything(cfg.seed)
        n_tasks = tasks if tasks is not None else cfg.meta_tasks
        if not self._cache:
            self.preload()

        outer_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.meta_outer_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            outer_opt, T_max=max(1, n_tasks // cfg.meta_batch))

        start = time.time()
        losses: List[float] = []
        type_accs: List[float] = []
        tau_errs: List[float] = []
        self.model.train()
        for task in range(n_tasks):
            clean_batch = self._sample_batch()
            images = np.stack(clean_batch, axis=0) if len(clean_batch) > 1 else clean_batch[0][None]
            x_clean = torch.from_numpy(
                np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(self.device)

            # ---- one sampled task per image in the batch -------------------
            degraded_list, clean_target_list = [], []
            type_targets, type_anchors, tau_targets = [], [], []
            for b in range(x_clean.shape[0]):
                tp = self.synth.sample_task(
                    self.rng, (x_clean.shape[2], x_clean.shape[3]),
                    images[b].reshape(-1, 3).mean(axis=0))
                d, c = self.synth.degrade(x_clean[b:b + 1], tp, self.rng)
                degraded_list.append(d[0])
                clean_target_list.append(c[0])
                # The drawn type is known here -- it is a property of the
                # synthetic task, not an annotated label -- so the outer loop
                # can hold the type head to it.  The soft mixture is the
                # honest target: the effective attenuation was built from it.
                if tp.type_mix is not None:
                    type_targets.append(list(tp.type_mix))
                else:
                    onehot = [0.0] * cfg.num_water_types
                    onehot[tp.water_type_idx % cfg.num_water_types] = 1.0
                    type_targets.append(onehot)
                type_anchors.append(tp.water_type_idx % cfg.num_water_types)
                # tau = beta * depth is the identifiable part of the task: the
                # level of murkiness a single frame actually shows.  Clamped
                # to the model's representable range, which the degradation's
                # own transmission clamp mirrors.
                tau = [min(bc * tp.depth_scale, cfg.tau_max) for bc in tp.beta]
                tau_targets.append(tau)
            x_degraded = torch.stack(degraded_list)
            x_target = torch.stack(clean_target_list)

            # The tau supervision targets the RESIDUAL the learned prior must
            # explain: the v2 prior is tau_data (measured from the image's own
            # per-channel contrast) plus the Jerlov component, so holding the
            # Jerlov component to the full tau would double-count the data
            # term.  tau_data is recomputed here exactly as the model's
            # forward computes it -- on the degraded frame, full resolution.
            tau_true = torch.tensor(tau_targets, device=self.device)
            if cfg.w_tau_sup > 0.0 and cfg.tau_data_weight > 0.0:
                from .physics import estimate_transmission_level as _etl
                with torch.no_grad():
                    t_data = _etl(x_degraded.permute(0, 2, 3, 1))
                    tau_data = -torch.log(t_data.clamp(min=cfg.t_min))
                tau_targets = (tau_true - cfg.tau_data_weight * tau_data) \
                    .clamp(0.0, cfg.tau_max).tolist()

            # ---- inner loop: unsupervised adaptation on the synthetic image
            # Same coordinate stride as test-time adaptation, so the inner-loop
            # dynamics the meta-initialisation is tuned for are the ones it will
            # actually meet at inference (§21 step 4a).
            stride = max(1, cfg.loss_stride)
            x_loss = strided_image(x_degraded, stride)
            inner = copy.deepcopy(self.model).to(self.device)
            inner_opt = torch.optim.Adam(inner.parameters(), lr=cfg.meta_inner_lr)
            for s in range(cfg.meta_inner_steps):
                inner_opt.zero_grad(set_to_none=True)
                out = inner(x_degraded, step=s, coord_stride=stride)
                bd = self.loss_fn(x_loss, out, s, cfg.meta_inner_steps)
                bd.total.backward()
                torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=1.0)
                inner_opt.step()
                self._project(inner)

            # ---- outer loss against the synthetic clean image (§18) --------
            # Scored on the full grid, like the enhanced image a user receives.
            outer_opt.zero_grad(set_to_none=True)
            out = inner(x_degraded, step=cfg.meta_inner_steps)
            pred = out.restored
            l1 = (pred - x_target.permute(0, 2, 3, 1)).abs().mean()
            from .losses import _ssim
            from .physics import channel_luminance as lum
            y_p = lum(pred)
            y_t = lum(x_target.permute(0, 2, 3, 1))
            l_ssim = 1.0 - _ssim(y_p, y_t)
            l_phys = (self.loss_fn._charbonnier(
                out.regraded_plus - x_degraded.permute(0, 2, 3, 1),
                cfg.charbonnier_eps)
                + cfg.w_range * self.loss_fn._range_loss(out.tau))
            meta_loss = (self.META_L1 * l1 + self.META_SSIM * l_ssim
                         + self.META_PHYS * l_phys)

            # Type supervision (§6 fix): cross-entropy of the predicted type
            # mixture against the mixture the task was drawn with, and
            # log-space MSE on the implied optical thickness.  Without them
            # the head's optimum is the marginal type distribution -- a
            # constant, which is the collapse the diagnostics measured.  Only
            # the outer loop sees them; the inner loop stays unsupervised.
            type_acc = float("nan")
            tau_err = float("nan")
            if cfg.w_type_sup > 0.0:
                log_w = torch.log(out.water_weights.clamp_min(1e-8))
                tgt = x_degraded.new_tensor(type_targets)
                l_type = -(tgt * log_w).sum(dim=1).mean()
                meta_loss = meta_loss + cfg.w_type_sup * l_type
                anchor = x_degraded.new_tensor(type_anchors, dtype=torch.long)
                type_acc = float(
                    (out.water_weights.argmax(dim=1) == anchor)
                    .float().mean())
                type_accs.append(type_acc)
            if cfg.w_tau_sup > 0.0:
                tau_true = x_degraded.new_tensor(tau_targets)
                log_tau = torch.log(out.tau_jerlov + 1e-4)
                resid = log_tau - torch.log(tau_true + 1e-4)
                meta_loss = meta_loss + cfg.w_tau_sup * resid.pow(2).mean()
                tau_err = float(resid.detach().abs().mean())
                tau_errs.append(tau_err)
            # v3 level supervision: the range estimator normalises the best
            # channel to t = 1, but that channel is itself attenuated in real
            # water.  The head predicts its tau; the task knows it exactly
            # (tau = beta * d, best channel = min tau).  Supervised only in
            # the outer loop, like the type head above.
            if cfg.w_level_sup > 0.0 and cfg.level_head:
                t_best_true = torch.exp(-tau_true.clamp(max=cfg.tau_max)) \
                    .max(dim=1).values                     # (B,)
                tau_best_true = -torch.log(t_best_true.clamp_min(1e-4))
                l_level = (out.tau_level - tau_best_true).pow(2).mean()
                meta_loss = meta_loss + cfg.w_level_sup * l_level

            # FOMAML: differentiate the outer loss at theta', then apply that
            # gradient to theta.  The inner path contributes no second-order
            # information, which is what makes the memory cost independent of
            # the number of inner steps.
            grads = torch.autograd.grad(meta_loss, list(inner.parameters()))
            for p, g in zip(self.model.parameters(), grads):
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.copy_(g.detach())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            outer_opt.step()
            sched.step()
            del inner

            losses.append(float(meta_loss.detach()))
            if (task + 1) % log_every == 0:
                recent = np.mean(losses[-log_every:])
                extra = ""
                if type_accs:
                    extra += f" type_acc={np.mean(type_accs[-log_every:]):.3f}"
                if tau_errs:
                    extra += f" tau_err={np.mean(tau_errs[-log_every:]):.3f}"
                self.logger.info(
                    f"task {task + 1}/{n_tasks} meta_loss={recent:.5f}{extra} "
                    f"({(time.time() - start) / (task + 1):.3f}s/task)")

        duration = time.time() - start
        record = MetaRecord(
            config=cfg.to_dict(),
            param_count=self.model.count_parameters(),
            tasks_completed=n_tasks,
            duration_seconds=duration,
            seed=cfg.seed,
            source_images=len(self._cache),
            dedup=provenance or {},
            final_meta_loss=float(np.mean(losses[-log_every:])) if losses else float("nan"),
            type_accuracy=float(np.mean(type_accs[-log_every:])) if type_accs else float("nan"),
            tau_error=float(np.mean(tau_errs[-log_every:])) if tau_errs else float("nan"),
            history=[{"task": i + 1, "meta_loss": v} for i, v in enumerate(losses)],
            jerlov_source=__import__("hydro_bench.methods.deep.physinr_zero.physics", fromlist=["JERLOV_SOURCE"]).JERLOV_SOURCE,
            device=self.device,
        )
        if out_path is not None:
            self.save(out_path, record)
        return record

    # --------------------------------------------------------------- helpers
    def _project(self, model: PhysINR) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.clamp_(-8.0, 8.0)

    def save(self, out_path: Path, record: MetaRecord) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "config": self.cfg.to_dict(),
                    "param_count": self.model.count_parameters(),
                    "tasks": record.tasks_completed,
                    "meta_loss": record.final_meta_loss}, out_path)
        write_provenance(out_path.with_suffix(".provenance.json"), record.to_dict())
        self.logger.info(f"saved meta-initialisation -> {out_path} "
                         f"({record.param_count} params, "
                         f"{record.tasks_completed} tasks, {duration(record)}")


def duration(record: MetaRecord) -> str:
    s = int(record.duration_seconds)
    return f"{s // 60}m{s % 60:02d}s"


__all__ = ["MetaTrainer", "MetaRecord"]
