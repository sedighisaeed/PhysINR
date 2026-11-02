"""PhysINR-Zero adapter (this work).

PhysINR-Zero is a physics-constrained implicit neural field with per-image
test-time adaptation.  It is *zero-shot* in the strict sense used by the
benchmark: no real underwater image of any dataset ever enters its training.
The meta-initialisation is fit on synthetic degradations applied to clean
*terrestrial* photographs, and the checkpoint is applied to each evaluation
image as-is, adapted with a reference-free objective, and then reset.

That distinction is the reason ``training_sets`` is empty below even though
the method does carry learned weights: HydroBench warns when a method is
evaluated on data it was developed on, and the whole point of this method is
that the warning should never fire.  Declaring a training set here would
either be false or would weaken the central claim of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from .._vendored import resolve_device


@dataclass
class PhysINRZero(BaseAlgorithm):
    """PhysINR-Zero: physics-constrained INR with test-time adaptation."""

    name = "PhysINR-Zero"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        requires_gpu=False,
        # Deliberately empty: see the module docstring.
        training_sets=(),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"
    adapt_steps: Optional[int] = None      # overrides cfg.adapt_steps (ablation)
    loss_stride: Optional[int] = None      # overrides cfg.loss_stride (ablation)
    # Ablation of the objective (§22): the named loss group is zeroed for the
    # whole run.  One group at a time -- combinations are not reported.
    drop_group: Optional[str] = None
    # Ablation of the physical bounds (§5).
    tau_max: Optional[float] = None
    delta_max: Optional[float] = None
    # Run a pre-refactoring checkpoint under the pre-refactoring objective and
    # physics, so a before/after comparison measures the colour-cast refactor
    # rather than two different defaults.
    legacy: bool = False

    used_fallback: bool = field(default=False, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.checkpoint = load_weights(self.checkpoint, must_exist=True)
        self._adapter = self._build()

    # ------------------------------------------------------------- internals
    def _build(self):
        """Construct the configuration and the adaptation loop.

        The configuration is read back out of the checkpoint rather than
        reconstructed, so an ablation checkpoint cannot be silently paired
        with the default hyperparameters and produce numbers that no
        configuration on disk actually describes.
        """
        import torch
        from dataclasses import replace

        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter, load_meta_checkpoint
        from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
        from hydro_bench.methods.deep.physinr_zero.model import PhysINR

        blob = torch.load(self.checkpoint, map_location="cpu",
                          weights_only=False)
        # A placeholder checkpoint written for plumbing tests has tasks == 0
        # (and meta_loss == nan). Benchmarking it would report the scores of a
        # random initialisation, which is not a result of anything, so refuse
        # here rather than letting a bad number reach a results file.
        if not blob.get("tasks"):
            raise ValueError(
                f"{self.name}: {self.checkpoint} contains no meta-trained "
                f"state (tasks={blob.get('tasks')!r}, "
                f"meta_loss={blob.get('meta_loss')!r}). Run "
                f"'python3 -m hydro_bench.methods.deep.physinr_zero.cli meta-train' first; a "
                f"placeholder checkpoint must never be benchmarked.")
        cfg = PhysINRConfig(**blob.get("config", {}))
        # Ablation overrides come from the benchmark invocation, not the
        # checkpoint, so they stay visible in the run's own configuration.
        overrides = {}
        if self.legacy:
            # The pre-refactoring objective and physics.  Read from
            # ``PhysINRConfig.legacy`` so there is one definition of "before"
            # shared with the CLI, instead of a second list here that could
            # drift out of sync with it.
            legacy_cfg = PhysINRConfig.legacy()
            overrides.update({k: getattr(legacy_cfg, k) for k in
                              ("use_spectral_beta", "ambient_adaptive",
                               "yiq_align", "w_cast", "w_bound",
                               "type_head_features", "w_type_sup",
                               "t_min", "tau_data_weight",
                               "tau_jerlov_centered", "level_head",
                               "w_level_sup")})
        if self.adapt_steps is not None:
            overrides["adapt_steps"] = int(self.adapt_steps)
        if self.loss_stride is not None:
            overrides["loss_stride"] = int(self.loss_stride)
        if self.tau_max is not None:
            overrides["tau_max"] = float(self.tau_max)
        if self.delta_max is not None:
            overrides["delta_max"] = float(self.delta_max)
        if self.drop_group is not None:
            # Zeroing the base weight takes the group out entirely; the
            # schedule multiplies it, so it stays out for the whole run.
            g = self.drop_group.strip().lower()
            allowed = {"phys", "cycle", "color", "struct", "spatial",
                       "jerlov", "prior", "cast", "bound"}
            if g not in allowed:
                raise ValueError(
                    f"drop_group={self.drop_group!r} is not one of {sorted(allowed)}")
            overrides[f"w_{g}"] = 0.0
        if overrides:
            cfg = replace(cfg, **overrides)

        model = PhysINR(cfg)
        load_meta_checkpoint(model, self.checkpoint)
        adapter = TestTimeAdapter(cfg, model, device=self.device)

        # Surface the provenance of the checkpoint itself, not just its path:
        # a manuscript claim about "the meta-initialisation" is a claim about
        # how many synthetic tasks it saw and from which source images.
        self._checkpoint_record = {
            "tasks": blob.get("tasks"),
            "meta_loss": blob.get("meta_loss"),
            "param_count": blob.get("param_count"),
            "config": blob.get("config"),
        }
        return adapter

    # ------------------------------------------------------------------ API
    def enhance(self, image: np.ndarray) -> np.ndarray:
        # Validate before the transpose: a rank-mismatched input would
        # otherwise die inside numpy with an opaque "axes don't match" error
        # instead of the contract's own message.
        if not isinstance(image, np.ndarray) or image.ndim != 3 \
                or image.shape[2] not in (1, 3):
            raise ValueError(
                f"{self.name}: expected HxWx3 uint8 RGB, got "
                f"{getattr(image, 'shape', type(image).__name__)}")
        x = image.astype(np.float32) / 255.0
        out, _ = self._adapter.adapt(x)
        arr = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)
        return self._validate_io(image, arr)

    def warmup(self, shape: tuple = (256, 256, 3)) -> None:
        """Run one short adaptation so the first timed image does not pay for
        lazy initialisation (MPS graph compilation in particular)."""
        rng = np.random.default_rng(0)
        x = (rng.random(shape) * 0.5 + 0.2).astype(np.float32)
        self._adapter.adapt(x, steps=1)

    # ----------------------------------------------------------- provenance
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": self.used_fallback,
            "note": "zero-shot: meta-initialised on synthetic degradations "
                    "only; adapted per image with a reference-free objective "
                    "and reset afterwards",
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "device": self.device,
            "checkpoint_record": self._checkpoint_record,
            **self.metadata,
        }


__all__ = ["PhysINRZero"]
