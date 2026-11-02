"""CycleGAN adapter (Zhu et al., ICCV 2017).

CycleGAN learns an unpaired image-to-image translation between two domains with
a cycle-consistency constraint: G(A) should be recoverable by F, and vice versa,
so the mapping cannot collapse every input onto a single target. It is included
here as the canonical *unpaired* baseline -- a network that never sees a
degraded/clean pair, which is a weaker training signal than any supervised
method in the table and a stronger one than nothing at all.

No pretrained underwater CycleGAN exists: the authors release checkpoints for
horse2zebra, monet2photo and similar, but not for underwater imagery. The
checkpoint used here is therefore trained by this work on the UIEB *train*
split only (623 images; domain A = raw underwater, domain B = reference) with
the published protocol -- 100 epochs at constant LR followed by 100 decayed,
``lambda_cycle`` 10, patch discriminators, 286->256 resize-and-crop
augmentation. The test split is untouched by that training, so the row is
held-out; the provenance records that the weights are ours, not the original
authors'.

Inference uses the repository's one-directional ``TestModel`` (``--model
test``): it builds and loads G_A alone, dropping the discriminators and the
reverse generator. The model is constructed through the repository's own option
parser and ``networks.define_G``, and the checkpoint is read by
``BaseModel.load_networks``, so the architecture and the weight mapping stay
entirely the authors' code. Driving it in-process rather than shelling out to
``test.py`` avoids a Python-startup and model-load cost per image across
hundreds of frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ....core.base_algorithm import (AlgorithmCapabilities, BaseAlgorithm,
                                     load_weights)
from .._vendored import forward_with_cpu_fallback, repo_on_syspath, resolve_device

CHECKPOINT_NAME = "cyclegan_uieb_latest.pth"
TRAINING_NOTE = (
    "no pretrained underwater CycleGAN exists; weights trained by this work on "
    "the UIEB train split (623 images) with the published protocol"
)


@dataclass
class CycleGAN(BaseAlgorithm):
    """Unpaired image-to-image translation, weights trained on UIEB-train."""

    name = "CycleGAN"
    capabilities = AlgorithmCapabilities(
        supports_color=True, supports_grayscale=False, max_bit_depth=8,
        # Trained by this work on the UIEB train split; the harness flags any
        # evaluation on UIEB as not-held-out, which is correct and intended.
        requires_gpu=True, training_sets=("UIEB",),
    )

    checkpoint: Optional[Path] = None
    device: str = "auto"

    metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        if self.checkpoint is None:
            self.checkpoint = Path(CHECKPOINT_NAME)
        self.checkpoint = load_weights(self.checkpoint, must_exist=True)
        self._model, self._opt = self._load_model()
        self.metadata["training"] = TRAINING_NOTE

    def _load_model(self):
        """Build the repository's TestModel with its own option parser."""
        repo = repo_on_syspath("cyclegan")

        from options.test_options import TestOptions

        # The parser is an argparse instance that reads sys.argv, so the
        # arguments are pushed explicitly and argv is cleared for the parse.
        argv = [str(repo / "test.py"),
                "--model", "test",
                "--no_dropout",
                "--preprocess", "none",
                "--input_nc", "3", "--output_nc", "3",
                "--norm", "instance",
                "--netG", "resnet_9blocks",
                "--checkpoints_dir", str(self.checkpoint.parent),
                "--name", self.checkpoint.parent.name]
        import sys
        old_argv = list(sys.argv)
        sys.argv = argv
        try:
            opt = TestOptions().parse()
        finally:
            sys.argv = old_argv
        opt.isTrain = False
        opt.device = self._torch_device()
        # The loader composes <checkpoints_dir>/<name>/<epoch>_net_G.pth; the
        # staged checkpoint is named latest_net_G.pth, matching --epoch latest.
        opt.epoch = "latest"

        from models import create_model

        model = create_model(opt)
        model.setup(opt)
        model.eval()
        return model, opt

    def _torch_device(self):
        import torch
        if self.device == "mps":
            return torch.device("mps")
        if self.device.startswith("cuda"):
            return torch.device(self.device)
        return torch.device("cpu")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        model = self._model

        def forward(device: str):
            import torch

            x = self._to_tensor(image).to(device)
            with torch.no_grad():
                fake = model.netG(x)
            return self._to_numpy(fake, image.shape)

        out = forward_with_cpu_fallback(
            forward, (model.netG,), self.device, self.metadata)
        return self._validate_io(image, out)

    # ------------------------------------------------------------- metadata
    def run_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "used_fallback": False,
            "note": "official implementation; " + TRAINING_NOTE,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "device": self.device,
            **self.metadata,
        }

    # ----------------------------------------------------------------- io
    @staticmethod
    def _to_tensor(image: np.ndarray) -> "torch.Tensor":
        import torch

        arr = image.astype(np.float32) / 255.0
        # The published data pipeline normalises to [-1, 1]; the generator was
        # trained on that range, so inference must match it exactly.
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    @staticmethod
    def _to_numpy(tensor, shape) -> np.ndarray:
        arr = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        arr = (arr * 0.5) + 0.5
        return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
