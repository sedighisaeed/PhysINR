"""Qualitative grid for the colour-cast refactor.

Enhances the strongest cases from the evaluation with all three sides -- the
raw input, the published checkpoint under its pre-refactoring configuration,
and the refactored checkpoint -- and tiles them into one figure per dataset.

These are the images the numbers in CAST_REFACTOR_RESULTS.md describe, so the
reader can see the cast the metrics measure.  Enhancement happens here (the
evaluation scripts did not save images), at the same 480 px the ablation used.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter, load_meta_checkpoint
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device, to_uint8

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "paper" / "03_figures"
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if __import__("torch").backends.mps.is_available()
                        else "cpu")

# Chosen from the measured records, not by eye: the largest cast reductions and
# the clearest clear-water regressions.
CASES = {
    "uccs": ["bg_41.jpg", "bg_75.jpg", "blue_25.jpg", "green_50.jpg"],
    "uieb": ["117_img_.png", "125_img_.png", "280_img_.png", "509_img_.png"],
}


def _side(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    model = PhysINR(cfg)
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def _resize(image: np.ndarray) -> np.ndarray:
    import cv2
    h, w = image.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return image
    s = MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    import torch
    torch.manual_seed(0)

    leg_cfg = PhysINRConfig.legacy()
    sides = [
        ("Input", None),
        ("Legacy (published)", _side(leg_cfg, REPO / "weights/physinr_zero_meta.pt")),
        ("Cast refactor", _side(PhysINRConfig(type_head_features="none"),
                                REPO / "weights/physinr_zero_cast.pt")),
        ("Content head", _side(PhysINRConfig(),
                               REPO / "weights/physinr_zero_typehead.pt")),
    ]
    # MPS warm-up: the first forward pays the kernel-compile cost.
    dummy = np.zeros((64, 64, 3), np.float32)
    for _, a in sides:
        if a is not None:
            a.adapt(dummy, steps=1)

    OUT.mkdir(parents=True, exist_ok=True)
    for dataset, files in CASES.items():
        base = (REPO / "datasets/UCCS/color_cast" if dataset == "uccs"
                else REPO / "datasets/UIEB/raw_input_imgs")
        tiles, labels = [], []
        for f in files:
            img = _resize(load_image(base / f, None))
            for name, adapter in sides:
                out = img if adapter is None else adapter.adapt(img, steps=STEPS)[0]
                tiles.append(to_uint8(out))
                labels.append(f"{f}  {name}")
            print(f"  {dataset}/{f} done", flush=True)
        _grid(tiles, labels, OUT / f"cast_refactor_{dataset}.png",
              cols=len(sides))
    return 0


def _grid(tiles, labels, path: Path, cols: int) -> None:
    import cv2
    th = min(t.shape[0] for t in tiles)
    tw = min(t.shape[1] for t in tiles)
    tiles = [cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = (len(tiles) + cols - 1) // cols
    label_h = 26
    cell = np.full((th + label_h, tw, 3), 255, np.uint8)
    canvas = np.full((rows * (th + label_h) + 8, cols * (tw + 6) + 6, 3),
                     255, np.uint8)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        cell[:th] = t
        cv2.putText(cell, lab, (4, th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y = r * (th + label_h) + 4
        x = c * (tw + 6) + 3
        canvas[y:y + th + label_h, x:x + tw] = cell
    cv2.imwrite(str(path), canvas)
    print(f"wrote {path}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    raise SystemExit(main())
