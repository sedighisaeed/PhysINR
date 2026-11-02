"""Prior-design qualitative grid: Input | lookup | measured | level | calibrated.

The figure the prior-design study (results §4.5) needs: the same frames whose
water-type-lookup outputs show the neon-wash failure, with each candidate
prior beside them.  Cases are chosen from the benchmark's own saved outputs
and the pathology measurements, not by eye.
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
# Table 6's protocol is 480 px, and the case images are 400-640 px natively, so
# this cap rarely binds and never upsamples an input: the tiles stay at their
# native resolution and the six-column canvas lands just under 140 mm at 500
# dpi, snapping to the 90 mm artwork width. build_latex places these two
# figures at their natural width so nothing is scaled up on the page.
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if __import__("torch").backends.mps.is_available()
                        else "cpu")

CASES = {
    "uieb": ["100_img_.png", "106_img_.png", "117_img_.png", "125_img_.png"],
    "uccs": ["bg_41.jpg", "bg_75.jpg", "blue_25.jpg", "green_50.jpg"],
}


def _side(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    model = PhysINR(cfg)
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def _side_v2_style(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    """Pre-v3 checkpoints were trained without a level head."""
    from dataclasses import replace
    model = PhysINR(replace(cfg, level_head=False))
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def _side_v3_style(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    """v3 has a level head but predates the depth recalibration."""
    from dataclasses import replace
    model = PhysINR(replace(cfg, synth_depth_range=(0.5, 12.0)))
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

    sides = [
        ("Input", None),
        ("A: water-type lookup",
         _side(PhysINRConfig.legacy(), REPO / "weights/physinr_zero_meta.pt")),
        ("content head",
         _side_v2_style(PhysINRConfig(type_head_features="physics",
                                      t_min=0.0, tau_data_weight=0.0),
                        REPO / "weights/physinr_zero_typehead.pt")),
        ("B: measured only",
         _side_v2_style(PhysINRConfig(), REPO / "weights/physinr_zero_v2.pt")),
        ("C: level, uncalibrated",
         _side_v3_style(PhysINRConfig(), REPO / "weights/physinr_zero_v3.pt")),
        ("D: level, calibrated",
         _side(PhysINRConfig(), REPO / "weights/physinr_zero_v4.pt")),
    ]
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
        _grid(tiles, labels, OUT / f"revision_v4_{dataset}.png", cols=6)
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
    _publisher_artwork(canvas, path)


def _publisher_artwork(canvas: np.ndarray, png_path: Path) -> None:
    """Emit the publisher's artwork forms for the grid (R12/R13).

    The canvas is already composed raster art, so unlike the charts in
    ``save_figure`` there is nothing to re-render: it is Lanczos-resized to the
    publisher's nearest standard width at exactly 500 dpi (combination art) and
    written as TIFF, with a PDF page of the same size for the LaTeX. Rendering
    directly at 500 dpi avoids the 1000-dpi supersample that would push a
    six-column grid past PIL's decompression-bomb limit.
    """
    from PIL import Image

    from hydro_bench.reporter.plots import (SUBMISSION_DPI,
                                            _target_width_mm)

    natural_mm = canvas.shape[1] / SUBMISSION_DPI * 25.4
    target_mm = _target_width_mm(natural_mm)
    target_px = int(round(target_mm / 25.4 * SUBMISSION_DPI))
    target_h = max(1, int(round(canvas.shape[0] * target_px / canvas.shape[1])))

    rgb = Image.fromarray(canvas[:, :, ::-1]).convert("RGB")  # BGR -> RGB
    rgb = rgb.resize((target_px, target_h), Image.LANCZOS)

    sub = REPO / "paper" / "06_submission" / "figures"
    sub.mkdir(parents=True, exist_ok=True)
    tiff = sub / f"{png_path.stem}.tiff"
    rgb.save(tiff, format="TIFF",
             dpi=(SUBMISSION_DPI, SUBMISSION_DPI), compression="tiff_deflate")
    # PIL sizes the PDF page from `resolution`, so the page is exactly the
    # target width -- matplotlib's figimage(resize=True) would instead resize
    # the figure to the image and blow past the width contract.
    rgb.save(sub / f"{png_path.stem}.pdf", format="PDF",
             resolution=float(SUBMISSION_DPI))
    print(f"  submission artwork: {tiff.name} ({target_mm:.0f} mm, "
          f"{target_px}x{target_h} @ {SUBMISSION_DPI} dpi)")


if __name__ == "__main__":
    raise SystemExit(main())
