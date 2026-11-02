"""v3 evaluation: does the learned level restore real colour?

The v2 fix killed the neon wash but the outputs stayed wrong: washed out and
desaturated.  Diagnosis: the range estimator normalises the best channel to
t = 1, but on real frames even the least attenuated channel sits at t ~ 0.59,
so the prior under-corrects by ~1.7x.  v3 adds a level head supervised on the
synthetic task's own t_best.

This is the metric v2 was judged blind on: Lab chroma and L against the UIEB
reference photographs, plus clipping (so the neon wash cannot come back).
Sides, all at 480 px, 12 adaptation steps:

  input         the raw underwater frame
  reference     the UIEB ground-truth photograph (natural colour)
  v1            weights/physinr_zero_meta.pt under PhysINRConfig.legacy()
  v2            weights/physinr_zero_v2.pt
  v3            weights/physinr_zero_v3.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter, load_meta_checkpoint
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device, seed_everything

REPO = Path(__file__).resolve().parents[4]
OUT_JSON = REPO / "experiments/v3_fix_results.json"
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if torch.backends.mps.is_available() else "cpu")

import cv2


def _resize(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return image
    s = MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _lab(image: np.ndarray) -> np.ndarray:
    from hydro_bench.metrics.no_reference import _rgb_to_lab
    return _rgb_to_lab(image)


def _score(image: np.ndarray, ref: np.ndarray | None) -> dict:
    """Colour-faithful metrics.  ``clip`` guards the neon wash; ``chroma``,
    ``L`` and ``dE`` are what v2 was blind to."""
    lab = _lab(image)
    a, b = float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean())
    lo = float((image <= 0.001).mean())
    hi = float((image >= 0.999).mean())
    out = {
        "chroma": round(float((a ** 2 + b ** 2) ** 0.5), 2),
        "L": round(float(lab[:, :, 0].mean()), 2),
        "a": round(a, 2),
        "b": round(b, 2),
        "meanR": round(float(image[:, :, 0].mean()), 3),
        "meanG": round(float(image[:, :, 1].mean()), 3),
        "meanB": round(float(image[:, :, 2].mean()), 3),
        "clip": round((lo + hi) / 2.0, 4),
    }
    if ref is not None:
        rlab = _lab(ref)
        out["dE"] = round(float(np.abs(lab - rlab).mean()), 2)
        fr = _pair(ref, image)
        out["psnr"] = round(fr.psnr, 2)
        out["ssim"] = round(fr.ssim, 4)
    return out


def _pair(ref: np.ndarray, image: np.ndarray):
    from hydro_bench.metrics.full_reference import evaluate_pair
    return evaluate_pair(ref, image, metrics=("psnr", "ssim"))


def _side(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    model = PhysINR(cfg)
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def _side_v2_style(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    """Load a checkpoint trained with ``level_head=False`` (v2)."""
    from dataclasses import replace
    model = PhysINR(replace(cfg, level_head=False))
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def _side_v3_style(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    """v3 has a level head but was trained before the depth recalibration, so
    it also needs the pre-v4 synthesis range to reload cleanly."""
    from dataclasses import replace
    model = PhysINR(replace(cfg, synth_depth_range=(0.5, 12.0)))
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def main() -> int:
    seed_everything(0)
    torch.manual_seed(0)

    sides = [
        ("v1", _side(PhysINRConfig.legacy(),
                     REPO / "weights/physinr_zero_meta.pt")),
        ("v2", _side_v2_style(PhysINRConfig(),
                              REPO / "weights/physinr_zero_v2.pt")),
        ("v3", _side_v3_style(PhysINRConfig(),
                              REPO / "weights/physinr_zero_v3.pt")),
        ("v4", _side(PhysINRConfig(),
                     REPO / "weights/physinr_zero_v4.pt")),
    ]
    dummy = np.zeros((64, 64, 3), np.float32)
    for _, a in sides:
        a.adapt(dummy, steps=1)

    manifest = json.load(open(REPO / "hydro_bench/datasets/manifests/uieb.json"))
    rng = np.random.default_rng(1)
    fixed = ["100_img_.png", "106_img_.png", "117_img_.png", "12299.png",
             "12324.png", "12336.png", "125_img_.png", "124_img_.png",
             "509_img_.png", "280_img_.png"]
    pool = [f for f in manifest["splits"]["test"] if f not in fixed]
    files = fixed + [str(f) for f in rng.choice(pool, size=10, replace=False)]

    raw = REPO / "datasets/UIEB/raw_input_imgs"
    ref_dir = REPO / "datasets/UIEB/reference_imgs"
    records = []
    for f in files:
        img = _resize(load_image(raw / f, None))
        ref = _resize(load_image(ref_dir / f, None))
        rec = {"image": f, "input": _score(img, ref),
               "reference": _score(ref, ref)}
        for name, adapter in sides:
            out = adapter.adapt(img, steps=STEPS)[0]
            rec[name] = _score(out, ref)
        records.append(rec)
        row = " ".join(
            f"{n}:chr={rec[n]['chroma']:5.1f},L={rec[n]['L']:5.1f},"
            f"clip={rec[n]['clip']:.3f}" for n in ("v1", "v2", "v3"))
        print(f"  {f:14} ref:chr={rec['reference']['chroma']:5.1f},"
              f"L={rec['reference']['L']:5.1f} | {row}", flush=True)

    OUT_JSON.write_text(json.dumps(
        {"records": records, "steps": STEPS, "max_side": MAX_SIDE,
         "device": str(DEVICE)}, indent=1))

    keys = ("input", "reference") + tuple(n for n, _ in sides)
    print(f"\n--- UIEB means (n={len(records)})")
    for k in keys:
        m = {f: float(np.mean([r[k][f] for r in records]))
             for f in ("chroma", "L", "clip", "meanR", "dE", "psnr", "ssim")}
        print(f"{k:10} chroma={m['chroma']:5.1f}  L={m['L']:5.1f}  "
              f"meanR={m['meanR']:.3f}  clip={m['clip']:.3f}  "
              f"dE={m['dE']:5.2f}  psnr={m['psnr']:6.2f}  ssim={m['ssim']:.4f}")
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
