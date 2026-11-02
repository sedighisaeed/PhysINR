"""v2 evaluation: does the data-driven prior + t_min floor kill the blowout?

Sides compared on the SAME images, all at 480 px max side:

  input         the raw underwater frame
  published     weights/physinr_zero_meta.pt under PhysINRConfig.legacy()
                (the exact v1 behaviour, including no floor)
  published+f   the same checkpoint with only the t_min floor enabled --
                isolates fix 2 without any retraining
  typehead      weights/physinr_zero_typehead.pt (the content-dependent prior,
                pre-v2)
  v2            weights/physinr_zero_v2.pt (both fixes, retrained)

Metrics: per-channel clipping percentage (the metric that actually sees the
neon wash), plus the earlier cast / PSNR / SSIM / UCIQE set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter, load_meta_checkpoint
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device, seed_everything

REPO = Path(__file__).resolve().parents[4]
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if torch.backends.mps.is_available() else "cpu")


def clipping(image: np.ndarray) -> dict:
    """Per-channel fraction of pixels pinned at 0 or 1, plus the total."""
    out = {}
    total = 0.0
    for c, name in enumerate("RGB"):
        ch = image[:, :, c]
        lo = float((ch <= 0.001).mean())
        hi = float((ch >= 0.999).mean())
        out[f"clip_{name}"] = round(lo + hi, 4)
        total += lo + hi
    out["clip_all"] = round(total / 3.0, 4)
    return out


def lab_cast(image: np.ndarray) -> float:
    from hydro_bench.metrics.no_reference import _rgb_to_lab
    lab = _rgb_to_lab(image)
    a, b = float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean())
    return (a ** 2 + b ** 2) ** 0.5


def resize(image: np.ndarray) -> np.ndarray:
    import cv2
    h, w = image.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return image
    s = MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _side(cfg: PhysINRConfig, ckpt: Path) -> TestTimeAdapter:
    model = PhysINR(cfg)
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return TestTimeAdapter(cfg, model, DEVICE)


def main() -> int:
    seed_everything(0)
    leg = PhysINRConfig.legacy()
    sides = {
        "published": _side(leg, REPO / "weights/physinr_zero_meta.pt"),
        "published_floor": _side(replace(leg, t_min=0.25),
                                 REPO / "weights/physinr_zero_meta.pt"),
        "typehead": _side(PhysINRConfig(type_head_features="physics",
                                        t_min=0.0, tau_data_weight=0.0),
                          REPO / "weights/physinr_zero_typehead.pt"),
        "v2": _side(PhysINRConfig(), REPO / "weights/physinr_zero_v2.pt"),
    }
    dummy = np.zeros((64, 64, 3), np.float32)              # MPS warm-up
    for a in sides.values():
        a.adapt(dummy, steps=1)

    from hydro_bench.metrics.full_reference import evaluate_pair
    from hydro_bench.metrics.no_reference import uciqe

    uieb_dir = REPO / "datasets/UIEB/raw_input_imgs"
    ref_dir = REPO / "datasets/UIEB/reference_imgs"
    manifest = json.load(open(REPO / "hydro_bench/datasets/manifests/uieb.json"))
    rng = np.random.default_rng(1)
    uieb_files = ["100_img_.png", "106_img_.png", "117_img_.png",
                  "12299.png", "12324.png", "12336.png", "125_img_.png",
                  "124_img_.png", "509_img_.png", "280_img_.png"]
    pool = [f for f in manifest["splits"]["test"] if f not in uieb_files]
    uieb_files += [str(f) for f in rng.choice(pool, size=10, replace=False)]

    records = []
    for f in uieb_files:
        img = resize(load_image(uieb_dir / f, None))
        ref = resize(load_image(ref_dir / f, None))
        rec = {"image": f, "set": "uieb",
               "input": _score(img, ref, evaluate_pair, uciqe)}
        for name, adapter in sides.items():
            rec[name] = _score(adapter.adapt(img, steps=STEPS)[0], ref,
                               evaluate_pair, uciqe)
        records.append(rec)
        row = " ".join(f"{n}:clip={rec[n]['clip_all']:.3f},psnr="
                       f"{rec[n].get('psnr', float('nan')):.1f}"
                       for n in sides)
        print(f"  {f:14} {row}", flush=True)

    uccs_dir = REPO / "datasets/UCCS/color_cast"
    for cls in ("bg", "blue", "green"):
        for i in (1, 25, 50, 75):
            f = f"{cls}_{i:02d}.jpg"
            img = resize(load_image(uccs_dir / f, None))
            rec = {"image": f, "set": "uccs", "input": _score(img, None,
                                                              None, uciqe)}
            for name, adapter in sides.items():
                rec[name] = _score(adapter.adapt(img, steps=STEPS)[0], None,
                                   None, uciqe)
            records.append(rec)
            row = " ".join(f"{n}:clip={rec[n]['clip_all']:.3f}"
                           for n in sides)
            print(f"  {f:14} {row}", flush=True)

    out = REPO / "experiments/v2_fix_results.json"
    out.write_text(json.dumps({"records": records, "steps": STEPS,
                               "max_side": MAX_SIDE, "device": DEVICE},
                              indent=1))

    uieb = [r for r in records if r["set"] == "uieb"]
    print("\n--- UIEB means (n=%d)" % len(uieb))
    for k in ("input",) + tuple(sides):
        print(f"{k:16} clip={_m(uieb,k,'clip_all'):.3f}  "
              f"clipG={_m(uieb,k,'clip_G'):.3f}  "
              f"psnr={_m(uieb,k,'psnr'):6.2f}  ssim={_m(uieb,k,'ssim'):.4f}  "
              f"cast={_m(uieb,k,'cast'):6.2f}  uciqe={_m(uieb,k,'uciqe'):6.2f}")
    uccs = [r for r in records if r.get("set") == "uccs"]
    print("--- UCCS means (n=%d)" % len(uccs))
    for k in ("input",) + tuple(sides):
        print(f"{k:16} clip={_m(uccs,k,'clip_all'):.3f}  "
              f"cast={_m(uccs,k,'cast'):6.2f}  uciqe={_m(uccs,k,'uciqe'):6.2f}")
    print("wrote", out)
    return 0


def _m(rows, key, field):
    v = [r[key][field] for r in rows if r[key].get(field) is not None]
    return sum(v) / len(v) if v else float("nan")


def _score(image, ref, evaluate_pair, uciqe_fn):
    out = clipping(image)
    out["cast"] = round(lab_cast(image), 2)
    out["uciqe"] = round(float(uciqe_fn(image)), 2)
    if ref is not None:
        fr = evaluate_pair(ref, image, metrics=("psnr", "ssim"))
        out["psnr"] = round(fr.psnr, 2)
        out["ssim"] = round(fr.ssim, 4)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
