"""Post-training evaluation of the water-type-head collapse fix.

Three checks, in order of importance:

1. Clear water (UIEB, 20 test images at 480 px): does the content-dependent
   prior recover the PSNR the coastal-weighted constant prior cost?  This is
   the number the whole fix exists for.
2. Turbid water (UCCS, 20 images per cast class): does the cast correction
   survive?  A fix that trades the turbid win back for clear water has failed.
3. Sensitivity: the predicted tau_jerlov across distinct real images, plus
   predicted-vs-synthetic-truth spread on held-out synthetic tasks.

The published checkpoint runs under its pre-refactoring configuration
(legacy), the type-head checkpoint under its own stored configuration.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter, load_meta_checkpoint
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR
from hydro_bench.methods.deep.physinr_zero.synthesize import SyntheticDegradation
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device, seed_everything

REPO = Path(__file__).resolve().parents[4]
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if torch.backends.mps.is_available() else "cpu")


def lab_cast(image: np.ndarray) -> tuple:
    from hydro_bench.metrics.no_reference import _rgb_to_lab
    lab = _rgb_to_lab(image)
    a, b = float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean())
    return (a ** 2 + b ** 2) ** 0.5, a, b


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
    leg_cfg = PhysINRConfig.legacy()
    new_cfg = PhysINRConfig()          # carries the content-dependent heads
    leg = _side(leg_cfg, REPO / "weights/physinr_zero_meta.pt")
    new = _side(new_cfg, REPO / "weights/physinr_zero_typehead.pt")

    dummy = np.zeros((64, 64, 3), np.float32)              # MPS warm-up
    for a in (leg, new):
        a.adapt(dummy, steps=1)

    # ---- 1. UIEB clear water: paired metrics ------------------------------
    from hydro_bench.metrics.full_reference import evaluate_pair
    uieb_dir = REPO / "datasets/UIEB/raw_input_imgs"
    ref_dir = REPO / "datasets/UIEB/reference_imgs"
    flagged = ["117_img_.png", "12299.png", "12324.png", "12336.png",
               "125_img_.png", "124_img_.png", "509_img_.png", "280_img_.png",
               "1407.png", "162_img_.png"]
    manifest = json.load(open(REPO / "hydro_bench/datasets/manifests/uieb.json"))
    rng = np.random.default_rng(1)
    pool = [f for f in manifest["splits"]["test"] if f not in flagged]
    uieb_files = flagged + [str(f) for f in rng.choice(pool, size=10, replace=False)]

    uieb = []
    for f in uieb_files:
        img = resize(load_image(uieb_dir / f, None))
        ref = resize(load_image(ref_dir / f, None))
        rec = {"image": f, "input": _score(img, ref, evaluate_pair),
               "legacy": _score(leg.adapt(img, steps=STEPS)[0], ref, evaluate_pair),
               "typehead": _score(new.adapt(img, steps=STEPS)[0], ref, evaluate_pair)}
        uieb.append(rec)
        print(f"  UIEB {f:14} leg_psnr={rec['legacy']['psnr']:5.2f} "
              f"new_psnr={rec['typehead']['psnr']:5.2f} "
              f"leg_cast={rec['legacy']['cast']:6.1f} "
              f"new_cast={rec['typehead']['cast']:6.1f}", flush=True)

    # ---- 2. UCCS turbid water: cast per class ------------------------------
    uccs_dir = REPO / "datasets/UCCS/color_cast"
    uccs = []
    for cls in ("bg", "blue", "green"):
        for i in (1, 13, 25, 37, 50, 62, 75, 87, 90, 99):
            f = f"{cls}_{i:02d}.jpg"
            img = resize(load_image(uccs_dir / f, None))
            rec = {"image": f, "class": cls,
                   "input": _score(img, None, None),
                   "legacy": _score(leg.adapt(img, steps=STEPS)[0], None, None),
                   "typehead": _score(new.adapt(img, steps=STEPS)[0], None, None)}
            uccs.append(rec)
            print(f"  UCCS {f:14} leg_cast={rec['legacy']['cast']:6.1f} "
                  f"new_cast={rec['typehead']['cast']:6.1f}", flush=True)

    # ---- 3. Prior spread across real images --------------------------------
    new_model = PhysINR(new_cfg)
    load_meta_checkpoint(new_model, REPO / "weights/physinr_zero_typehead.pt")
    new_model.eval()
    probe_imgs = (flagged[:4]
                  + ["860_img_.png", "76_img_.png"]
                  + ["bg_41.jpg", "green_50.jpg", "blue_25.jpg"])
    spread = {}
    with torch.no_grad():
        for p in probe_imgs:
            base = (uccs_dir if p.endswith(".jpg") else uieb_dir)
            img = resize(load_image(base / p, None))
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).float()
            o = new_model(x, step=0)
            spread[p] = {"tau": [round(v, 3) for v in o.tau_jerlov[0].tolist()],
                         "type": [round(v, 3) for v in o.water_weights[0].tolist()]}
    # held-out synthetic spread + correlation
    synth = SyntheticDegradation(new_cfg)
    srng = np.random.default_rng(11)
    src = sorted((REPO / "datasets/clean_photos/DIV2K_train_HR").glob("*.png"))[:6]
    crops = []
    for p in src:
        img = load_image(p, 384)
        h, w = img.shape[:2]
        if h < 200 or w < 200:
            continue
        y, x = srng.integers(0, h - 192), srng.integers(0, w - 192)
        crops.append(img[y:y + 192, x:x + 192])
    pred, true = [], []
    with torch.no_grad():
        for _ in range(80):
            k = int(srng.integers(0, len(crops)))
            clean = torch.from_numpy(crops[k].transpose(2, 0, 1)[None]).float()
            tp = synth.sample_task(srng, (192, 192),
                                   crops[k].reshape(-1, 3).mean(0))
            deg, _ = synth.degrade(clean, tp, srng)
            o = new_model(deg, step=0)
            pred.append(np.log(o.tau_jerlov[0].numpy() + 1e-4))
            true.append(np.log(np.array(tp.beta) * tp.depth_scale + 1e-6))
    pred, true = np.stack(pred), np.stack(true)
    synth_summary = {
        "pred_log_tau_std": [round(float(v), 3) for v in pred.std(0)],
        "true_log_tau_std": [round(float(v), 3) for v in true.std(0)],
        "corr_per_channel": [round(float(np.corrcoef(pred[:, i], true[:, i])[0, 1]), 3)
                             for i in range(3)],
    }

    out = REPO / "experiments/type_head_fix_results.json"
    out.write_text(json.dumps({"uieb": uieb, "uccs": uccs,
                               "prior_spread": spread,
                               "synthetic": synth_summary,
                               "steps": STEPS, "max_side": MAX_SIDE,
                               "device": DEVICE}, indent=1))

    print("\n--- means over", len(uieb), "UIEB (clear) images")
    for k in ("input", "legacy", "typehead"):
        print(f"{k:10} psnr={_m(uieb, k, 'psnr'):6.2f}  ssim={_m(uieb, k, 'ssim'):.4f}  "
              f"cast={_m(uieb, k, 'cast'):6.2f}  uciqe={_m(uieb, k, 'uciqe'):6.2f}")
    print("--- means over", len(uccs), "UCCS (turbid) images")
    for cls in ("bg", "blue", "green"):
        rows = [r for r in uccs if r["class"] == cls]
        print(f"{cls:6} leg_cast={_m(rows,'legacy','cast'):6.2f}  "
              f"new_cast={_m(rows,'typehead','cast'):6.2f}  "
              f"input_cast={_m(rows,'input','cast'):6.2f}")
    print("--- synthetic prior sensitivity")
    print(json.dumps(synth_summary))
    print("wrote", out)
    return 0


def _m(rows, key, field):
    v = [r[key][field] for r in rows if r[key].get(field) is not None]
    return sum(v) / len(v) if v else float("nan")


def _score(image, ref, evaluate_pair):
    from hydro_bench.metrics.no_reference import uciqe
    cast, a, b = lab_cast(image)
    out = {"cast": round(cast, 2), "mu_a": round(a, 2), "mu_b": round(b, 2),
           "uciqe": round(float(uciqe(image)), 2)}
    if ref is not None:
        fr = evaluate_pair(ref, image, metrics=("psnr", "ssim"))
        out["psnr"] = round(fr.psnr, 2)
        out["ssim"] = round(fr.ssim, 4)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
