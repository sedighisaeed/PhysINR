"""Inference-time ablation: which module costs UIEB PSNR?

The refactored checkpoint reduces cast on turbid water but *loses* PSNR/SSIM on
clear UIEB images relative to the published model.  The loss is either baked
into the meta-initialisation (the coastal-weighted task distribution) or it is
produced by one of the three inference-time modules.  The modules can be
switched off without retraining -- they change the forward path, not the
parameters -- so this ablation separates the two.
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
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device

REPO = Path(__file__).resolve().parents[4]
MAX_SIDE = 480
STEPS = 12
DEVICE = resolve_device("mps" if torch.backends.mps.is_available() else "cpu")

VARIANTS = {
    "full":            {},
    "no_yiq":          {"yiq_align": False},
    "no_adaptive_amb": {"ambient_adaptive": False},
    "no_spectral":     {"use_spectral_beta": False},
    "modules_all_off": {"yiq_align": False, "ambient_adaptive": False,
                        "use_spectral_beta": False},
}


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
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    ckpt = REPO / "weights/physinr_zero_cast.pt"

    adapters = {}
    for name, over in VARIANTS.items():
        cfg = replace(PhysINRConfig(), **over)
        model = PhysINR(cfg)
        load_meta_checkpoint(model, ckpt)
        model.eval()
        adapters[name] = TestTimeAdapter(cfg, model, DEVICE)

    leg_cfg = PhysINRConfig.legacy()
    leg_model = PhysINR(leg_cfg)
    load_meta_checkpoint(leg_model, REPO / "weights/physinr_zero_meta.pt")
    leg_model.eval()
    adapters["legacy_published"] = TestTimeAdapter(leg_cfg, leg_model, DEVICE)

    dummy = np.zeros((64, 64, 3), np.float32)          # MPS warm-up
    for a in adapters.values():
        a.adapt(dummy, steps=1)

    uieb_dir = REPO / "datasets/UIEB/raw_input_imgs"
    ref_dir = REPO / "datasets/UIEB/reference_imgs"
    flagged = ["117_img_.png", "12299.png", "12324.png", "12336.png",
               "125_img_.png", "124_img_.png", "509_img_.png", "280_img_.png",
               "1407.png", "162_img_.png"]
    rng = np.random.default_rng(1)
    manifest = json.load(open(REPO / "hydro_bench/datasets/manifests/uieb.json"))
    pool = [f for f in manifest["splits"]["test"] if f not in flagged]
    files = flagged + [str(f) for f in rng.choice(pool, size=10, replace=False)]

    from hydro_bench.metrics.full_reference import evaluate_pair

    records = []
    for f in files:
        img = resize(load_image(uieb_dir / f, None))
        ref = resize(load_image(ref_dir / f, None))
        rec = {"image": f, "input": _score(img, ref, evaluate_pair)}
        t0 = time.time()
        for name, adapter in adapters.items():
            rec[name] = _score(adapter.adapt(img, steps=STEPS)[0], ref,
                               evaluate_pair)
        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        print(f"  {f:14} in={rec['input']['psnr']:5.2f} "
              + " ".join(f"{k}={rec[k]['psnr']:5.2f}"
                         for k in adapters), flush=True)

    out = REPO / "experiments/module_ablation.json"
    out.write_text(json.dumps({"records": records, "max_side": MAX_SIDE,
                               "steps": STEPS, "device": DEVICE,
                               "variants": {k: dict(v) for k, v in VARIANTS.items()}},
                              indent=1))

    print("\n--- mean over", len(records), "UIEB images")
    keys = ["input"] + list(adapters)
    for k in keys:
        vals = [r[k] for r in records]
        print(f"{k:16} psnr={_m(vals,'psnr'):6.2f}  ssim={_m(vals,'ssim'):.4f}  "
              f"cast={_m(vals,'cast'):6.2f}  mu_a={_m(vals,'mu_a'):7.2f}  "
              f"uciqe={_m(vals,'uciqe'):6.2f}")
    return 0


def _m(vals, field):
    v = [x[field] for x in vals if x.get(field) is not None]
    return sum(v) / len(v) if v else float("nan")


def _score(image, ref, evaluate_pair) -> dict:
    from hydro_bench.metrics.no_reference import uciqe
    cast, a, b = lab_cast(image)
    fr = evaluate_pair(ref, image, metrics=("psnr", "ssim"))
    return {"cast": round(cast, 2), "mu_a": round(a, 2), "mu_b": round(b, 2),
            "uciqe": round(float(uciqe(image)), 2),
            "psnr": round(fr.psnr, 2), "ssim": round(fr.ssim, 4)}


if __name__ == "__main__":
    raise SystemExit(main())
