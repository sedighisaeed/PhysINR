"""A/B the YIQ gain source and score the clear-water regression.

Sits outside the test suite: this is the experiment that decides between two
inference-time variants of the post-processing stage, using the checkpoints
that already exist.  No training happens here.

Images are downscaled to ``max_side`` because the 8 GiB MPS host dies at the
UIEB native resolution (800x450 at 12 adaptation steps over both sides); the
downscale is applied identically to the input and its reference.
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
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device

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
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _loaded(cfg: PhysINRConfig, ckpt: Path) -> PhysINR:
    from hydro_bench.methods.deep.physinr_zero.model import PhysINR
    model = PhysINR(cfg)
    load_meta_checkpoint(model, ckpt)
    model.eval()
    return model


def main() -> int:
    # ---- three sides: the published model, and the refactored one under each
    # gain source -----------------------------------------------------------
    legacy_cfg = PhysINRConfig.legacy()
    leg = TestTimeAdapter(
        legacy_cfg,
        _loaded(legacy_cfg, REPO / "weights/physinr_zero_meta.pt"), DEVICE)

    sides = {}
    for src in ("input", "restored"):
        cfg = replace(PhysINRConfig(), yiq_gains_source=src)
        sides[src] = TestTimeAdapter(cfg, _loaded(cfg, REPO / "weights/physinr_zero_cast.pt"), DEVICE)

    # warm up MPS: the first forward pays the kernel-compile cost, and paying
    # it inside the timing loop would credit one image with the whole run's.
    dummy = np.zeros((64, 64, 3), np.float32)
    leg.adapt(dummy, steps=1)
    for a in sides.values():
        a.adapt(dummy, steps=1)

    uieb_dir = REPO / "datasets/UIEB/raw_input_imgs"
    ref_dir = REPO / "datasets/UIEB/reference_imgs"
    manifest = json.load(open(REPO / "hydro_bench/datasets/manifests/uieb.json"))
    test_split = manifest["splits"]["test"]

    # The images that exposed the regression, plus a random spread.
    flagged = ["117_img_.png", "12299.png", "12324.png", "12336.png",
               "125_img_.png", "124_img_.png"]
    rng = np.random.default_rng(0)
    pool = [f for f in test_split if f not in flagged]
    rest = list(rng.choice(pool, size=14, replace=False))
    uieb_files = flagged + [str(f) for f in rest]

    uccs_dir = REPO / "datasets/UCCS/color_cast"
    uccs_files = [f"{c}_{i:02d}.jpg" for c in ("bg", "blue", "green")
                  for i in (1, 25, 50, 75)]

    records = []
    for name, path, ref in _iter_images(uieb_files, uieb_dir, ref_dir,
                                        uccs_files, uccs_dir):
        t0 = time.time()
        img = resize(load_image(path, None))
        rec = {"image": name, "set": "uieb" if ref is not None else "uccs",
               "input": _score(img, ref)}
        rec["legacy"] = _score(leg.adapt(img, steps=STEPS)[0], ref)
        for src, adapter in sides.items():
            rec[f"cast_{src}"] = _score(adapter.adapt(img, steps=STEPS)[0], ref)
        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        _log_row(rec)

    out = REPO / "experiments/gains_source_ab.json"
    out.write_text(json.dumps({"records": records, "max_side": MAX_SIDE,
                               "steps": STEPS, "device": DEVICE},
                              indent=1))
    _summary(records)
    return 0


def _iter_images(uieb_files, uieb_dir, ref_dir, uccs_files, uccs_dir):
    for f in uieb_files:
        p = uieb_dir / f
        r = ref_dir / f
        yield f, p, (r if r.is_file() else None)
    for f in uccs_files:
        yield f, uccs_dir / f, None


def _score(image: np.ndarray, ref_path) -> dict:
    from hydro_bench.metrics.full_reference import evaluate_pair
    from hydro_bench.metrics.no_reference import uciqe
    cast, a, b = lab_cast(image)
    out = {"cast": round(cast, 2), "mu_a": round(a, 2), "mu_b": round(b, 2),
           "uciqe": round(float(uciqe(image)), 2)}
    if ref_path is not None:
        ref = resize(load_image(ref_path, None))
        fr = evaluate_pair(ref, image, metrics=("psnr", "ssim"))
        out["psnr"] = round(fr.psnr, 2)
        out["ssim"] = round(fr.ssim, 4)
    return out


def _log_row(rec: dict) -> None:
    print(f"  {rec['image']:16} in={rec['input']['cast']:6.1f} "
          f"leg={rec['legacy']['cast']:6.1f} "
          f"in_g={rec['cast_input']['cast']:6.1f} "
          f"re_g={rec['cast_restored']['cast']:6.1f} "
          f"({rec['seconds']}s)", flush=True)


def _summary(records) -> None:
    def mean(key, field):
        vals = [r[key][field] for r in records if r[key].get(field) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n--- mean over", len(records), "images")
    for key in ("input", "legacy", "cast_input", "cast_restored"):
        line = f"{key:14} cast={mean(key,'cast'):7.2f}  mu_a={mean(key,'mu_a'):7.2f}  mu_b={mean(key,'mu_b'):7.2f}  uciqe={mean(key,'uciqe'):7.2f}"
        if records[0]["legacy"].get("psnr") is not None:
            line += f"  psnr={mean(key,'psnr'):6.2f}  ssim={mean(key,'ssim'):.4f}"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
