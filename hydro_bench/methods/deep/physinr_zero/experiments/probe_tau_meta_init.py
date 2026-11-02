"""Recompute the meta-init tau prior for the UIEB/UCCS frames of §6.3.

Patches ``transmission_from_tau`` during a plain forward (no adaptation) at
480 px, so the recorded tau is exactly what the inverse sees at step 0 under
``PhysINRConfig.legacy()``.  Also reports the per-channel saturation fraction
of the *inverse output* at step 0, which is the quantity §4.5 quotes.

CPU-only by default so it can run beside an MPS benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero import physics
from hydro_bench.methods.deep.physinr_zero.adapt import load_meta_checkpoint
from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.model import PhysINR
from hydro_bench.methods.deep.physinr_zero.utils import load_image, resolve_device

REPO = Path(__file__).resolve().parents[4]
MAX_SIDE = 480
DEVICE = resolve_device("cpu")  # keep MPS free for the running benchmark

FRAMES = [
    ("uieb 100", "datasets/UIEB/raw_input_imgs/100_img_.png"),
    ("uieb 125", "datasets/UIEB/raw_input_imgs/125_img_.png"),
    ("uccs bg_41", "datasets/UCCS/color_cast/bg_41.jpg"),
    ("uccs green_50", "datasets/UCCS/color_cast/green_50.jpg"),
]

CKPT = REPO / "weights" / "physinr_zero_meta.pt"


def probe(frame_path: Path, cfg: PhysINRConfig) -> dict:
    model = PhysINR(cfg)
    load_meta_checkpoint(model, CKPT)
    model.eval()

    arr = load_image(frame_path, max_side=MAX_SIDE)
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        phys = model(img)

    # tau is (B, H, W, 3), channel-last
    tau = phys.tau.detach().cpu().reshape(-1, 3)
    t = torch.exp(-tau.clamp(min=0.0))
    o = phys.restored.detach().cpu().squeeze(0).numpy()
    rec = {}
    for i, name in enumerate("RGB"):
        rec[f"tau_{name}"] = round(float(tau[:, i].mean()), 3)
        rec[f"t_{name}"] = round(float(t[:, i].mean()), 3)
        ch = o[:, :, i]
        rec[f"saturate_hi_{name}"] = round(float((ch >= 0.999).mean()) * 100, 1)
        rec[f"saturate_lo_{name}"] = round(float((ch <= 0.001).mean()) * 100, 1)
    # how many pixels are clipped at either rail, per channel and overall
    rec["clip_all_pct"] = round(
        float(np.mean([(o[:, :, i] >= 0.999).mean() + (o[:, :, i] <= 0.001).mean()
                       for i in range(3)])) * 100, 1)
    return rec


def main() -> None:
    legacy = PhysINRConfig.legacy()
    results = {}
    for label, rel in FRAMES:
        p = REPO / rel
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        results[label] = probe(p, legacy)
        r = results[label]
        amp = r["t_G"] and 1.0 / r["t_G"]
        print(
            f"{label:14s} tau=({r['tau_R']:.2f},{r['tau_G']:.2f},{r['tau_B']:.2f}) "
            f"t=({r['t_R']:.2f},{r['t_G']:.2f},{r['t_B']:.2f}) "
            f"green_amp={amp:.2f}x  "
            f"green>=1.0: {r['saturate_hi_G']:.1f}%  "
            f"red<=0: {r['saturate_lo_R']:.1f}%"
        )

    out = REPO / "experiments" / "tau_meta_init_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
