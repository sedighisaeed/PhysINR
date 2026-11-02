#!/usr/bin/env python3
"""Generate a synthetic paired underwater benchmark for smoke testing.

Real datasets (UIEB, EUVP, LSUI, SUIM, DUO) cannot be redistributed, so this
script builds a *physically plausible* stand-in: a clear scene is degraded by
a Jaffe–McGlamery style attenuation model with wavelength-dependent
absorption, backscatter, and additive noise. It then writes the manifest that
the real DatasetManager verifies, so the entire pipeline -- leakage guards,
runners, metrics, statistics, reporter -- is exercised exactly as it would be
on UIEB.

Use it to verify an installation or a code change without downloading data:

    python scripts/make_synthetic_benchmark.py --out /tmp/hydro_data
    HYDROBENCH_DATA=/tmp/hydro_data python run_benchmark.py --dataset SynthUIB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running directly from a source checkout without installing the package.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

# Wavelength-dependent attenuation coefficients (per metre), the physical
# reason underwater images go blue-green: red is absorbed fastest.
ATTENUATION = {"R": 0.45, "G": 0.10, "B": 0.05}
BACKSCATTER = {"R": 0.02, "G": 0.06, "B": 0.11}


def synthetic_scene(width: int, height: int, generator: np.random.Generator,
                    n_objects: int = 6) -> np.ndarray:
    """A structured, colourful scene: gradient background plus blobs."""
    image = np.zeros((height, width, 3), dtype=np.float64)
    # Depth-varying background.
    yy, xx = np.mgrid[0:height, 0:width]
    depth = yy / max(height - 1, 1)
    base = np.stack([0.25 + 0.35 * (1 - depth),
                     0.35 + 0.25 * (1 - depth),
                     0.30 + 0.20 * depth], axis=-1)
    image[:] = base
    # Detectable "objects": high-contrast coloured discs.
    for _ in range(n_objects):
        radius = generator.integers(max(6, width // 28), max(12, width // 12))
        cx = generator.integers(radius, width - radius)
        cy = generator.integers(radius, height - radius)
        color = generator.uniform(0.15, 1.0, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        image[mask] = color
    return np.clip(image, 0.0, 1.0)


def degrade(image: np.ndarray, generator: np.random.Generator,
            beta: float = 0.55, gamma: float = 0.85,
            noise_std: float = 0.02) -> np.ndarray:
    """Jaffe–McGlamery style underwater degradation.

    I(x) = J(x) * exp(-eta_c * depth(x)) + B_c * (1 - exp(-eta_c * depth(x)))
    plus additive sensor noise and a mild gamma shift toward the blue channel.
    """
    height, width = image.shape[:2]
    depth = (np.mgrid[0:height, 0:width][0] / max(height - 1, 1)) * 2.0 + 0.5
    out = np.empty_like(image)
    for c, channel in enumerate("RGB"):
        eta = ATTENUATION[channel] * (1.0 + beta)
        transmission = np.exp(-eta * depth)
        back = BACKSCATTER[channel] * (0.8 + 0.4 * beta)
        out[:, :, c] = (image[:, :, c] * transmission
                        + back * (1.0 - transmission))
    if noise_std > 0:
        out += generator.normal(0.0, noise_std, out.shape)
    # Blue cast amplification, the signature of turbid water.
    out[:, :, 2] *= 1.0 + 0.15 * (1.0 - gamma)
    out = np.clip(out, 0.0, 1.0)
    return out


def write_image(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8)).save(path)


def generate(root: Path, n_train: int = 24, n_val: int = 8, n_test: int = 12,
             width: int = 128, height: int = 96, seed: int = 0) -> Path:
    """Write the dataset tree plus its verifiable manifest."""
    # HYDROBENCH_DATA is the *base* directory; one dataset lives in a named
    # subdirectory of it, matching the layout configs/datasets.yaml expects.
    root = Path(root) / "SynthUIB"
    generator = np.random.default_rng(seed)
    counts = {"train": n_train, "val": n_val, "test": n_test}
    splits = {"train": [], "val": [], "test": []}
    raw_dir = root / "raw_input_imgs"
    gt_dir = root / "reference_imgs"

    for split, count in counts.items():
        for index in range(count):
            sample_id = f"{split}_{index:04d}.png"
            splits[split].append(sample_id)
            scene = synthetic_scene(width, height, generator)
            write_image(gt_dir / sample_id, scene)
            write_image(raw_dir / sample_id,
                        degrade(scene, generator))

    # Manifest: hashes + explicit disjoint split lists.
    from hydro_bench.core.manifest import build_manifest

    flat_splits = {k: list(v) for k, v in splits.items()}
    manifest = build_manifest(
        dataset="SynthUIB", root=root, splits=flat_splits,
        raw_subdir="raw_input_imgs", gt_subdir="reference_imgs",
        out_path=root / "manifests" / "synthuib.json")
    (root / "dataset_info.json").write_text(json.dumps({
        "dataset": "SynthUIB",
        "description": "Synthetic paired underwater benchmark (smoke test)",
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "image_size": [width, height],
    }, indent=2), encoding="utf-8")
    print(f"wrote SynthUIB to {root}")
    print(f"  splits: {({k: len(v) for k, v in splits.items()})}")
    print(f"  manifest: {root / 'manifests' / 'synthuib.json'}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/hydro_data"))
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.out, args.n_train, args.n_val, args.n_test,
             args.width, args.height, args.seed)


if __name__ == "__main__":
    main()
