"""Can the physics features separate the Jerlov types at all?

Before spending an hour of MPS meta-training on the type-head fix, check that
the content descriptor the new head consumes (channel means, log-ratios,
estimated background light) actually carries type information in the synthetic
degradation family.  Nearest-centroid top-1 accuracy over the drawn type;
chance is 10%.

This is a feasibility probe, not an evaluation: the head also gets the pooled
extractor features, which training will shape, and the drawn type is only the
anchor of a soft mixture -- so even the Bayes accuracy here is well below 100.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.physics import estimate_background_light
from hydro_bench.methods.deep.physinr_zero.synthesize import SyntheticDegradation
from hydro_bench.methods.deep.physinr_zero.utils import load_image

REPO = Path(__file__).resolve().parents[4]
N_PER_TYPE = 40
CROP = 96


def features(img: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(img.transpose(2, 0, 1)[None]).float()
    m = x.mean(dim=(2, 3))[0].numpy()
    b = estimate_background_light(
        x.permute(0, 2, 3, 1), 0.10, 0.90)[0].numpy()
    with np.errstate(divide="ignore"):
        lr = np.log((m + 1e-4) / np.roll(m + 1e-4, -1))
    # Depth-gradient signature: the water column attenuates each channel
    # multiplicatively with depth, so the top/bottom ratio of a frame divides
    # out the scene and leaves the per-channel attenuation -- and the
    # cross-channel ratios of THAT are the type signature, scene-free.
    h = img.shape[0]
    top = img[:h // 3].mean(axis=(0, 1))
    bot = img[-h // 3:].mean(axis=(0, 1))
    with np.errstate(divide="ignore"):
        grad = np.log((bot + 1e-4) / (top + 1e-4))
    grad_rel = grad - grad[1]              # relative to green: scene-free beta ratios
    return np.concatenate([m, lr, b, grad, grad_rel])


def main() -> int:
    cfg = PhysINRConfig()
    synth = SyntheticDegradation(cfg)
    rng = np.random.default_rng(1234)

    src = sorted((REPO / "datasets/clean_photos/DIV2K_train_HR").glob("*.png"))[:8]
    imgs = []
    for p in src:
        img = load_image(p, 256)
        h, w = img.shape[:2]
        imgs.append(img[:h // CROP * CROP, :w // CROP * CROP]
                    .reshape(h // CROP, CROP, w // CROP, CROP, 3)
                    .transpose(0, 2, 1, 3, 4).reshape(-1, CROP, CROP, 3))
    crops = np.concatenate(imgs)

    feats, labels = [], []
    for t in range(len(synth.types)):
        for _ in range(N_PER_TYPE):
            # Rejection-draw until the task's anchor type matches the label:
            # sample_task picks the type itself, and the label must be the
            # type actually drawn, not the loop counter.
            while True:
                tp = synth.sample_task(rng, (CROP, CROP),
                                       crops[0].reshape(-1, 3).mean(0))
                if tp.water_type_idx == t:
                    break
            k = rng.integers(0, len(crops))
            deg, _ = synth.degrade(
                torch.from_numpy(crops[k].transpose(2, 0, 1)[None]).float(),
                tp, rng)
            feats.append(features(deg[0].permute(1, 2, 0).numpy()))
            labels.append(t)
    feats = np.stack(feats)
    labels = np.array(labels)

    # The head is Linear, so probe with a linear readout: closed-form ridge
    # regression to the one-hot labels, argmin as the decision.  Train/test
    # split by half, two passes so every sample is scored once.
    def ridge_scores(tr, te):
        y = np.eye(len(synth.types))[labels[tr]]
        A = feats[tr]
        W = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ y)
        return feats[te] @ W

    n = len(feats)
    idx = np.arange(n)
    first, second = idx[:n // 2], idx[n // 2:]
    preds = np.empty(n, dtype=int)
    preds[second] = ridge_scores(first, second).argmax(1)
    preds[first] = ridge_scores(second, first).argmax(1)
    acc = float((preds == labels).mean())
    print(f"linear-readout top-1 over {len(synth.types)} types, "
          f"n={n}: {acc:.3f}  (chance {1/len(synth.types):.2f})")

    # Per-type confusion summary: which types are distinguishable.
    cents = np.stack([feats[labels == t].mean(0)
                      for t in range(len(synth.types))])
    d = ((cents[:, None] - cents[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    names = synth.types
    for t in np.argsort(d.min(1))[:4]:
        j = int(np.argmin(d[t]))
        print(f"  most confusable pair: {names[t]} <-> {names[j]} "
              f"(centroid dist {d[t].min():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
