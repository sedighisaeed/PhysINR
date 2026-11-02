"""Build the qualitative comparison grid from saved enhanced outputs.

The grid shows, for a chosen set of images: the input, each method's enhanced
output, and the reference (when the dataset has one). It is the figure a reader
uses to judge perceptual quality, and it must be built from the *same pixels*
the metrics scored -- which is why the benchmark writes them with --save-images
rather than this script re-running the methods.

Inputs  : <out_dir>/<DATASET>__<split>/images/<method>/<id>.png
          + the dataset manifests, for raw and reference images
Outputs : paper/03_figures/grid_<DATASET>.png

Nothing here enhances an image or computes a score. If a (method, image) tile
is missing the cell is left blank and the gap is reported at the end, so the
figure is honest about an incomplete run instead of substituting a neighbour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIG_DIR = REPO / "paper" / "03_figures"

# Column order of the grid. PRIMARY is drawn last so the proposed method sits
# next to the reference; classical first, then learned baselines.
COLUMN_ORDER = [
    "Input",
    "UDCP", "Retinex",
    "Water-Net", "FUnIE-GAN", "UIEC2-Net", "Five-A+",
    "TOPAL", "U-Shape-Transformer", "WF-Diff", "WaterMamba",
    "PhysINR-Zero",
    "Reference",
]

# Methods whose checkpoints or CUDA-only kernels were unavailable and which
# therefore ran an analytic fallback. Their tiles carry a marker so the figure
# cannot be read as the published model's output.
FALLBACK_METHODS = {"UIEC2-Net", "Five-A+", "WF-Diff", "WaterMamba"}


# ------------------------------------------------------------------ loading
def _read_rgb(path: Path) -> Optional["np.ndarray"]:
    import cv2

    if not path.is_file():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr[:, :, ::-1]


def image_ids_for(dataset: str, images_root: Path, n: int) -> List[str]:
    """The ids that every method should have, taken from PhysINR-Zero's column
    if present (it runs on every dataset) else from the first column found.

    Ids are the full filenames HydroBench used, extension included, so they
    match what ``load_split`` reports. Selection is spread evenly across the
    sorted id list, not the first N: the first N is usually one capture
    session and would bias the grid's scenes.
    """
    cols = sorted(p.name for p in images_root.iterdir() if p.is_dir())
    source = None
    for prefer in ("PhysINR-Zero", "UDCP"):
        if prefer in cols:
            source = images_root / prefer
            break
    if source is None and cols:
        source = images_root / cols[0]
    if source is None:
        return []
    ids = sorted(p.name for p in source.glob("*.png"))
    if not ids:
        return []
    if n >= len(ids):
        return ids
    step = len(ids) / n
    picked = [ids[min(len(ids) - 1, int(i * step))] for i in range(n)]
    # Preserve order and drop any coincidental duplicate from rounding.
    seen, out = set(), []
    for pid in picked:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def raw_and_reference(dataset: str, image_id: str, split: str = "test"):
    """Locate the input and reference image for one id via HydroBench's loader.

    Every dataset has its own pairing rule (UIEB pairs by stem, RUIE has no
    reference at all, EUVP uses A/B directories), and the loader already
    encodes it. Using it here means the grid's Input/Reference tiles are the
    same files the metrics compared against.
    """
    from hydro_bench.core.dataset_manager import DatasetError, make_manager

    try:
        manager = make_manager(dataset, verify=False)
        for sample in manager.load_split(split, require_gt=False):
            if sample.id == image_id:
                return sample.raw_path, sample.gt_path
    except (DatasetError, Exception) as exc:  # noqa: BLE001 - report, don't crash
        print(f"[grid] could not load split for '{dataset}': {exc}")
    return None, None


# ------------------------------------------------------------------ drawing
def _fit(img, tile_h: int, tile_w: int):
    import cv2

    h, w = img.shape[:2]
    scale = max(tile_h / h, tile_w / w)
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_NEAREST  # upscale without inventing detail
    out = cv2.resize(img, (max(tile_w, 1), max(tile_h, 1)), interpolation=interp)
    return out


def _label_bar(text: str, width: int, height: int):
    """A dark bar with white text; drawn at the top of each column."""
    import cv2
    import numpy as np

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(2, (width - tw) // 2)
    y = (height + th) // 2
    cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    return bar


def _mark_fallback(tile):
    """Small corner badge for analytic-fallback columns."""
    import cv2

    h = 11
    bar = _label_bar("fallback*", tile.shape[1], h)
    tile[0:h, :, :] = bar
    return tile


def build_grid(dataset: str, images_root: Path, ids: List[str],
               tile_h: int, tile_w: int, label_h: int
               ) -> Tuple[Optional["np.ndarray"], List[str]]:
    import numpy as np

    method_cols = [p.name for p in sorted(images_root.iterdir()) if p.is_dir()]
    # Keep the canonical order but append any unexpected column at the end so a
    # new method is visible instead of silently dropped.
    ordered = [m for m in COLUMN_ORDER if m in method_cols and m != "Input"]
    ordered += [m for m in method_cols if m not in ordered]

    n_rows, n_cols = len(ids), len(ordered) + 2  # + Input, Reference
    canvas = np.zeros((n_rows * tile_h + label_h,
                       n_cols * tile_w, 3), dtype=np.uint8)

    missing: List[str] = []
    for r, image_id in enumerate(ids):
        raw_path, gt_path = raw_and_reference(dataset, image_id)
        y0 = label_h + r * tile_h

        # Input column
        raw = _read_rgb(raw_path) if raw_path else None
        if raw is None:
            missing.append(f"input/{image_id}")
        canvas[y0:y0 + tile_h, 0:tile_w] = _fit(raw, tile_h, tile_w) if raw is not None \
            else _label_bar("input missing", tile_w, tile_h)

        for c, method in enumerate(ordered, start=1):
            x0 = c * tile_w
            path = images_root / method / image_id
            img = _read_rgb(path)
            if img is None:
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = \
                    _label_bar("not run", tile_w, tile_h)
                missing.append(f"{method}/{image_id}")
                continue
            tile = _fit(img, tile_h, tile_w)
            if method in FALLBACK_METHODS:
                tile = _mark_fallback(tile)
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        # Reference column
        x0 = (n_cols - 1) * tile_w
        gt = _read_rgb(gt_path) if gt_path else None
        if gt is None:
            # Unpaired datasets (RUIE, UCCS) legitimately have none: leave black
            # rather than fabricating one.
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = \
                _label_bar("no reference", tile_w, tile_h)
        else:
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = _fit(gt, tile_h, tile_w)

    # Column labels across the top.
    labels = ["Input"] + [
        ("*" + m if m in FALLBACK_METHODS else m) for m in ordered
    ] + ["Reference"]
    for c, name in enumerate(labels):
        canvas[0:label_h, c * tile_w:(c + 1) * tile_w] = _label_bar(
            name, tile_w, label_h)

    print(f"[grid] {dataset}: {n_rows} rows x {n_cols} cols "
          f"({len(ordered)} methods); {len(missing)} missing tiles")
    for m in missing[:20]:
        print(f"        missing {m}")
    if len(missing) > 20:
        print(f"        ... and {len(missing) - 20} more")
    return canvas, ordered


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="e.g. UIEB, LSUI, EUVP, RUIE, UCCS")
    ap.add_argument("--results-root", default="hydro_bench_results",
                    help="dir holding <DATASET>__test/images/")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="specific sample ids; default: first N available")
    ap.add_argument("--n", type=int, default=4,
                    help="rows when --ids is not given")
    ap.add_argument("--tile-h", type=int, default=192)
    ap.add_argument("--tile-w", type=int, default=256)
    ap.add_argument("--out", default=None, help="default: paper/03_figures/")
    args = ap.parse_args(argv)

    images_root = Path(args.results_root)

    root = images_root / f"{args.dataset}__test" / "images"
    if not root.is_dir():
        print(f"[grid] no saved images at {root}; re-run the benchmark with "
              f"--save-images")
        return 2

    ids = args.ids if args.ids else image_ids_for(args.dataset, root, args.n)
    if not ids:
        print(f"[grid] no image ids found under {root}")
        return 2

    canvas, ordered = build_grid(args.dataset, root, ids, args.tile_h,
                                  args.tile_w, label_h=18)
    if canvas is None:
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else FIG_DIR / f"grid_{args.dataset}.png"
    cv2.imwrite(str(out), canvas[:, :, ::-1])
    print(f"[grid] wrote {out} ({canvas.shape[1]}x{canvas.shape[0]} px)")

    # Record provenance in the same manifest make_figures.py writes, so the
    # grid's pixels trace back to the run that scored them.
    _record_provenance(out, args.dataset, images_root, ids, ordered)
    return 0


def _record_provenance(out: Path, dataset: str, images_root: Path,
                        ids: List[str], methods: List[str]) -> None:
    import json

    prov_path = REPO / "paper" / "07_provenance" / "figures_provenance.json"
    try:
        provenance = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    except json.JSONDecodeError:
        provenance = {}
    artefacts = provenance.setdefault("artefacts", {})
    artefacts[str(out)] = {
        "source": f"{images_root.name}/{dataset}__test/images/<method>/<id>.png",
        "kind": "qualitative grid (not a metric)",
        "methods": methods,
        "image_ids": ids,
        "generator": str(Path(__file__).resolve()),
    }
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"[grid] recorded provenance in {prov_path}")


if __name__ == "__main__":
    raise SystemExit(main())
