"""I/O, seeding, device selection and the deduplication pass that keeps the
meta-training source honest.

The deduplication here matters for the paper's integrity claim.  The clean
source photographs used for meta-training come from a real dataset, so before
any of them are used we check them against the *test* images of every
evaluation set and drop near-duplicates.  Without this, a meta-initialisation
could quietly have seen content that later appears in the test split, and the
zero-shot claim would be false by construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# ------------------------------------------------------------------- device --
def resolve_device(device: str = "auto") -> str:
    """``auto`` -> mps when available, else cpu.

    The machine this project runs on has no CUDA and no conda; MPS is the only
    accelerator, and it has a hard memory ceiling, so callers must be able to
    fall back to CPU.  See the memory note on environment constraints.
    """
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Torch's CUDA seeding is skipped deliberately: no CUDA on this machine.


# ------------------------------------------------------------------ logging --
def get_logger(name: str, log_dir: Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------- image I/O --
def load_image(path: Path, max_side: int | None = None) -> np.ndarray:
    """Read an image as HxWx3 float32 in [0, 1]."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_side is not None and max(img.size) > max_side:
            ratio = max_side / max(img.size)
            size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
            img = img.resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    """HxWx3 numpy -> (1,3,H,W) tensor."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {image.shape}")
    t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return t.unsqueeze(0).to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) -> HxWx3 numpy in [0,1]."""
    arr = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return arr[0].transpose(1, 2, 0)


# ------------------------------------------------------------------ hashes --
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phash(image: np.ndarray, hash_size: int = 8,
          highfreq: int = 8) -> np.ndarray:
    """Perceptual hash: the sign of the low-frequency DCT coefficients.

    Robust to rescaling and mild JPEG artefacts, which is exactly the
    duplicate class that matters when two datasets were scraped from
    overlapping web sources.

    Returns ``highfreq**2 - 1`` bits (63 at the defaults): the DC coefficient
    is excluded from the comparison, since it encodes only the global mean
    brightness and would make the hash sensitive to exposure rather than
    content. ``deduplicate``'s Hamming threshold is calibrated against this
    width, so changing ``highfreq`` invalidates it.
    """
    from scipy.fftpack import dct

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h, w = img.shape[0], img.shape[1]
    if h < hash_size or w < hash_size:
        # An axis shorter than the grid cannot be divided into eight blocks, so
        # reflect-pad it to the minimum. Better than dropping the source: a
        # tiny image is still worth checking against the reference set.
        img = np.pad(img,
                     ((0, max(0, hash_size - h)), (0, max(0, hash_size - w))),
                     mode="reflect")
        h, w = img.shape
    # Resize to a square via area-style averaging by block means. The block
    # size is per axis: a portrait image has a smaller width than height, so
    # one shared block computed from the height slices past the width and the
    # reshape comes up short. Portrait sources are 10% of the DIV2K pool, and
    # dropping them would both shrink the pool and leave them un-checked
    # against the reference set.
    bh = max(1, h // hash_size)
    bw = max(1, w // hash_size)
    small = img[: bh * hash_size, : bw * hash_size].reshape(
        hash_size, bh, hash_size, bw).mean(axis=(1, 3))
    dct_low = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    coeffs = dct_low[:highfreq, :highfreq].flatten()
    # Drop the DC term before thresholding (see the docstring).
    return (coeffs[1:] > np.median(coeffs[1:])).astype(np.uint8)


def phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def deduplicate(source_dir: Path, reference_sets: Sequence[Tuple[str, Sequence[Path]]],
                threshold: int = 5, max_side: int = 128,
                logger: logging.Logger | None = None) -> Dict[str, object]:
    """Drop source images that duplicate any reference (evaluation) image.

    Parameters
    ----------
    source_dir : Path
        Directory of candidate clean images for meta-training.
    reference_sets : sequence of (name, paths)
        The test-split images of every dataset the method will be evaluated
        on.  Matches against these are what keeps the zero-shot claim honest.
    threshold : int
        Hamming distance at or below which two 8x8 hashes count as the same
        image (64 bits; 5 bits is a conventional loose threshold).

    Returns
    -------
    dict
        ``kept`` / ``dropped`` path lists, plus per-reference-set drop counts,
        suitable for the provenance record.
    """
    source_dir = Path(source_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    sources = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in exts)

    ref_hashes: List[Tuple[str, str, np.ndarray]] = []
    for name, paths in reference_sets:
        for p in paths:
            try:
                ref_hashes.append((name, str(p), phash(load_image(p, max_side))))
            except Exception as exc:  # a corrupt reference must not kill the pass
                if logger:
                    logger.warning(f"could not hash reference {p}: {exc}")

    kept: List[Path] = []
    dropped: List[Tuple[Path, str, str]] = []
    for src in sources:
        try:
            h = phash(load_image(src, max_side))
        except Exception as exc:
            if logger:
                logger.warning(f"could not hash source {src}: {exc}")
            continue
        match = None
        for ref_name, ref_path, ref_h in ref_hashes:
            if phash_distance(h, ref_h) <= threshold:
                match = (ref_name, ref_path)
                break
        if match is None:
            kept.append(src)
        else:
            dropped.append((src, match[0], match[1]))

    counts = {name: sum(1 for _, ref_name, _ in dropped if ref_name == name)
              for name, _ in reference_sets}
    if logger:
        logger.info(
            f"dedup: kept {len(kept)}/{len(sources)} meta-training sources; "
            f"dropped {len(dropped)} as duplicates of evaluation images "
            f"({counts})"
        )
    return {"kept": kept, "dropped": dropped, "drop_counts": counts,
            "threshold": threshold}


# ------------------------------------------------------------------- proven.--
def write_provenance(path: Path, record: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


__all__ = [
    "resolve_device", "seed_everything", "get_logger",
    "load_image", "to_uint8", "to_tensor", "to_numpy",
    "sha256_file", "phash", "phash_distance", "deduplicate", "write_provenance",
]
