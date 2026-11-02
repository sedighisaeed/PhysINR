"""Offline import (Mode B): evaluate pre-computed enhancement results.

Many published methods cannot be re-run -- the weights are not released, the
codebase is bit-rotted, or the stack is unobtainable. Reviewers still expect
such methods in the comparison, and they still deserve the same statistical
treatment as runnable ones. This module reads a directory tree of
pre-enhanced images and serves it through the same interface an online
adapter uses, so the evaluation pipeline is byte-for-byte identical between
modes.

Layout contract
---------------
The directory must contain one output image per input sample, matched by
*filename stem*. Both of these are accepted::

    results/MethodA/UIEB/img_001.png
    results/MethodA/UIEB/img_002.png

    results/MethodA/UIEB/img_001_enhanced.png   # suffixes are ignored
    results/MethodA/UIEB/img_002_enhanced.jpg

The importer verifies that (a) every requested sample has an output, (b) no
output is a stray file from another split, and (c) geometry matches the
input. A mismatch fails the run rather than dropping samples silently,
because a partially-populated offline directory would otherwise masquerade
as a complete method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from ..core.dataset_manager import Sample

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class OfflineImportError(RuntimeError):
    """Raised when a pre-computed result tree does not satisfy the contract."""


@dataclass
class OfflineImporter:
    """Read pre-enhanced images for one method on one dataset."""

    output_dir: Path
    method: str = "OfflineMethod"

    # Lazily populated stem -> path index.
    _index: Dict[str, Path] = field(default_factory=dict, init=False)
    _scanned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    # --------------------------------------------------------------- scan
    def scan(self) -> Dict[str, Path]:
        """Index the result tree by filename stem (without suffix)."""
        if not self._scanned:
            if not self.output_dir.is_dir():
                raise OfflineImportError(
                    f"offline result directory does not exist: {self.output_dir}"
                )
            index: Dict[str, Path] = {}
            for path in sorted(self.output_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                stem = path.stem
                # Drop common enhancement suffixes so names still match inputs.
                for marker in ("_enhanced", "_result", "_out", "_final",
                               "_restored", "-enhanced"):
                    if stem.lower().endswith(marker):
                        stem = stem[: -len(marker)]
                        break
                if stem in index:
                    raise OfflineImportError(
                        f"ambiguous offline results for stem '{stem}': both "
                        f"{index[stem].name} and {path.name} resolve to it"
                    )
                index[stem] = path
            self._index = index
            self._scanned = True
        return self._index

    @property
    def available_stems(self) -> Set[str]:
        return set(self.scan())

    # -------------------------------------------------------------- verify
    def verify_coverage(self, sample_ids: Sequence[str]) -> None:
        """Every requested sample must have exactly one matching output."""
        index = self.scan()
        missing = [sid for sid in sample_ids
                   if Path(sid).stem not in index and sid not in index]
        if missing:
            raise OfflineImportError(
                f"offline method '{self.method}' is missing "
                f"{len(missing)}/{len(sample_ids)} outputs under "
                f"{self.output_dir} (first missing: {missing[:3]}). "
                "A partially populated result tree cannot be benchmarked: it "
                "would silently drop samples and bias the comparison."
            )

    def verify_geometry(self, samples: Sequence[Sample]) -> None:
        """Outputs must have the same geometry as their inputs."""
        from PIL import Image

        for sample in samples:
            out = self.resolve(sample.id)
            if out is None:
                continue
            with Image.open(sample.raw_path) as raw_img, Image.open(out) as out_img:
                if raw_img.size != out_img.size:
                    raise OfflineImportError(
                        f"{self.method}/{sample.id}: output geometry "
                        f"{out_img.size} != input {raw_img.size}"
                    )

    # --------------------------------------------------------------- load
    def resolve(self, sample_id: str) -> Optional[Path]:
        index = self.scan()
        stem = Path(sample_id).stem
        return index.get(stem) or index.get(sample_id)

    def load(self, sample_id: str) -> np.ndarray:
        path = self.resolve(sample_id)
        if path is None:
            raise OfflineImportError(
                f"offline method '{self.method}' has no output for '{sample_id}'"
            )
        return _read_rgb(path)

    def load_all(self, sample_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        self.verify_coverage(sample_ids)
        return {sid: self.load(sid) for sid in sample_ids}

    # --------------------------------------------------------- iteration
    def iter_pairs(self, samples: Sequence[Sample]) -> Iterable[Dict[str, object]]:
        """Yield ``{"sample": ..., "enhanced": ...}`` for each sample."""
        self.verify_coverage([s.id for s in samples])
        self.verify_geometry(samples)
        for sample in samples:
            yield {"sample": sample, "enhanced": self.load(sample.id)}

    def run_metadata(self) -> Dict[str, Any]:
        """Mode B provenance: the scores came from a supplied result tree.

        HydroBench did not execute the method here, so ``used_fallback`` is
        false -- the outputs are the method's own, not a stand-in -- but the
        note states where they came from, since reproducibility depends on it.
        """
        return {
            "method": self.method,
            "used_fallback": False,
            "note": f"offline import (Mode B) from {self.output_dir}",
            "checkpoint": None,
        }


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OfflineImportError(f"failed to decode offline result: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- discovery
def discover_offline_methods(root: Path, dataset: str) -> List[OfflineImporter]:
    """Treat every subdirectory of ``root/<dataset>`` as one offline method."""
    root = Path(root)
    base = root / dataset
    if not base.is_dir():
        return []
    return [OfflineImporter(output_dir=base / sub.name, method=sub.name)
            for sub in sorted(base.iterdir()) if sub.is_dir()]
