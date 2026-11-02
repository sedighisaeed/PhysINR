"""Immutable dataset manifests: checksums + explicit, leak-proof split lists.

A manifest is a plain JSON file pinned to a specific dataset snapshot. It is
the single source of truth for *which* files exist and *which* files belong to
which split. HydroBench refuses to evaluate anything until the on-disk tree
re-hashes to the manifest, which is what prevents silent dataset drift (e.g.
regenerated data, hand-edited test images) from contaminating results.

Schema (v1)
-----------
.. code-block:: json

    {
      "schema_version": 1,
      "dataset": "UIEB",
      "root": "absolute or env-relative root recorded at manifest time",
      "created_utc": "2024-01-01T00:00:00Z",
      "splits": {
        "train":     ["img_0.png", "img_1.png", ...],
        "val":       [...],
        "test":      [...],
        "challenge": [...]
      },
      "files": {
        "raw/img_0.png":     "sha256hex...",
        "reference/img_0.png": "sha256hex..."
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

SCHEMA_VERSION = 1

# Byte chunk size for streaming file hashes.
_HASH_CHUNK = 1 << 20

# Split that is never allowed to be empty for paired benchmarks.
_REQUIRED_SPLITS = ("train", "val", "test")


class ManifestError(RuntimeError):
    """Raised when a dataset tree does not match its manifest."""


def sha256_of_file(path: Path) -> str:
    """Stream a file through sha256 without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Manifest:
    """Parsed, in-memory representation of a dataset manifest."""

    dataset: str
    splits: Dict[str, List[str]]
    files: Dict[str, str]
    root: Optional[str] = None
    created_utc: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_path(cls, path: Path) -> "Manifest":
        path = Path(path)
        if not path.is_file():
            raise ManifestError(f"manifest not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
        if raw.get("schema_version", 1) != SCHEMA_VERSION:
            raise ManifestError(
                f"manifest {path} has schema_version={raw.get('schema_version')} but "
                f"HydroBench expects {SCHEMA_VERSION}"
            )
        if "splits" not in raw or "files" not in raw:
            raise ManifestError(f"manifest {path} is missing 'splits' or 'files'")
        return cls(
            dataset=raw.get("dataset", path.stem),
            splits={k: list(v) for k, v in raw["splits"].items()},
            files=dict(raw["files"]),
            root=raw.get("root"),
            created_utc=raw.get("created_utc"),
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            extra={k: v for k, v in raw.items()
                   if k not in {"dataset", "splits", "files", "root",
                                "created_utc", "schema_version"}},
        )

    def to_path(self, path: Path) -> Path:
        """Serialise the manifest (deterministic key order for reviewability)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "root": self.root,
            "created_utc": self.created_utc
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "splits": {k: sorted(v) for k, v in sorted(self.splits.items())},
            "files": dict(sorted(self.files.items())),
            **self.extra,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------- queries
    @property
    def all_ids(self) -> Set[str]:
        """Every sample id referenced by any split."""
        out: Set[str] = set()
        for ids in self.splits.values():
            out.update(ids)
        return out

    def split_of(self, sample_id: str) -> Optional[str]:
        for split, ids in self.splits.items():
            if sample_id in ids:
                return split
        return None

    def list_split(self, split: str) -> List[str]:
        if split not in self.splits:
            raise ManifestError(f"split '{split}' is not defined by manifest "
                                f"(available: {sorted(self.splits)})")
        return list(self.splits[split])

    # ----------------------------------------------------------- integrity
    def check_split_disjointness(self) -> None:
        """No sample id may appear in two splits -- the leakage invariant."""
        seen: Dict[str, str] = {}
        for split, ids in sorted(self.splits.items()):
            for sample_id in ids:
                if sample_id in seen:
                    raise ManifestError(
                        f"LEAKAGE: sample '{sample_id}' appears in both split "
                        f"'{seen[sample_id]}' and '{split}'"
                    )
                seen[sample_id] = split
        for split, ids in self.splits.items():
            if len(ids) != len(set(ids)):
                dupes = sorted({i for i in ids if ids.count(i) > 1})
                raise ManifestError(
                    f"split '{split}' contains duplicate ids: {dupes[:5]}"
                )

    def check_required_splits(self, require: Sequence[str] = _REQUIRED_SPLITS) -> None:
        missing = [s for s in require if not self.splits.get(s)]
        if missing:
            raise ManifestError(
                f"manifest for '{self.dataset}' is missing required splits: {missing}"
            )

    def check_file_coverage(self) -> None:
        """Every file referenced by a split must have a manifest entry."""
        referenced = {f for f in self.files}
        for sample_id in self.all_ids:
            # Compare stems too: a dataset may pair a .jpg input with a .bmp
            # reference under the same id, so the exact name differs even
            # though the sample is covered.
            stem = Path(sample_id).stem
            matches = [f for f in referenced if Path(f).name == sample_id
                       or Path(f).stem == sample_id or Path(f).stem == stem]
            if not matches:
                # A sample id must resolve to at least one hashed file.
                raise ManifestError(
                    f"sample '{sample_id}' has no hashed file entry in manifest"
                )

    def validate(self, require_splits: bool = True) -> None:
        """Run every self-consistency check (no filesystem access required)."""
        if require_splits:
            self.check_required_splits()
        self.check_split_disjointness()
        self.check_file_coverage()

    # ------------------------------------------------- filesystem verification
    def verify_tree(self, root: Path, strict: bool = True,
                    progress: bool = False) -> "VerificationReport":
        """Re-hash the on-disk tree and compare against this manifest.

        Parameters
        ----------
        root : Path
            Dataset root directory. Manifest paths are interpreted relative
            to it.
        strict : bool
            If True (default), any mismatch raises :class:`ManifestError`.
            If False, mismatches are collected into the returned report.
        progress : bool
            Emit a tqdm progress bar while hashing.
        """
        root = Path(root)
        report = VerificationReport(dataset=self.dataset, root=root)
        iterator: Any = sorted(self.files)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"verify {self.dataset}",
                                unit="file")
            except Exception:  # tqdm is optional
                pass

        for rel in iterator:
            expected = self.files[rel]
            disk_path = root / rel
            if not disk_path.is_file():
                report.missing.append(rel)
                continue
            actual = sha256_of_file(disk_path)
            if actual != expected:
                report.hash_mismatch.append(rel)
            else:
                report.verified += 1

        # Files on disk that the manifest does not know about. These do not
        # invalidate results by themselves, but they are reported so that a
        # reviewer can tell the benchmark saw extra data.
        for rel in sorted(_walk_tree(root)):
            if rel not in self.files:
                report.untracked.append(rel)

        if strict and not report.ok:
            raise ManifestError(str(report))
        return report


@dataclass
class VerificationReport:
    dataset: str
    root: Path
    verified: int = 0
    missing: List[str] = field(default_factory=list)
    hash_mismatch: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.hash_mismatch

    def __str__(self) -> str:
        lines = [f"Manifest verification FAILED for '{self.dataset}' ({self.root}):"]
        if self.missing:
            lines.append(f"  missing files ({len(self.missing)}): "
                         f"{self.missing[:5]}{'...' if len(self.missing) > 5 else ''}")
        if self.hash_mismatch:
            lines.append(f"  checksum mismatches ({len(self.hash_mismatch)}): "
                         f"{self.hash_mismatch[:5]}{'...' if len(self.hash_mismatch) > 5 else ''}")
        lines.append(f"  verified: {self.verified}, untracked: {len(self.untracked)}")
        return "\n".join(lines)


def _walk_tree(root: Path) -> Set[str]:
    """All files under *root* as manifest-relative posix paths."""
    out: Set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            rel = Path(dirpath, name).relative_to(root).as_posix()
            out.add(rel)
    return out


# ------------------------------------------------------------------ builders
# Extensions tried when a sample id's ground truth does not carry one and the
# exact name is absent -- SUIM, for instance, pairs .jpg images with .bmp masks.
_LOCATE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _resolve_sibling(directory: Path, sample_id: str,
                     suffixes: Sequence[str] = _LOCATE_SUFFIXES) -> Optional[Path]:
    """Find the file backing *sample_id* under *directory*.

    Sample ids may or may not carry an extension; both are handled, matching
    :func:`hydro_bench.core.dataset_manager._locate` so a manifest built here
    verifies the same file the benchmark later loads.
    """
    direct = directory / sample_id
    if direct.is_file():
        return direct
    stem = Path(sample_id).stem
    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_manifest(dataset: str, root: Path, splits: Dict[str, List[str]],
                   raw_subdir: str, gt_subdir: Optional[str] = None,
                   out_path: Optional[Path] = None,
                   progress: bool = False) -> Manifest:
    """Hash a dataset tree and emit a manifest pinned to it.

    Use this **once**, right after obtaining a dataset, then commit the output
    to version control. Re-running it on a modified tree *must* change the
    hashes -- that is the point.

    A sample id need not carry the extension of its ground truth: the file is
    resolved by stem when the exact name is missing, so a dataset whose inputs
    are ``.jpg`` and whose reference images are ``.bmp`` still hashes the pair.
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"cannot build manifest: {root} is not a directory")

    files: Dict[str, str] = {}
    sample_ids: Set[str] = set()
    for ids in splits.values():
        sample_ids.update(ids)

    for sample_id in sorted(sample_ids):
        raw_path = _resolve_sibling(root / raw_subdir, sample_id)
        if raw_path is None:
            raise ManifestError(
                f"raw sample missing on disk: {raw_subdir}/{sample_id}")
        files[raw_path.relative_to(root).as_posix()] = sha256_of_file(raw_path)
        if gt_subdir is not None:
            gt_path = _resolve_sibling(root / gt_subdir, sample_id)
            if gt_path is None:
                raise ManifestError(
                    f"ground-truth sample missing on disk: "
                    f"{gt_subdir}/{sample_id}")
            files[gt_path.relative_to(root).as_posix()] = sha256_of_file(gt_path)

    manifest = Manifest(
        dataset=dataset,
        splits={k: sorted(v) for k, v in splits.items()},
        files=files,
        root=str(root),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest.validate(require_splits=False)
    if out_path is not None:
        manifest.to_path(out_path)
    return manifest
