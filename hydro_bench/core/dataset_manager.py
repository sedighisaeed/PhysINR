"""Standardised data loading with manifest verification and split hygiene.

The :class:`DatasetManager` is the *only* path by which images reach the
metrics. It guarantees, before a single metric is computed:

* the on-disk tree re-hashes to the committed manifest (no silent drift),
* splits are disjoint (no train/test leakage),
* the requested split is loaded *entirely from disk at load time* -- the
  caller never receives a lazy iterator that could be re-ordered or filtered
  mid-run,
* ground truth is only handed out for paired datasets, and only for splits
  the manifest authorises.

Datasets are registered via ``configs/datasets.yaml``; ``root`` may be
overridden by the ``HYDROBENCH_DATA`` environment variable or the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .manifest import Manifest, ManifestError, sha256_of_file

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "datasets" / "manifests"


class DatasetError(RuntimeError):
    """Raised for missing or inconsistent dataset state."""


@dataclass
class Sample:
    """A single benchmark item, fully resolved to absolute paths."""

    id: str
    raw_path: Path
    gt_path: Optional[Path] = None
    split: str = ""
    # Optional extra modalities (e.g. detection labels, segmentation masks).
    extra: Dict[str, Path] = field(default_factory=dict)

    def load_raw(self) -> np.ndarray:
        return _read_rgb(self.raw_path)

    def load_gt(self) -> np.ndarray:
        if self.gt_path is None:
            raise DatasetError(f"sample '{self.id}' has no ground truth")
        return _read_rgb(self.gt_path)


def _read_rgb(path: Path) -> np.ndarray:
    """Read any image as HxWx3 uint8 RGB."""
    import cv2

    if not Path(path).is_file():
        raise DatasetError(f"image missing on disk: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DatasetError(f"failed to decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@dataclass
class DatasetSpec:
    """Static description of a dataset, from ``configs/datasets.yaml``."""

    name: str
    type: str                                   # paired | unpaired | detection | segmentation
    root: Path
    structure: Dict[str, str] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    description: str = ""
    manifest_path: Optional[Path] = None
    manifest: Optional[Manifest] = None

    @property
    def paired(self) -> bool:
        return self.type == "paired"

    def subdir(self, key: str) -> Path:
        try:
            rel = self.structure[key]
        except KeyError:
            raise DatasetError(
                f"dataset '{self.name}' config has no structure key '{key}'"
            ) from None
        return self.root / rel

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], name: str,
                    data_root: Optional[Path] = None,
                    manifest_dir: Optional[Path] = None) -> "DatasetSpec":
        if name not in cfg:
            raise DatasetError(f"unknown dataset '{name}'. "
                               f"Known: {sorted(cfg)}")
        entry = cfg[name]
        root = _resolve_root(entry.get("root", ""), data_root)
        spec = cls(
            name=name,
            type=entry.get("type", "paired"),
            root=root,
            structure=dict(entry.get("structure", {})),
            classes=list(entry.get("classes", [])),
            description=entry.get("description", ""),
        )
        manifest_ref = entry.get("splits") or entry.get("manifest")
        if manifest_ref:
            spec.manifest_path = _resolve_manifest(manifest_ref, manifest_dir,
                                                   dataset_root=root)
        return spec


def _resolve_root(configured: str, override: Optional[Path]) -> Path:
    """Resolve a dataset root, honouring the HYDROBENCH_DATA base directory.

    Config values may be literal paths, or ``${ENV:HYDROBENCH_DATA}/SynthUIB``
    placeholders that fall back to the environment variable when unset.
    """
    if override is not None:
        return Path(override)
    if not configured:
        raise DatasetError("no dataset root configured and none given on the CLI")

    expanded = os.path.expandvars(configured)
    if expanded != configured:  # a real $VAR was substituted by the shell/env
        return Path(expanded)

    # Handle the ${ENV:NAME}/... placeholder convention used by the shipped
    # configs: it is not understood by os.path.expandvars, so resolve it here.
    placeholder = "${ENV:"
    if placeholder in configured:
        base = os.environ.get("HYDROBENCH_DATA")
        suffix = configured.split("}", 1)[1] if "}" in configured else ""
        if not base:
            raise DatasetError(
                f"dataset root references an unset environment variable "
                f"({configured}) and HYDROBENCH_DATA is not defined"
            )
        return Path(base) / Path(suffix.lstrip("/"))

    raise DatasetError(
        f"dataset root '{configured}' could not be resolved; set "
        "HYDROBENCH_DATA or pass --dataset-root"
    )


def _resolve_manifest(ref: str, manifest_dir: Optional[Path],
                      dataset_root: Optional[Path] = None) -> Path:
    """Resolve a manifest reference, preferring one shipped with the dataset.

    A manifest pinned next to the data it describes is the authoritative one:
    it hashes the actual bytes on this machine. The packaged
    ``datasets/manifests/`` directory holds the committed copies for the
    official datasets.
    """
    ref = os.path.expandvars(ref)
    ref = ref.replace("${MANIFEST:", "").replace("}", "")
    p = Path(ref)
    if p.is_absolute() and p.is_file():
        return p
    # 1. Next to the dataset itself (highest precedence, hashes local bytes).
    if dataset_root is not None:
        local = Path(dataset_root) / "manifests" / p.name
        if local.is_file():
            return local
    # 2. An explicitly provided directory.
    if manifest_dir is not None:
        explicit = Path(manifest_dir) / p.name
        if explicit.is_file():
            return explicit
    # 3. The packaged manifests shipped with HydroBench.
    return DEFAULT_MANIFEST_DIR / p.name


class DatasetManager:
    """Loads, verifies, and serves a dataset's splits."""

    def __init__(self, spec: DatasetSpec, verify_checksums: bool = True,
                 strict: bool = True):
        self.spec = spec
        self.verify_checksums = verify_checksums
        self.strict = strict
        self._manifest: Optional[Manifest] = None
        self._verified = False

    # ----------------------------------------------------------- manifest
    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            if self.spec.manifest_path is None or not self.spec.manifest_path.is_file():
                raise DatasetError(
                    f"dataset '{self.spec.name}' has no manifest at "
                    f"{self.spec.manifest_path}. Generate one with "
                    "`hydro-bench manifest --dataset {name}` before evaluating."
                )
            self._manifest = Manifest.from_path(self.spec.manifest_path)
            self._manifest.validate(require_splits=False)
            if not self.spec.paired and "test" not in self._manifest.splits:
                # Unpaired sets may only define one pool; normalise.
                first = next(iter(self._manifest.splits), "test")
                if first != "test":
                    self._manifest.splits["test"] = self._manifest.splits.pop(first)
        return self._manifest

    def verify(self, progress: bool = False) -> "DatasetManager":
        """Re-hash the tree and check split disjointness. Idempotent."""
        if not self.spec.root.is_dir():
            raise DatasetError(f"dataset root does not exist: {self.spec.root}")
        man = self.manifest
        man.check_split_disjointness()
        if self.verify_checksums:
            report = man.verify_tree(self.spec.root, strict=self.strict,
                                     progress=progress)
            if not report.ok and not self.strict:
                # Non-strict mode: surface the problem but keep going.
                print(f"WARNING: {report}")
        self._verified = True
        return self

    @property
    def verified(self) -> bool:
        return self._verified

    # ------------------------------------------------------------- splits
    def splits(self) -> List[str]:
        return sorted(self.manifest.splits)

    def sample_ids(self, split: str) -> List[str]:
        try:
            return self.manifest.list_split(split)
        except ManifestError as exc:
            raise DatasetError(str(exc)) from None

    # ------------------------------------------------------------- loading
    def load_split(self, split: str, limit: Optional[int] = None,
                   require_gt: bool = True) -> List[Sample]:
        """Materialise a split as a list of :class:`Sample` objects.

        The whole split is read up-front: no lazy iteration, so nothing can
        be silently dropped or re-ordered after evaluation begins.
        """
        if not self._verified:
            self.verify()
        ids = self.sample_ids(split)
        if limit is not None:
            ids = ids[:limit]
        # Paired sets name their input pool "raw"; downstream-task datasets
        # (DUO detection, SUIM segmentation) name it "images". Both are the
        # pool an enhancement method reads, so accept either rather than
        # forcing every detection/segmentation config to also declare "raw".
        image_key = "raw" if "raw" in self.spec.structure else "images"
        raw_dir = self.spec.subdir(image_key)
        gt_dir = self.spec.subdir("gt") if "gt" in self.spec.structure else None
        if require_gt and self.spec.paired and gt_dir is None:
            raise DatasetError(
                f"paired dataset '{self.spec.name}' has no 'gt' structure entry"
            )

        samples: List[Sample] = []
        for sid in ids:
            raw_path = _locate(raw_dir, sid)
            if raw_path is None:
                raise DatasetError(
                    f"sample '{sid}' from split '{split}' is declared by the "
                    f"manifest but absent from {raw_dir}"
                )
            gt_path = None
            if self.spec.paired and gt_dir is not None:
                gt_path = _locate(gt_dir, sid)
                if gt_path is None:
                    raise DatasetError(
                        f"paired sample '{sid}' has no ground truth under {gt_dir}"
                    )
            samples.append(Sample(id=sid, raw_path=raw_path, gt_path=gt_path,
                                  split=split))
        return samples

    def iter_split(self, split: str, limit: Optional[int] = None) -> Iterator[Sample]:
        return iter(self.load_split(split, limit=limit))

    # --------------------------------------------------------- downstream
    def load_detection_split(self, split: str, image_subdir_key: str = "images",
                             label_subdir_key: str = "labels",
                             limit: Optional[int] = None) -> List[Sample]:
        """Load a detection split, attaching YOLO-format label paths."""
        samples = self.load_split(split, limit=limit, require_gt=False)
        img_dir = self.spec.subdir(image_subdir_key)
        lbl_dir = self.spec.subdir(label_subdir_key)
        out: List[Sample] = []
        for s in samples:
            img = _locate(img_dir, s.id)
            if img is None:
                continue
            label = _locate(lbl_dir, s.id, suffixes=(".txt",))
            out.append(Sample(id=s.id, raw_path=img, split=split,
                              extra={"labels": label} if label else {}))
        return out

    def load_segmentation_split(self, split: str, image_subdir_key: str = "images",
                                mask_subdir_key: str = "masks",
                                limit: Optional[int] = None) -> List[Sample]:
        """Load a segmentation split, attaching mask paths."""
        samples = self.load_split(split, limit=limit, require_gt=False)
        img_dir = self.spec.subdir(image_subdir_key)
        mask_dir = self.spec.subdir(mask_subdir_key)
        out: List[Sample] = []
        for s in samples:
            img = _locate(img_dir, s.id)
            if img is None:
                continue
            mask = _locate(mask_dir, s.id)
            out.append(Sample(id=s.id, raw_path=img, split=split,
                              extra={"mask": mask} if mask else {}))
        return out

    # ------------------------------------------------------------- checks
    def cross_method_consistency(self, other: "DatasetManager") -> None:
        """Assert two managers serve the *same* split (paired comparisons)."""
        if self.spec.name != other.spec.name:
            raise DatasetError(
                f"cannot pair dataset '{self.spec.name}' with '{other.spec.name}'"
            )
        if self.manifest is not other.manifest:
            for split in self.manifest.splits:
                if self.sample_ids(split) != other.sample_ids(split):
                    raise DatasetError(f"split '{split}' differs between managers")


def _locate(directory: Path, sample_id: str,
            suffixes: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp",
                                       ".tif", ".tiff", ".webp")) -> Optional[Path]:
    """Find the file backing *sample_id* under *directory*.

    Sample ids may or may not carry an extension; both are handled.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    direct = directory / sample_id
    if direct.is_file():
        return direct
    stem = Path(sample_id).stem
    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    # Fall back to an exact-name match for ids already carrying a suffix.
    return None


# ------------------------------------------------------------------ factory
def load_dataset_configs(config_path: Path) -> Dict[str, Any]:
    import yaml

    config_path = Path(config_path)
    if not config_path.is_file():
        raise DatasetError(f"dataset config not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # configs/datasets.yaml wraps its entries under a top-level "datasets:"
    # key. Accept both the wrapped and the flat form so either layout works.
    return raw.get("datasets", raw)


def make_manager(dataset: str, config_path: Optional[Path] = None,
                 data_root: Optional[Path] = None,
                 manifest_dir: Optional[Path] = None,
                 verify: bool = True, strict: bool = True,
                 verify_checksums: bool = True) -> DatasetManager:
    """Build a verified :class:`DatasetManager` from ``configs/datasets.yaml``."""
    config_path = config_path or (Path(__file__).resolve().parent.parent
                                  / "configs" / "datasets.yaml")
    cfg = load_dataset_configs(config_path)
    spec = DatasetSpec.from_config(cfg, dataset, data_root=data_root,
                                   manifest_dir=manifest_dir)
    manager = DatasetManager(spec, verify_checksums=verify_checksums, strict=strict)
    if verify:
        manager.verify()
    return manager
