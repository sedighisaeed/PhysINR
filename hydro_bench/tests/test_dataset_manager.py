"""Manifest integrity and dataset management: the zero-leakage guarantees."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hydro_bench.core import manifest as manifest_mod
from hydro_bench.core.dataset_manager import (DatasetError, DatasetManager,
                                              DatasetSpec, make_manager)


class TestManifest:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "m.json"
        man = manifest_mod.Manifest(dataset="X",
                                    splits={"train": ["a"], "test": ["b"]},
                                    files={"raw/a": "deadbeef"})
        man.to_path(path)
        loaded = manifest_mod.Manifest.from_path(path)
        assert loaded.dataset == "X"
        assert loaded.splits == {"train": ["a"], "test": ["b"]}
        assert loaded.files == {"raw/a": "deadbeef"}

    def test_disjoint_splits_are_required(self):
        man = manifest_mod.Manifest(dataset="X",
                                    splits={"train": ["a"], "test": ["a"]},
                                    files={"raw/a": "deadbeef"})
        with pytest.raises(manifest_mod.ManifestError, match="LEAKAGE"):
            man.check_split_disjointness()

    def test_duplicate_ids_rejected(self):
        man = manifest_mod.Manifest(dataset="X", splits={"train": ["a", "a"]},
                                    files={"raw/a": "x"})
        with pytest.raises(manifest_mod.ManifestError, match="LEAKAGE"):
            man.check_split_disjointness()

    def test_missing_file_entry_rejected(self):
        man = manifest_mod.Manifest(dataset="X", splits={"test": ["ghost"]},
                                    files={})
        with pytest.raises(manifest_mod.ManifestError, match="no hashed file"):
            man.validate(require_splits=False)

    def test_bad_schema_version(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"schema_version": 99, "splits": {},
                                    "files": {}}))
        with pytest.raises(manifest_mod.ManifestError, match="schema_version"):
            manifest_mod.Manifest.from_path(path)

    def test_missing_manifest_file(self, tmp_path):
        with pytest.raises(manifest_mod.ManifestError, match="not found"):
            manifest_mod.Manifest.from_path(tmp_path / "nope.json")

    def test_sha256_of_file_matches_bytes(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello")
        assert manifest_mod.sha256_of_file(path) == manifest_mod.sha256_of_bytes(b"hello")

    def test_build_and_verify_round_trip(self, tmp_path):
        raw = tmp_path / "raw"
        gt = tmp_path / "gt"
        raw.mkdir()
        gt.mkdir()
        for i in range(3):
            (raw / f"i{i}.png").write_bytes(b"raw")
            (gt / f"i{i}.png").write_bytes(b"gt")
        man = manifest_mod.build_manifest(
            "X", tmp_path, {"test": [f"i{i}.png" for i in range(3)]},
            raw_subdir="raw", gt_subdir="gt")
        report = man.verify_tree(tmp_path)
        assert report.ok
        assert report.verified == 6  # 3 raw + 3 gt

    def test_tampered_file_is_detected(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "i0.png").write_bytes(b"original")
        man = manifest_mod.build_manifest("X", tmp_path, {"test": ["i0.png"]},
                                          raw_subdir="raw")
        (raw / "i0.png").write_bytes(b"TAMPERED")
        report = man.verify_tree(tmp_path, strict=False)
        assert not report.ok
        assert report.hash_mismatch == ["raw/i0.png"]

    def test_missing_file_is_detected(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "i0.png").write_bytes(b"x")
        man = manifest_mod.build_manifest("X", tmp_path, {"test": ["i0.png"]},
                                          raw_subdir="raw")
        (raw / "i0.png").unlink()
        report = man.verify_tree(tmp_path, strict=False)
        assert report.missing == ["raw/i0.png"]

    def test_extra_files_reported_as_untracked(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "i0.png").write_bytes(b"x")
        (raw / "stray.png").write_bytes(b"y")
        man = manifest_mod.build_manifest("X", tmp_path, {"test": ["i0.png"]},
                                          raw_subdir="raw")
        report = man.verify_tree(tmp_path, strict=False)
        assert "raw/stray.png" in report.untracked


class TestDatasetManager:
    def test_verification_passes(self, synth_root):
        manager = make_manager("SynthUIB", verify=True, verify_checksums=True)
        assert manager.verified
        assert set(manager.splits()) >= {"train", "val", "test"}

    def test_splits_are_disjoint(self, synth_root):
        manager = make_manager("SynthUIB")
        train = set(manager.sample_ids("train"))
        test = set(manager.sample_ids("test"))
        val = set(manager.sample_ids("val"))
        assert not (train & test)
        assert not (train & val)
        assert not (val & test)

    def test_load_split_returns_samples_with_gt(self, synth_root):
        manager = make_manager("SynthUIB")
        samples = manager.load_split("test", limit=3)
        assert len(samples) == 3
        for sample in samples:
            assert sample.gt_path is not None
            assert sample.raw_path.is_file()
            raw = sample.load_raw()
            assert raw.shape[2] == 3
            assert raw.dtype == np.uint8

    def test_limit_truncates(self, synth_root):
        manager = make_manager("SynthUIB")
        assert len(manager.load_split("test", limit=2)) == 2

    def test_unknown_dataset(self, synth_root):
        with pytest.raises(DatasetError, match="unknown dataset"):
            make_manager("DoesNotExist")

    def test_unknown_split(self, synth_root):
        manager = make_manager("SynthUIB")
        with pytest.raises(DatasetError):
            manager.sample_ids("nope")

    def test_checksum_verification_catches_tampering(self, synth_dataset, tmp_path):
        # Work on an isolated copy: the session fixture must stay pristine for
        # every other test in the suite.
        base = tmp_path / "tampered"
        root = base / "SynthUIB"
        shutil.copytree(synth_dataset, root)
        manifests = root / "manifests"
        # data_root replaces the configured root wholesale, so point it at the
        # dataset directory itself (base/SynthUIB), not at the parent.
        manager = make_manager("SynthUIB", verify=False, data_root=root,
                               manifest_dir=manifests)
        sample = manager.load_split("test", limit=1)[0]
        original = sample.raw_path.read_bytes()
        try:
            sample.raw_path.write_bytes(b"corrupted")
            with pytest.raises(manifest_mod.ManifestError):
                make_manager("SynthUIB", verify=True, verify_checksums=True,
                             data_root=root,
                             manifest_dir=manifests).load_split("test")
        finally:
            sample.raw_path.write_bytes(original)

    def test_manifest_missing_raises(self, synth_root, monkeypatch):
        manager = make_manager("SynthUIB", verify=False)
        # Point at a nonexistent manifest.
        manager.spec.manifest_path = Path("/nonexistent/manifest.json")
        manager._manifest = None
        with pytest.raises(DatasetError, match="no manifest"):
            manager.load_split("test")

    def test_root_override(self, synth_root):
        manager = make_manager("SynthUIB", verify=False,
                               data_root=synth_root / "SynthUIB")
        assert manager.spec.root == synth_root / "SynthUIB"

    def test_downstream_loaders(self, synth_root):
        # The segmentation/detection loaders share the same split machinery.
        manager = make_manager("SynthUIB")
        samples = manager.load_split("test", limit=2)
        assert len(samples) == 2
