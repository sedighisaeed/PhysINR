"""The ``manifest`` subcommand: how a dataset becomes a pinned, split manifest.

Two properties matter enough to pin down with tests:

* a fractional split spec (``train=0.7``) partitions *one* seeded permutation,
  so the splits are disjoint by construction rather than disjoint by
  inspection -- an overlapping split would be a silent leak into the test set;
* the same seed must reproduce the same partition, because the manifest is
  committed and a reviewer needs to be able to rebuild it.
"""

from __future__ import annotations

import json

import pytest

from hydro_bench.core.manifest import ManifestError, build_manifest
from hydro_bench.run_benchmark import _parse_split_specs


def _dataset(root, n=20, raw="raw", gt=None, raw_ext=".png", gt_ext=".png"):
    root.mkdir(parents=True, exist_ok=True)
    (root / raw).mkdir(parents=True, exist_ok=True)
    if gt:
        (root / gt).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / raw / f"img_{i:03d}{raw_ext}").write_bytes(b"raw")
        if gt:
            (root / gt / f"img_{i:03d}{gt_ext}").write_bytes(b"gt")


class TestFractionalSplits:
    def test_fractions_partition_the_whole_set(self, tmp_path):
        _dataset(tmp_path, n=20)
        splits = _parse_split_specs(["train=0.7", "val=0.15", "test=0.15"],
                                    tmp_path, "raw", seed=20260926)
        total = sum(len(v) for v in splits.values())
        assert total == 20, "every id must land in exactly one split"
        assert splits["train"] == sorted(splits["train"])

    def test_fractions_are_disjoint(self, tmp_path):
        _dataset(tmp_path, n=20)
        splits = _parse_split_specs(["train=0.7", "val=0.15", "test=0.15"],
                                    tmp_path, "raw", seed=20260926)
        seen = []
        for ids in splits.values():
            seen.extend(ids)
        assert len(seen) == len(set(seen)), "a sample appears in two splits"

    def test_the_same_seed_reproduces_the_same_partition(self, tmp_path):
        _dataset(tmp_path, n=20)
        first = _parse_split_specs(["train=0.7", "val=0.15", "test=0.15"],
                                   tmp_path, "raw", seed=20260926)
        second = _parse_split_specs(["train=0.7", "val=0.15", "test=0.15"],
                                    tmp_path, "raw", seed=20260926)
        assert first == second

    def test_a_different_seed_changes_the_partition(self, tmp_path):
        _dataset(tmp_path, n=40)
        first = _parse_split_specs(["train=0.5", "test=0.5"], tmp_path,
                                   "raw", seed=1)
        second = _parse_split_specs(["train=0.5", "test=0.5"], tmp_path,
                                    "raw", seed=2)
        assert first["train"] != second["train"]

    def test_fractions_match_the_uieb_protocol(self, tmp_path):
        """890 images at 70/15/15 split 623/133/134, as the UIEB manifest does.

        Truncation (not rounding) is what makes the last split absorb the
        remainder; rounding would hand the extra sample to val instead.
        """
        _dataset(tmp_path, n=890)
        splits = _parse_split_specs(["train=0.7", "val=0.15", "test=0.15"],
                                    tmp_path, "raw", seed=20260926)
        assert (len(splits["train"]), len(splits["val"]),
                len(splits["test"])) == (623, 133, 134)

    def test_all_and_count_specs_still_work(self, tmp_path):
        _dataset(tmp_path, n=10)
        splits = _parse_split_specs(["train=ALL", "test=3"], tmp_path, "raw",
                                    seed=0)
        assert len(splits["train"]) == 10
        assert splits["test"] == splits["train"][:3]

    def test_fraction_out_of_range_is_rejected(self, tmp_path):
        _dataset(tmp_path, n=4)
        with pytest.raises(ValueError, match="out of range"):
            _parse_split_specs(["train=1.5"], tmp_path, "raw", seed=0)

    def test_malformed_spec_is_rejected(self, tmp_path):
        _dataset(tmp_path, n=4)
        with pytest.raises(ValueError, match="malformed split spec"):
            _parse_split_specs(["=0.5"], tmp_path, "raw", seed=0)

    def test_missing_id_file_is_rejected(self, tmp_path):
        _dataset(tmp_path, n=4)
        with pytest.raises(ValueError, match="id file not found"):
            _parse_split_specs([f"test={tmp_path / 'nope.txt'}"], tmp_path,
                               "raw", seed=0)

    def test_id_file_is_read_verbatim(self, tmp_path):
        _dataset(tmp_path, n=4)
        ids = tmp_path / "ids.txt"
        ids.write_text("img_000.png\n\n img_003.png \n")
        splits = _parse_split_specs([f"test={ids}"], tmp_path, "raw", seed=0)
        assert splits["test"] == ["img_000.png", "img_003.png"]


class TestExtensionTolerantHashing:
    def test_a_jpg_input_pairs_with_a_bmp_reference(self, tmp_path):
        """SUIM pairs .jpg images with .bmp masks under the same stem.

        The manifest must hash both, and must resolve the reference by stem
        rather than by the input's extension.
        """
        _dataset(tmp_path, n=3, raw="images", gt="masks",
                 raw_ext=".jpg", gt_ext=".bmp")
        manifest = build_manifest(dataset="X", root=tmp_path,
                                  splits={"test": ["img_000.jpg",
                                                   "img_001.jpg",
                                                   "img_002.jpg"]},
                                  raw_subdir="images", gt_subdir="masks")
        assert "images/img_000.jpg" in manifest.files
        assert "masks/img_000.bmp" in manifest.files

    def test_a_missing_sample_is_reported_with_its_name(self, tmp_path):
        _dataset(tmp_path, n=2)
        with pytest.raises(ManifestError, match="raw sample missing"):
            build_manifest(dataset="X", root=tmp_path,
                           splits={"test": ["img_500.png"]},
                           raw_subdir="raw")

    def test_coverage_holds_across_mismatched_extensions(self, tmp_path):
        # The coverage check compares stems, so a .jpg/.bmp pair validates.
        _dataset(tmp_path, n=2, gt="masks", raw_ext=".jpg", gt_ext=".bmp")
        manifest = build_manifest(dataset="X", root=tmp_path,
                                  splits={"test": ["img_000.jpg",
                                                   "img_001.jpg"]},
                                  raw_subdir="raw", gt_subdir="masks")
        manifest.validate(require_splits=False)  # must not raise
