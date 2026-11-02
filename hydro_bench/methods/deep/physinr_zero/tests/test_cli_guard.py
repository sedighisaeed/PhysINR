"""Tests for the meta-training source guard.

PhysINR-Zero's central claim is that no real underwater image enters
meta-training.  The most efficient way to falsify it is also the most
tempting shortcut: point ``--source`` at a benchmark dataset's reference
directory, which is the cleanest imagery in the repo.  These tests pin the
refusal so the guard cannot silently go stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathlib import Path

from hydro_bench.methods.deep.physinr_zero.cli import _assert_source_is_clean, _evaluation_datasets

REPO = Path(__file__).resolve().parents[5]


class TestEvaluationDatasets:
    def test_every_manifest_with_a_test_split_is_listed(self):
        out = _evaluation_datasets()
        # Derived from the manifests, so this grows as datasets are added.
        for expected in ("uieb", "lsui", "euvp", "ruie", "uccs"):
            assert expected in out, f"{expected} has a test split but is missing"

    def test_datasets_without_a_test_split_are_absent(self):
        out = _evaluation_datasets()
        manifest_dir = REPO / "hydro_bench" / "datasets" / "manifests"
        for path in manifest_dir.glob("*.json"):
            blob = json.loads(path.read_text())
            has_test = bool(blob.get("splits", {}).get("test"))
            assert (path.stem in out) == has_test


class TestSourceGuard:
    def _logger(self):
        class _L:
            def __init__(self):
                self.messages = []

            def error(self, msg):
                self.messages.append(msg)

            def warning(self, msg):
                self.messages.append(("warn", msg))
        return _L()

    def test_an_evaluation_dataset_root_is_refused(self):
        log = self._logger()
        src = REPO / "datasets" / "LSUI"
        assert not _assert_source_is_clean(src, ["lsui"], False, log)
        assert any("zero-shot" in m for m in log.messages)

    def test_an_evaluation_reference_dir_is_refused(self):
        log = self._logger()
        src = REPO / "datasets" / "LSUI" / "GT"
        assert not _assert_source_is_clean(src, ["lsui"], False, log)

    def test_a_case_mismatch_is_caught(self):
        """``datasets/LSUI`` vs the manifest key ``lsui``: the comparison is
        case-insensitive so a capitalisation slip does not let a dataset
        through."""
        log = self._logger()
        src = REPO / "datasets" / "LSUI"
        assert not _assert_source_is_clean(src, ["LSUI"], False, log)

    def test_a_relative_path_cannot_bypass_the_guard(self):
        """A relative source path must still be recognised.

        ``relative_to`` against an absolute base would raise ValueError for a
        relative path, returning None -- "not an evaluation dataset" -- and
        letting the very directory the guard exists to catch straight through.
        """
        log = self._logger()
        assert not _assert_source_is_clean(Path("datasets/LSUI/GT"), ["lsui"],
                                           False, log)
        assert not _assert_source_is_clean(Path("datasets/euvp"), ["euvp"],
                                           False, log)

    def test_every_evaluation_dataset_is_refused(self):
        """The guard must fire for the whole derived list, not the one that
        happened to be written into a test."""
        ds = _evaluation_datasets()
        log = self._logger()
        for name in ds:
            src = REPO / "datasets" / name.upper()
            assert not _assert_source_is_clean(src, ds, False, log), \
                f"{name} is in the evaluation list but passed the guard"

    def test_a_clean_external_source_is_allowed(self, tmp_path):
        log = self._logger()
        src = tmp_path / "clean_photos"
        src.mkdir()
        assert _assert_source_is_clean(src, ["lsui", "uieb"], False, log)
        assert log.messages == []

    def test_the_override_is_allowed_but_flagged(self, tmp_path):
        """``--allow-eval-source`` is an escape hatch, not a silence button:
        the run proceeds and the warning text says the claim must be dropped."""
        log = self._logger()
        src = REPO / "datasets" / "LSUI"
        assert _assert_source_is_clean(src, ["lsui"], True, log)
        assert any("OVERRIDE" in str(m) and "zero-shot" in str(m)
                   for m in log.messages)

    def test_a_path_outside_datasets_is_allowed(self, tmp_path):
        log = self._logger()
        assert _assert_source_is_clean(tmp_path / "elsewhere", ["lsui"],
                                       False, log)


class TestEnhanceLossReporting:
    """``enhance`` logs a per-image ``final_loss``.

    Adaptation always computes the loss — it is the objective it descends — so
    the logged value must be a real number even when ``--trace`` is off.  A NaN
    there is not a missing feature: it is the log line that would hide a
    diverging run behind a formatting default, which is exactly how the broken
    SSIM denominator went unnoticed for 1,400 tasks.
    """

    @pytest.fixture
    def adapter(self):
        import torch

        from hydro_bench.methods.deep.physinr_zero.adapt import TestTimeAdapter
        from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
        from hydro_bench.methods.deep.physinr_zero.model import PhysINR

        cfg = PhysINRConfig(device="cpu")
        model = PhysINR(cfg)
        return TestTimeAdapter(cfg, model)

    @pytest.fixture
    def logger(self, tmp_path):
        from hydro_bench.methods.deep.physinr_zero.utils import get_logger

        return get_logger("physinr.test_enhance", tmp_path)

    def test_the_loss_is_reported_without_a_trace(self, adapter, logger, caplog):
        """The regression: ``return_trace=False`` returned ``trace=None``, and
        the log line printed ``final_loss=nan`` for every image."""
        import numpy as np

        from hydro_bench.methods.deep.physinr_zero.cli import enhance_one

        rng = np.random.default_rng(0)
        image = (rng.random((64, 72, 3)) * 0.5 + 0.2).astype(np.float32)
        with caplog.at_level("INFO", logger="physinr.test_enhance"):
            _, trace, _ = enhance_one(adapter, image, "in.png", "out.png",
                                      steps=2, return_trace=False,
                                      logger=logger)
        line = [r.getMessage() for r in caplog.records
                if "final_loss" in r.getMessage()]
        assert line, "no per-image log line was emitted"
        assert "nan" not in line[0], line[0]
        # The trace is discarded when not requested, but the loss it reported
        # was the real one.
        assert trace is None

    def test_a_trace_still_records_the_terms(self, adapter, logger, caplog):
        import numpy as np

        from hydro_bench.methods.deep.physinr_zero.cli import enhance_one

        rng = np.random.default_rng(1)
        image = (rng.random((64, 72, 3)) * 0.5 + 0.2).astype(np.float32)
        with caplog.at_level("INFO", logger="physinr.test_enhance"):
            _, trace, _ = enhance_one(adapter, image, "in.png", "out.png",
                                      steps=2, return_trace=True,
                                      logger=logger)
        assert trace is not None
        assert len(trace.losses) == 2
        assert all(l == l for l in trace.losses)  # no NaN
