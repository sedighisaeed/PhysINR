"""Incremental checkpointing and --resume for the in-process benchmark runner.

A full UIEB run takes hours on this host, and one oversized image can cost
thirty minutes on its own. The runner therefore checkpoints every image and
can continue from that checkpoint, so an interrupt costs at most the image in
flight rather than the whole run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydro_bench.evaluation import BenchmarkRunner, RunConfig


def _config(tmp_path: Path, **overwrites) -> RunConfig:
    return RunConfig(
        dataset="DS",
        split="test",
        methods=[],
        out_dir=tmp_path / "out",
        show_progress=False,
        verify_checksums=False,
        **overwrites,
    )


class TestPartialCheckpoint:
    def test_partial_path_lives_under_the_run_directory(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        assert runner._partial_path() == tmp_path / "out" / "DS__test" / "partial.json"

    def test_load_partial_returns_empty_when_no_checkpoint_exists(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        assert runner._load_partial() == {}

    def test_persist_partial_is_atomic_and_round_trips(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        runner._partial = {
            "img_1.png": {"UDCP": {"psnr": 12.5, "ssim": 0.8}},
            "img_2.png": {"UDCP": {"psnr": 11.2}, "Retinex": {"psnr": 10.1}},
        }
        runner._persist_partial()

        path = runner._partial_path()
        assert path.is_file()
        assert not path.with_suffix(".json.tmp").exists()
        assert runner._load_partial() == runner._partial

    def test_persist_partial_creates_the_output_directory(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        runner._partial = {"a": {}}
        runner._persist_partial()
        assert runner._partial_path().is_file()

    def test_corrupt_checkpoint_raises_with_actionable_message(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        runner._partial_path().parent.mkdir(parents=True, exist_ok=True)
        runner._partial_path().write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="resume checkpoint"):
            runner._load_partial()


class TestResumeSemantics:
    def test_resume_flag_defaults_to_off(self):
        config = RunConfig(dataset="DS")
        assert config.resume is False

    def test_resume_flag_is_settable(self, tmp_path):
        config = _config(tmp_path, resume=True)
        assert config.resume is True

    def test_score_pair_records_per_method_under_the_sample(self, tmp_path):
        import numpy as np

        runner = BenchmarkRunner(_config(tmp_path, nr_metrics=["uciqe"]))
        runner.results = type(runner.results)(dataset="DS", split="test")

        class FakeSample:
            id = "img_1.png"
            gt_path = None

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        runner._score_pair(FakeSample(), image, image, "UDCP")
        entry = runner._partial["img_1.png"]["UDCP"]
        assert "uciqe" in entry and isinstance(entry["uciqe"], float)

    def test_score_pair_with_no_scores_leaves_no_entry(self, tmp_path):
        runner = BenchmarkRunner(_config(tmp_path))
        runner.results = type(runner.results)(dataset="DS", split="test")

        class FakeSample:
            id = "img_1.png"
            gt_path = None

        # No FR and no NR metrics requested: nothing is scored, and the
        # checkpoint must not record an empty entry that --resume would
        # mistake for a finished image.
        runner._score_pair(FakeSample(), None, None, "UDCP")
        assert runner._partial == {}

    def test_resume_keeps_the_replayed_scores_in_the_checkpoint(self, tmp_path):
        """A replayed checkpoint must not be clobbered by the next persist.

        Regression: run() replayed the on-disk checkpoint into the results but
        left _partial empty, so the first newly-scored pair overwrote
        partial.json and silently discarded every replayed image. The next
        interrupt would then redo hours of work.
        """
        runner = BenchmarkRunner(_config(tmp_path))
        runner._partial = {
            "img_1.png": {"UDCP": {"psnr": 12.5}},
            "img_2.png": {"UDCP": {"psnr": 11.2}},
        }
        runner._persist_partial()

        resumed = BenchmarkRunner(_config(tmp_path, resume=True))
        done = resumed._load_partial()
        # run() seeds the in-memory checkpoint from the one it just read.
        resumed._partial.update(done)
        # A later pair arrives; persisting must keep the replayed entries.
        resumed._partial.setdefault("img_3.png", {})["Retinex"] = {"psnr": 9.9}
        resumed._persist_partial()

        on_disk = json.loads(resumed._partial_path().read_text(encoding="utf-8"))
        assert set(on_disk) == {"img_1.png", "img_2.png", "img_3.png"}
        assert on_disk["img_1.png"]["UDCP"]["psnr"] == 12.5
        assert on_disk["img_3.png"]["Retinex"]["psnr"] == 9.9
