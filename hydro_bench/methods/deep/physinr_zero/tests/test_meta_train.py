"""Tests for the meta-training provenance record.

The checkpoint's provenance file is the only place the paper's convergence
figure and the "meta-trained on N tasks" claim can be checked, so the record
must actually carry the history and the counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydro_bench.methods.deep.physinr_zero.config import PhysINRConfig
from hydro_bench.methods.deep.physinr_zero.meta_train import MetaRecord, MetaTrainer


def _tiny_cfg(**kw) -> PhysINRConfig:
    base = dict(meta_tasks=2, meta_batch=1, meta_inner_steps=1,
                meta_crop=32, adapt_steps=1, device="cpu", seed=0)
    base.update(kw)
    return PhysINRConfig(**base)


@pytest.fixture
def sources(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        arr = (rng.random((48, 64, 3)) * 0.5 + 0.25)
        img = Image.fromarray(np.clip(arr * 255, 0, 255).round().astype(np.uint8))
        img.save(tmp_path / f"clean_{i}.png")
    return sorted(tmp_path.glob("*.png"))


class TestHistory:
    def test_a_run_records_per_task_loss(self, sources, tmp_path):
        trainer = MetaTrainer(_tiny_cfg(), sources)
        out = tmp_path / "meta.pt"
        record = trainer.train(tasks=2, out_path=out)

        assert record.tasks_completed == 2
        assert len(record.history) == 2
        assert all(set(h) == {"task", "meta_loss"} for h in record.history)
        # A real outer loss is finite; nan would mean the meta-step is broken.
        assert all(np.isfinite(h["meta_loss"]) for h in record.history)

    def test_history_reaches_the_provenance_file(self, sources, tmp_path):
        out = tmp_path / "meta.pt"
        MetaTrainer(_tiny_cfg(), sources).train(tasks=2, out_path=out)

        blob = json.loads(
            (out.with_suffix(".provenance.json")).read_text(encoding="utf-8"))
        assert blob["tasks_completed"] == 2
        assert len(blob["history"]) == 2
        assert blob["history"][0]["task"] == 1
        assert blob["history"][-1]["task"] == 2
        # The figure script reads exactly this key.
        assert all("meta_loss" in h for h in blob["history"])

    def test_a_checkpoint_from_this_run_is_not_a_placeholder(self, sources,
                                                              tmp_path):
        """The adapter refuses tasks == 0; a real run must not look like one."""
        import torch

        out = tmp_path / "meta.pt"
        MetaTrainer(_tiny_cfg(), sources).train(tasks=2, out_path=out)
        blob = torch.load(out, map_location="cpu", weights_only=False)
        assert blob["tasks"] == 2
        assert np.isfinite(blob["meta_loss"])


class TestPreloadSizing:
    """The load size must not silently discard most of the pool.

    A source is only usable if it can yield a ``meta_crop`` patch, so the load
    size has to be derived from the crop. DIV2K frames are ~2040px on the long
    side; loading at 256px scaled a 2040x1356 frame down to 256x170, whose 170px
    short side cannot produce a 192px crop, and 780 of 900 images were dropped
    without a word.
    """

    def test_the_default_load_side_tracks_the_crop(self, sources):
        """The load size is derived from the crop, not hardcoded.

        A source is only eligible if its short side can yield a
        ``meta_crop`` patch after downscaling, so the default must stay coupled
        to the crop as the crop changes.
        """
        import inspect

        from hydro_bench.methods.deep.physinr_zero.meta_train import MetaTrainer

        # The derivation lives in the default expression itself.
        src = inspect.getsource(MetaTrainer.preload)
        assert "2 * self.cfg.meta_crop" in src, \
            "preload's default max_side must be derived from meta_crop"

        # And it actually produces an eligible pool for both orientations.
        cfg = _tiny_cfg(meta_crop=32)
        trainer = MetaTrainer(cfg, sources)
        trainer.preload()
        # meta_crop=32 -> max_side=64, and every fixture frame is 48x64 or 64x48,
        # so all sources should be eligible.
        assert len(trainer._cache) == len(trainer.image_paths), (
            f"preload kept {len(trainer._cache)} of "
            f"{len(trainer.image_paths)}")


class TestEmptyRecord:
    def test_no_history_serialises_cleanly(self):
        record = MetaRecord(config={}, param_count=0, tasks_completed=0,
                            duration_seconds=0.0, seed=0, source_images=0)
        blob = record.to_dict()
        assert blob["history"] == []
        # A zero-task record is exactly what the adapter must refuse to build.
        assert blob["tasks_completed"] == 0


class TestTypeSupervision:
    """The collapse fix: the outer loop must hold the type head to the drawn
    task type.  Without it the head's optimum is the marginal type
    distribution -- a constant, which is exactly what the diagnostics
    measured in both released checkpoints."""

    def test_a_supervised_run_records_the_type_accuracy(self, sources, tmp_path):
        trainer = MetaTrainer(_tiny_cfg(w_type_sup=0.5, w_tau_sup=0.5), sources)
        record = trainer.train(tasks=2)
        assert np.isfinite(record.type_accuracy)
        assert 0.0 <= record.type_accuracy <= 1.0
        assert np.isfinite(record.tau_error)
        assert record.tau_error >= 0.0

    def test_an_unsupervised_run_reports_no_type_accuracy(self, sources):
        trainer = MetaTrainer(_tiny_cfg(w_type_sup=0.0, w_tau_sup=0.0), sources)
        record = trainer.train(tasks=2)
        assert np.isnan(record.type_accuracy)
        assert np.isnan(record.tau_error)

    def test_the_task_mixture_reaches_the_target(self, sources):
        """The supervision target is the soft mixture the degradation was
        built from, not a hard one-hot -- a sampled task mixes types, so a
        one-hot target would punish a physically correct prediction."""
        from hydro_bench.methods.deep.physinr_zero.synthesize import SyntheticDegradation

        synth = SyntheticDegradation(_tiny_cfg())
        tp = synth.sample_task(np.random.default_rng(0), (32, 32),
                               np.array([0.5, 0.5, 0.5]))
        assert tp.type_mix is not None
        assert len(tp.type_mix) == synth.cfg.num_water_types
        assert abs(sum(tp.type_mix) - 1.0) < 1e-6
        assert tp.type_mix[tp.water_type_idx] == max(tp.type_mix)
