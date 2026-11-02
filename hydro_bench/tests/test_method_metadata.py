"""Per-method provenance: what each row in the results table actually was.

A results file that reports scores without saying which methods ran the
published model and which ran a labelled fallback is not honest reporting --
the fallback is deliberately still scored, so the two are indistinguishable
from the numbers alone. The runner therefore collects each adapter's
``run_metadata`` into ``metadata.method_metadata``, and the rendered table
marks a fallback row rather than presenting it as the method's own result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from hydro_bench.core.base_algorithm import BaseAlgorithm
from hydro_bench.evaluation import BenchmarkRunner, RunConfig
from hydro_bench.reporter.latex_tables import (BenchmarkResults,
                                              render_markdown)


@dataclass
class _FakeFallback(BaseAlgorithm):
    """An adapter that admits it did not run the published model."""

    name = "FakeFallback"
    used_fallback: bool = field(default=False, init=False)

    def __post_init__(self):
        self.used_fallback = True

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image

    def run_metadata(self) -> Dict[str, Any]:
        return {"method": self.name, "used_fallback": True,
                "note": "checkpoint unavailable; ran the labelled fallback",
                "checkpoint": None}


@dataclass
class _FakeReal(BaseAlgorithm):
    """An adapter that ran its published weights."""

    name = "FakeReal"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image


def _config(tmp_path: Path, **overwrites) -> RunConfig:
    defaults = dict(dataset="DS", split="test", methods=[],
                    out_dir=tmp_path / "out", show_progress=False,
                    verify_checksums=False)
    defaults.update(overwrites)
    return RunConfig(**defaults)


class TestMethodMetadataCollection:
    def test_a_fallback_adapter_reports_its_provenance(self):
        meta = BenchmarkRunner._method_metadata("FakeFallback",
                                                _FakeFallback())
        assert meta["used_fallback"] is True
        assert "fallback" in meta["note"]

    def test_an_adapter_without_run_metadata_still_gets_an_entry(self):
        class Bare(BaseAlgorithm):
            name = "Bare"

            def enhance(self, image):
                return image

        meta = BenchmarkRunner._method_metadata("Bare", Bare())
        assert meta["method"] == "Bare"
        assert meta["used_fallback"] is False

    def test_an_adapter_whose_metadata_raises_does_not_sink_the_run(self):
        class Broken(BaseAlgorithm):
            name = "Broken"

            def enhance(self, image):
                return image

            def run_metadata(self):
                raise RuntimeError("provenance explosion")

        meta = BenchmarkRunner._method_metadata("Broken", Broken())
        assert "run_metadata unavailable" in meta["note"]

    def test_the_method_name_comes_from_the_adapter(self):
        """An adapter reports its own name, not the registry key it was built under.

        The two agree in practice (the registry is keyed by ``name``), but the
        adapter is the authority -- the runner's key is just how it was looked up.
        """
        meta = BenchmarkRunner._method_metadata("SomeRegistryKey",
                                                _FakeReal())
        assert meta["method"] == _FakeReal.name


class TestRunnerRecordsMetadata:
    def test_results_json_carries_method_metadata(self, tmp_path, synth_root):
        """The saved run must state, per method, whether it was the real one.

        Runs the actual runner over the synthetic dataset with one adapter that
        admits it fell back and one that does not, then reads back the file.
        """
        from hydro_bench.core.base_algorithm import registered_algorithms

        registry = registered_algorithms()
        registry[_FakeFallback.name] = _FakeFallback
        registry[_FakeReal.name] = _FakeReal
        try:
            config = _config(tmp_path, dataset="SynthUIB", split="test",
                             methods=[_FakeFallback.name, _FakeReal.name],
                             nr_metrics=["uciqe"], limit=2)
            runner = BenchmarkRunner(config=config)
            runner.run()
        finally:
            for name in (_FakeFallback.name, _FakeReal.name):
                registry.pop(name, None)

        payload = json.loads(
            (tmp_path / "out" / "SynthUIB__test" / "results.json").read_text())
        meta = payload["metadata"]["method_metadata"]
        assert meta[_FakeFallback.name]["used_fallback"] is True
        assert meta[_FakeReal.name]["used_fallback"] is False
        # The provenance is per method, and both rows are accounted for.
        assert set(meta) >= {_FakeFallback.name, _FakeReal.name}


class TestTrainingSetWarning:
    """A method scored on a set it was trained on is not held-out.

    Evaluating on the training set is a legitimate thing to do -- UIEB is what
    most of these networks were built on -- but a table that mixes trained-on
    and held-out rows without saying so invites the wrong comparison. The run
    records the fact as a warning rather than refusing, because the user is
    entitled to the numbers.
    """

    def _run_with(self, training_sets, dataset, synth_root) -> list:
        from hydro_bench.core.base_algorithm import (AlgorithmCapabilities,
                                                     registered_algorithms)

        @dataclass
        class Probe(BaseAlgorithm):
            name = "Probe"
            capabilities = AlgorithmCapabilities(training_sets=training_sets)

            def enhance(self, image: np.ndarray) -> np.ndarray:
                return image

        registry = registered_algorithms()
        registry[Probe.name] = Probe
        try:
            config = _config(synth_root.parent / "out_tmp", dataset=dataset,
                             split="test", methods=[Probe.name],
                             nr_metrics=["uciqe"], limit=1)
            runner = BenchmarkRunner(config=config)
            runner.run()
        finally:
            registry.pop(Probe.name, None)
        return runner.warnings

    def test_a_trained_on_dataset_produces_a_warning(self, synth_root):
        warnings = self._run_with(("SynthUIB",), "SynthUIB", synth_root)
        assert any("was trained on 'SynthUIB'" in w for w in warnings)

    def test_a_held_out_dataset_produces_no_such_warning(self, synth_root):
        warnings = self._run_with(("EUVP",), "SynthUIB", synth_root)
        assert not any("was trained on" in w for w in warnings)


class TestReporterMarksFallbacks:
    def test_a_fallback_row_is_labelled_in_the_markdown(self):
        results = BenchmarkResults(
            dataset="DS", split="test",
            scores={"psnr": {"Real": [20.0], "Ghost": [10.0]}},
            metadata={"Ghost": {"used_fallback": True,
                                "note": "checkpoint unavailable"}})
        md = render_markdown(results)
        assert "Ghost [fallback]" in md
        assert "Real" in md and "Real [fallback]" not in md

    def test_the_note_is_reproduced_under_the_table(self):
        results = BenchmarkResults(
            dataset="DS", split="test",
            scores={"psnr": {"Ghost": [10.0]}},
            metadata={"Ghost": {"used_fallback": True,
                                "note": "checkpoint unavailable"}})
        md = render_markdown(results)
        assert "checkpoint unavailable" in md

    def test_a_table_without_fallbacks_has_no_warning_block(self):
        results = BenchmarkResults(
            dataset="DS", split="test",
            scores={"psnr": {"Real": [20.0]}},
            metadata={"Real": {"used_fallback": False}})
        md = render_markdown(results)
        assert "[fallback]" not in md
