"""Offline import (Mode B) and the full evaluation pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hydro_bench.evaluation import BenchmarkRunner, RunConfig
from hydro_bench.methods.classical.retinex.adapter import Retinex
from hydro_bench.methods.offline_importer import (OfflineImportError,
                                                 OfflineImporter)
from hydro_bench.run_benchmark import cmd_run


class TestOfflineImporter:
    def test_scan_and_load(self, synth_dataset, tmp_path):
        src = synth_dataset / "raw_input_imgs"
        out = tmp_path / "MethodB"
        out.mkdir()
        for path in sorted(src.iterdir())[:4]:
            shutil.copy(path, out / path.name)
        importer = OfflineImporter(output_dir=out, method="MethodB")
        ids = [p.name for p in sorted(out.iterdir())]
        importer.verify_coverage(ids)
        image = importer.load(ids[0])
        assert image.dtype == np.uint8
        assert image.shape[2] == 3

    def test_missing_output_raises(self, tmp_path):
        out = tmp_path / "empty"
        out.mkdir()
        importer = OfflineImporter(output_dir=out, method="M")
        with pytest.raises(OfflineImportError, match="missing 1/1 outputs"):
            importer.verify_coverage(["img_0000.png"])

    def test_nonexistent_directory_raises(self, tmp_path):
        importer = OfflineImporter(output_dir=tmp_path / "nope", method="M")
        with pytest.raises(OfflineImportError, match="does not exist"):
            importer.scan()

    def test_suffix_stripping(self, tmp_path):
        out = tmp_path / "M"
        out.mkdir()
        from PIL import Image

        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(out / "a_enhanced.png")
        importer = OfflineImporter(output_dir=out, method="M")
        assert "a" in importer.available_stems

    def test_ambiguous_names_rejected(self, tmp_path):
        out = tmp_path / "M"
        out.mkdir()
        from PIL import Image

        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(out / "a.png")
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(out / "a_enhanced.png")
        importer = OfflineImporter(output_dir=out, method="M")
        with pytest.raises(OfflineImportError, match="ambiguous"):
            importer.scan()

    def test_geometry_mismatch_caught(self, synth_dataset, tmp_path):
        from PIL import Image

        src = synth_dataset / "raw_input_imgs"
        out = tmp_path / "M"
        out.mkdir()
        sample = sorted(src.iterdir())[0]
        shutil.copy(sample, out / sample.name)
        importer = OfflineImporter(output_dir=out, method="M")
        from hydro_bench.core.dataset_manager import Sample

        tiny = out / "tiny.png"
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tiny)
        sample_obj = Sample(id=sample.name, raw_path=Path(tiny))
        with pytest.raises(OfflineImportError, match="geometry"):
            importer.verify_geometry([sample_obj])

    def test_discovery(self, tmp_path):
        from hydro_bench.methods import offline_importer

        base = tmp_path / "results" / "SynthUIB"
        (base / "MethodA").mkdir(parents=True)
        (base / "MethodB").mkdir(parents=True)
        found = offline_importer.discover_offline_methods(tmp_path / "results",
                                                         "SynthUIB")
        assert {i.method for i in found} == {"MethodA", "MethodB"}

    def test_discovery_missing_dataset(self, tmp_path):
        from hydro_bench.methods import offline_importer

        assert offline_importer.discover_offline_methods(tmp_path, "Nope") == []


class TestEvaluationPipeline:
    def test_run_with_online_methods(self, synth_root, tmp_path):
        config = RunConfig(dataset="SynthUIB", split="test",
                           methods=["UDCP", "Retinex"],
                           fr_metrics=["psnr", "ssim"],
                           nr_metrics=["uciqe"],
                           out_dir=tmp_path / "results",
                           show_progress=False)
        runner = BenchmarkRunner(config=config)
        results = runner.run()
        assert set(results.methods()) == {"UDCP", "Retinex"}
        assert "psnr" in results.scores
        assert "uciqe" in results.scores
        # Every method must have scored every sample (no silent drops).
        for metric in ("psnr", "uciqe"):
            for method in results.methods():
                assert len(results.scores[metric][method]) == 6

    def test_results_json_written(self, synth_root, tmp_path):
        out = tmp_path / "results"
        config = RunConfig(dataset="SynthUIB", split="test", methods=["UDCP"],
                           fr_metrics=["psnr"], nr_metrics=[],
                           out_dir=out, show_progress=False)
        BenchmarkRunner(config=config).run()
        payload = json.loads((out / "SynthUIB__test" / "results.json").read_text())
        assert payload["dataset"] == "SynthUIB"
        assert payload["metadata"]["n_samples"] == 6
        assert payload["metadata"]["methods_evaluated"] == ["UDCP"]

    def test_offline_and_online_share_metrics(self, synth_root, tmp_path):
        # A Mode B directory must produce the same metric keys as Mode A.
        offline_dir = tmp_path / "precomputed"
        offline_dir.mkdir()
        algorithm = Retinex()
        manager_src = synth_root / "SynthUIB" / "raw_input_imgs"
        for path in sorted(manager_src.iterdir()):
            if path.name.startswith("test_"):
                from PIL import Image

                image = np.asarray(Image.open(path).convert("RGB"))
                Image.fromarray(algorithm.enhance(image)).save(
                    offline_dir / path.name)
        config = RunConfig(dataset="SynthUIB", split="test",
                           methods=["OfflineRetinex", "UDCP"],
                           fr_metrics=["psnr", "ssim"],
                           nr_metrics=["uciqe"],
                           offline_dirs={"OfflineRetinex": offline_dir},
                           out_dir=tmp_path / "results",
                           show_progress=False)
        results = BenchmarkRunner(config=config).run()
        assert set(results.methods()) == {"OfflineRetinex", "UDCP"}
        assert set(results.scores) == {"psnr", "ssim", "uciqe"}

    def test_limit_truncates(self, synth_root, tmp_path):
        config = RunConfig(dataset="SynthUIB", split="test", methods=["UDCP"],
                           fr_metrics=["psnr"], nr_metrics=[],
                           limit=3, out_dir=tmp_path / "results",
                           show_progress=False)
        results = BenchmarkRunner(config=config).run()
        assert len(results.scores["psnr"]["UDCP"]) == 3

    def test_unknown_method_is_reported_not_fatal(self, synth_root, tmp_path):
        config = RunConfig(dataset="SynthUIB", split="test",
                           methods=["UDCP", "NoSuchMethod"],
                           fr_metrics=["psnr"], nr_metrics=[],
                           out_dir=tmp_path / "results", show_progress=False)
        runner = BenchmarkRunner(config=config)
        results = runner.run()
        assert "NoSuchMethod" not in results.methods()
        assert any("NoSuchMethod" in w for w in runner.warnings)

    def test_no_methods_raises(self, synth_root, tmp_path):
        config = RunConfig(dataset="SynthUIB", split="test", methods=[],
                           fr_metrics=["psnr"], nr_metrics=[],
                           out_dir=tmp_path / "results", show_progress=False)
        with pytest.raises(RuntimeError, match="no methods"):
            BenchmarkRunner(config=config).run()

    def test_empty_split_raises(self, synth_root, tmp_path):
        config = RunConfig(dataset="SynthUIB", split="val",
                           methods=["UDCP"], fr_metrics=["psnr"],
                           nr_metrics=[], limit=0,
                           out_dir=tmp_path / "results", show_progress=False)
        with pytest.raises(RuntimeError, match="empty"):
            BenchmarkRunner(config=config).run()

    def test_gpu_cache_released_between_frames(self, synth_root, tmp_path):
        # The MPS caching allocator holds freed buffers rather than returning
        # them to the OS, so over hundreds of multi-megapixel frames resident
        # memory climbs until a Metal command buffer deadlocks. The per-frame
        # release is what keeps long runs alive on a memory-tight host.
        config = RunConfig(dataset="SynthUIB", split="test", methods=["UDCP"],
                           fr_metrics=["psnr"], nr_metrics=[],
                           limit=3, out_dir=tmp_path / "results",
                           show_progress=False)
        runner = BenchmarkRunner(config=config)
        calls = []
        runner._release_gpu_cache = lambda: calls.append(1)
        runner.run()
        assert len(calls) == 3  # once per frame, not once per (frame, method)


class TestCLIRun:
    def test_cli_run_command(self, synth_root, tmp_path, monkeypatch):
        monkeypatch.setenv("HYDROBENCH_DATA", str(synth_root))
        args = _namespace(dataset="SynthUIB", split="test",
                          methods=["UDCP", "Retinex"],
                          metrics=["psnr", "uciqe"],
                          out_dir=str(tmp_path / "cli_results"),
                          baseline="UDCP")
        assert cmd_run(args) == 0
        out = tmp_path / "cli_results" / "SynthUIB__test"
        assert (out / "results.json").is_file()
        assert (out / "results_table.tex").is_file()
        assert (out / "results_table.md").is_file()

    def test_cli_report_command(self, synth_root, tmp_path, monkeypatch):
        monkeypatch.setenv("HYDROBENCH_DATA", str(synth_root))
        args = _namespace(dataset="SynthUIB", split="test",
                          methods=["UDCP"], metrics=["psnr"],
                          out_dir=str(tmp_path / "cli_results"))
        assert cmd_run(args) == 0
        results_json = tmp_path / "cli_results" / "SynthUIB__test" / "results.json"
        report_args = _namespace(results_json=str(results_json),
                                 out_dir=str(tmp_path / "report_only"),
                                 baseline="UDCP", dataset="SynthUIB",
                                 split="test", methods=["UDCP"])
        from hydro_bench.run_benchmark import cmd_report

        assert cmd_report(report_args) == 0
        assert (tmp_path / "report_only" / "SynthUIB__test" /
                "results_table.tex").is_file()


def _namespace(**kwargs):
    import argparse

    defaults = dict(dataset="SynthUIB", split="test", methods=None,
                    metrics=None, keypoints=None, detection=False,
                    segmentation=False, limit=None, baseline=None,
                    out_dir="hydro_bench_results", offline_dir=[], params=[],
                    skip_hashes=False, quiet=True, decimals=2, seed=0,
                    resume=False, cap_mps_memory=True, results_json=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)
