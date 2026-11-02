"""Execution runners: the environment-isolation layer.

Conda and Docker are unavailable on the test host, so these tests exercise the
runner *contract* directly -- manifest generation, output validation, failure
detection -- using a fake runner that stands in for the real subprocess. This
is exactly the seam that lets the harness trust a method it never imported.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from hydro_bench.core import base_runner
from hydro_bench.core.conda_runner import CondaRunner
from hydro_bench.core.docker_runner import DockerRunner


@pytest.fixture
def adapter_script(tmp_path):
    """A stand-in for a method's container/conda-side entry point."""
    path = tmp_path / "adapter.py"
    path.write_text("#!/usr/bin/env python\nprint('hello')\n")
    return path


class FakeRunner(base_runner.ExecutionRunner):
    """A runner that pretends to invoke a method, for testing the contract."""

    kind = "fake"

    def __init__(self, workdir, behavior="ok", timeout=None, dry_run=False):
        super().__init__(workdir, timeout, dry_run)
        self.behavior = behavior
        self.commands = []

    def _build_command(self, job):
        return [sys.executable, "-c", "pass", job.method]

    def run(self, job, show_command=False):
        # Mirror the real runner: prepare the sandbox, then either short-circuit
        # on a dry run or "execute" and validate what the child produced.
        cmd = self._build_command(job)
        self.commands.append(cmd)
        self.prepare(job)
        if self.dry_run:
            return base_runner.RunResult(
                returncode=0, command=cmd, workdir=self.workdir,
                output_manifest={"method": job.method, "samples": [],
                                 "dry_run": True})
        if self.behavior == "ok":
            return self._simulate_success(job)
        if self.behavior in ("fail_exit", "fail_exit_strict"):
            # run() returns the raw status; only raise_for_status() (or a
            # strict runner) turns a nonzero code into an exception.
            result = base_runner.RunResult(returncode=1, stdout="", stderr="boom",
                                           workdir=self.workdir, command=cmd)
            if self.behavior == "fail_exit_strict":
                result.raise_for_status()
            return result
        if self.behavior == "no_manifest":
            # Exit 0 but write nothing: the harness must still detect this.
            return self.finalize(job, base_runner.RunResult(
                returncode=0, workdir=self.workdir, command=cmd))
        if self.behavior == "missing_samples":
            # Report fewer samples than requested: partial trees must fail
            # loudly rather than silently biasing the paired statistics.
            (job.workdir / base_runner.OUTPUT_MANIFEST).write_text(json.dumps({
                "method": job.method, "samples": [],
            }))
            return self.finalize(job, base_runner.RunResult(
                returncode=0, workdir=self.workdir, command=cmd))
        raise ValueError(f"unknown behavior {self.behavior}")

    def _simulate_success(self, job):
        from PIL import Image

        for sid in job.sample_ids:
            image = np.full((8, 8, 3), 120, np.uint8)
            Image.fromarray(image).save(job.output_dir / f"{sid}.png")
        manifest = {
            "method": job.method,
            "samples": [
                {"id": sid,
                 "input": str(job.input_dir / f"{sid}.png"),
                 "output": str(job.output_dir / f"{sid}.png"),
                 "shape": [8, 8, 3]}
                for sid in job.sample_ids
            ],
        }
        (job.workdir / base_runner.OUTPUT_MANIFEST).write_text(
            json.dumps(manifest))
        result = base_runner.RunResult(returncode=0, workdir=self.workdir)
        return self.finalize(job, result)


def _job(workdir, method="FakeMethod", sample_ids=("a", "b")):
    workdir = Path(workdir)
    return base_runner.RunnerJob(
        method=method,
        adapter="hydro_bench.methods.fake.adapter",
        input_dir=workdir / "inputs",
        output_dir=workdir / "outputs",
        workdir=workdir,
        sample_ids=list(sample_ids),
    )


class TestRunnerContract:
    def test_manifest_round_trip(self, tmp_path):
        job = _job(tmp_path)
        path = job.write_input_manifest(gt_available=True)
        payload = json.loads(path.read_text())
        assert payload["method"] == "FakeMethod"
        assert len(payload["samples"]) == 2
        assert payload["gt_available"] is True
        # Ground truth is never referenced in the manifest body itself.
        assert "gt" not in json.dumps(payload["samples"])

    def test_successful_run_collects_outputs(self, tmp_path):
        runner = FakeRunner(tmp_path)
        job = _job(tmp_path)
        result = runner.run(job)
        assert result.ok
        outputs = base_runner.collect_outputs(job)
        assert set(outputs) == {"a", "b"}
        for image in outputs.values():
            assert image.shape == (8, 8, 3)

    def test_failing_exit_code_is_reported(self, tmp_path):
        runner = FakeRunner(tmp_path, behavior="fail_exit")
        result = runner.run(_job(tmp_path))
        assert not result.ok
        assert result.returncode == 1
        with pytest.raises(base_runner.RunnerError, match="exited with code 1"):
            result.raise_for_status()

    def test_strict_runner_raises_on_failure(self, tmp_path):
        runner = FakeRunner(tmp_path, behavior="fail_exit_strict")
        with pytest.raises(base_runner.RunnerError, match="exited with code 1"):
            runner.run(_job(tmp_path))

    def test_missing_output_manifest_raises(self, tmp_path):
        runner = FakeRunner(tmp_path, behavior="no_manifest")
        result = runner.run(_job(tmp_path))
        with pytest.raises(base_runner.RunnerError, match="did not produce"):
            result.raise_for_status()

    def test_incomplete_sample_list_raises(self, tmp_path):
        # A child that reports fewer samples than requested must fail loudly:
        # silently dropping samples would bias the paired statistics.
        runner = FakeRunner(tmp_path, behavior="missing_samples")
        with pytest.raises(base_runner.RunnerError, match="did not report"):
            runner.run(_job(tmp_path))

    def test_dry_run_executes_nothing(self, tmp_path):
        runner = FakeRunner(tmp_path, dry_run=True)
        result = runner.run(_job(tmp_path))
        assert result.returncode == 0
        assert result.output_manifest["dry_run"] is True
        # A dry run must not have written any outputs.
        job = _job(tmp_path)
        assert not (job.output_dir / "a.png").exists()

    def test_materialise_and_collect_round_trip(self, tmp_path):
        job = _job(tmp_path)
        job.input_dir.mkdir(parents=True, exist_ok=True)
        images = {"a": np.full((8, 8, 3), 10, np.uint8),
                  "b": np.full((8, 8, 3), 200, np.uint8)}
        base_runner.materialise_inputs(job, images)
        # The child process turns inputs into outputs (identity here).
        job.output_dir.mkdir(parents=True, exist_ok=True)
        for sid in images:
            shutil.copy(job.input_dir / f"{sid}.png",
                        job.output_dir / f"{sid}.png")
        collected = base_runner.collect_outputs(job)
        assert collected["a"].flat[0] == 10
        assert collected["b"].flat[0] == 200

    def test_materialise_rejects_non_uint8(self, tmp_path):
        job = _job(tmp_path)
        with pytest.raises(ValueError, match="uint8"):
            base_runner.materialise_inputs(job, {"a": np.zeros((8, 8, 3), np.float32)})

    def test_collect_outputs_detects_missing(self, tmp_path):
        job = _job(tmp_path)
        job.output_dir.mkdir(parents=True)
        with pytest.raises(base_runner.RunnerError, match=r"0/2"):
            base_runner.collect_outputs(job)

    def test_fresh_workdir_is_deterministic(self, tmp_path):
        first = base_runner.fresh_workdir(tmp_path, "M", "test")
        (first / "marker.txt").write_text("x")
        second = base_runner.fresh_workdir(tmp_path, "M", "test")
        assert first == second
        assert not (second / "marker.txt").exists()


class TestCondaRunner:
    def test_requires_env_name(self, tmp_path):
        with pytest.raises(base_runner.RunnerError, match="requires env_name"):
            CondaRunner(workdir=tmp_path, env_name="")

    def test_command_shape(self, tmp_path, monkeypatch, adapter_script):
        # Conda is absent on the test host; the command is still fully
        # constructible as long as *some* conda is resolvable.
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        runner = CondaRunner(workdir=tmp_path, env_name="myenv",
                             adapter_script=adapter_script)
        job = _job(tmp_path)
        cmd = runner._build_command(job)
        assert cmd[0] == "/usr/bin/conda"
        assert "run" in cmd and "--no-capture-output" in cmd
        assert "-n" in cmd and "myenv" in cmd
        assert "--input-manifest" in cmd

    def test_env_override(self, tmp_path, monkeypatch):
        runner = CondaRunner(workdir=tmp_path, env_name="myenv")
        monkeypatch.setenv("HYDROBENCH_CONDA_ENV_OVERRIDE", "other")
        assert runner._resolved_env() == "other"

    def test_adapter_resolution_raises_without_script(self, tmp_path):
        runner = CondaRunner(workdir=tmp_path, env_name="myenv")
        with pytest.raises(base_runner.RunnerError, match="no adapter script"):
            runner._adapter_path(_job(tmp_path))

    def test_env_override_path_injection(self, tmp_path, monkeypatch,
                                         adapter_script):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        runner = CondaRunner(workdir=tmp_path, env_name="myenv",
                             adapter_script=adapter_script,
                             extra_path=["/custom/bin"])
        original = dict(os.environ)
        # dry_run builds the command (and exercises the PATH injection) but
        # never launches a subprocess.
        runner.dry_run = True
        runner.run(_job(tmp_path))
        # The parent environment must be restored afterwards.
        assert os.environ.get("PATH") == original.get("PATH")


class TestDockerRunner:
    def test_requires_image(self, tmp_path):
        with pytest.raises(base_runner.RunnerError, match="requires image"):
            DockerRunner(workdir=tmp_path, image="")

    def test_command_shape(self, tmp_path, monkeypatch, adapter_script):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        runner = DockerRunner(workdir=tmp_path, image="my:latest",
                              adapter_script=adapter_script)
        cmd = runner._build_command(_job(tmp_path))
        assert cmd[0] == "/usr/bin/docker"
        assert "run" in cmd
        assert "--gpus" in cmd
        assert "my:latest" in cmd
        # Inputs mounted read-only, outputs read-write.
        joined = " ".join(cmd)
        assert ":ro" in joined

    def test_gt_dir_never_mounted(self, tmp_path, monkeypatch, adapter_script):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        runner = DockerRunner(workdir=tmp_path, image="my:latest",
                              adapter_script=adapter_script)
        job = _job(tmp_path)
        job.gt_dir = tmp_path / "secret_gt"
        cmd = runner._build_command(job)
        joined = " ".join(cmd)
        assert "secret_gt" not in joined

    def test_network_isolated_by_default(self, tmp_path, monkeypatch,
                                         adapter_script):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        runner = DockerRunner(workdir=tmp_path, image="my:latest",
                              adapter_script=adapter_script)
        cmd = runner._build_command(_job(tmp_path))
        assert "--network" in cmd
        assert "none" in cmd

    def test_sanity_check_without_docker(self, tmp_path, monkeypatch):
        runner = DockerRunner(workdir=tmp_path, image="my:latest")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        report = runner.env_sanity_check()
        assert report["available"] is False
