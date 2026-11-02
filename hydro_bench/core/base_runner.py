"""Execution runner abstraction.

Different restoration methods depend on mutually exclusive software stacks
(Python 3.7 + PyTorch 1.7 vs. Python 3.10 + PyTorch 2.4, conflicting CUDA
versions, etc.). HydroBench never imports those stacks into its own process.
Instead, each method is launched behind an :class:`ExecutionRunner`, and the
only thing that crosses the process boundary is a JSON manifest plus two
directories (input / output).

The on-disk contract (identical for every runner type):

.. code-block:: text

    <workdir>/
      input_manifest.json     # written by the harness
      inputs/<sample_id>.png  # raw images to restore
      outputs/<sample_id>.png # produced by the child process
      output_manifest.json    # written by the child, validated by the harness

A child that fails to produce ``output_manifest.json`` -- or that returns an
image whose size differs from its input -- causes the run to fail loudly
rather than silently dropping samples.
"""

from __future__ import annotations

import abc
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

INPUT_MANIFEST = "input_manifest.json"
OUTPUT_MANIFEST = "output_manifest.json"

MANIFEST_SCHEMA_VERSION = 1


@dataclass
class RunResult:
    """Outcome of a single runner invocation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    output_manifest: Optional[Dict[str, Any]] = None
    workdir: Optional[Path] = None
    command: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_manifest is not None

    def raise_for_status(self) -> "RunResult":
        if self.returncode != 0:
            raise RunnerError(
                f"runner exited with code {self.returncode}\n"
                f"command: {' '.join(self.command)}\n"
                f"stdout:\n{self.stdout[-2000:]}\n"
                f"stderr:\n{self.stderr[-4000:]}"
            )
        if self.output_manifest is None:
            raise RunnerError(
                f"child process did not produce {OUTPUT_MANIFEST} in {self.workdir}"
            )
        return self


class RunnerError(RuntimeError):
    """Raised when an isolated execution fails."""


@dataclass
class RunnerJob:
    """One method applied to one list of samples."""

    method: str
    adapter: str                     # import path or container-side script path
    input_dir: Path
    output_dir: Path
    workdir: Path
    sample_ids: List[str]
    gt_dir: Optional[Path] = None    # never copied into the sandbox (leakage guard)
    params: Dict[str, Any] = field(default_factory=dict)

    def write_input_manifest(self, gt_available: bool = False) -> Path:
        """Serialise the job description the child process consumes.

        ``gt_available`` controls whether the child is *told* ground truth
        exists. Ground truth files are deliberately NOT mounted into the
        sandbox: even a malicious adapter cannot read what was never copied.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "method": self.method,
            "params": self.params,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "gt_available": gt_available,
            "samples": [
                {"id": sid,
                 "input": str(Path(self.input_dir) / f"{sid}.png"),
                 "output": str(Path(self.output_dir) / f"{sid}.png")}
                for sid in self.sample_ids
            ],
        }
        path = self.workdir / INPUT_MANIFEST
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def read_output_manifest(self) -> Dict[str, Any]:
        path = self.workdir / OUTPUT_MANIFEST
        if not path.is_file():
            raise RunnerError(f"missing {OUTPUT_MANIFEST} at {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunnerError(f"corrupt {OUTPUT_MANIFEST}: {exc}") from exc
        if payload.get("method") != self.method:
            raise RunnerError(
                f"{OUTPUT_MANIFEST} declares method "
                f"'{payload.get('method')}' but job is for '{self.method}'"
            )
        return payload


@dataclass
class ExecutionRunner(abc.ABC):
    """Isolated invocation of an external restoration pipeline."""

    #: Human identifier used in configs (``process`` / ``conda`` / ``docker``).
    kind: ClassVar[str] = "abstract"

    workdir: Path = field(default_factory=lambda: Path(".hydro_bench_work"))
    timeout: Optional[float] = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)

    # ---------------------------------------------------------------- API
    @abc.abstractmethod
    def _build_command(self, job: RunnerJob) -> List[str]:
        """Translate a job into an argv for :func:`subprocess.run`."""
        raise NotImplementedError

    def prepare(self, job: RunnerJob) -> None:
        """Create directories and materialise the input manifest."""
        job.input_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.write_input_manifest(gt_available=job.gt_dir is not None)

    def finalize(self, job: RunnerJob, raw_result: RunResult) -> RunResult:
        """Validate child output before the harness trusts it."""
        if raw_result.returncode != 0:
            return raw_result
        try:
            manifest = job.read_output_manifest()
        except RunnerError:
            return raw_result
        raw_result.output_manifest = manifest

        # Schema-level sanity: every requested sample must be reported, and
        # every reported image must exist with the right geometry.
        produced = {s["id"]: s for s in manifest.get("samples", [])}
        missing = [sid for sid in job.sample_ids if sid not in produced]
        if missing:
            raise RunnerError(
                f"{manifest.get('method', job.method)} did not report "
                f"{len(missing)}/{len(job.sample_ids)} samples "
                f"(first: {missing[:3]})"
            )
        for sid in job.sample_ids:
            entry = produced[sid]
            out_path = Path(entry["output"])
            if not out_path.is_file():
                raise RunnerError(f"reported output missing on disk: {out_path}")
            shape = tuple(entry.get("shape", []))
            if shape:
                in_path = Path(produced[sid].get("input", job.input_dir / f"{sid}.png"))
                if in_path.is_file():
                    from PIL import Image  # optional, geometry check only
                    expected = Image.open(in_path).size[::-1]  # (H, W)
                    if tuple(shape[:2]) != expected:
                        raise RunnerError(
                            f"{sid}: output shape {shape[:2]} != input {expected}; "
                            "restoration must preserve geometry"
                        )
        return raw_result

    # ----------------------------------------------------------- invocation
    def run(self, job: RunnerJob, show_command: bool = False) -> RunResult:
        """Execute *job* in isolation and return a validated :class:`RunResult`."""
        self.prepare(job)
        command = self._build_command(job)
        if show_command:
            print("  $", " ".join(map(str, command)))
        if self.dry_run:
            return RunResult(returncode=0, command=command, workdir=self.workdir,
                             output_manifest={"method": job.method,
                                              "samples": [], "dry_run": True})

        import subprocess

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"runner '{self.kind}' is unavailable on this host: {exc.filename}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(
                f"runner timed out after {self.timeout}s: {' '.join(map(str, command))}"
            ) from exc

        result = RunResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            workdir=self.workdir,
            command=command,
        )
        return self.finalize(job, result)


# ----------------------------------------------------------------- utilities
def materialise_inputs(job: RunnerJob, images: Dict[str, np.ndarray]) -> None:
    """Write the raw images a job operates on into its sandbox input dir."""
    from PIL import Image

    for sid, img in images.items():
        if img.dtype != np.uint8:
            raise ValueError(f"{sid}: sandbox inputs must be uint8 RGB")
        Image.fromarray(img).save(job.input_dir / f"{sid}.png", optimize=False)


def collect_outputs(job: RunnerJob) -> Dict[str, np.ndarray]:
    """Read the restored images a child process produced."""
    from PIL import Image

    out: Dict[str, np.ndarray] = {}
    for sid in job.sample_ids:
        path = job.output_dir / f"{sid}.png"
        if path.is_file():
            out[sid] = np.asarray(Image.open(path).convert("RGB"))
    if len(out) != len(job.sample_ids):
        missing = set(job.sample_ids) - set(out)
        raise RunnerError(f"child produced {len(out)}/{len(job.sample_ids)} "
                          f"images (missing: {sorted(missing)[:3]})")
    return out


def fresh_workdir(base: Path, method: str, split: str) -> Path:
    """Deterministic, per-(method, split) sandbox root."""
    path = Path(base) / f"{method}__{split}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
