"""Conda subprocess runner (``conda run -n <env> python adapter.py ...``).

Executes a method's adapter inside an isolated conda environment so that
conflicting Python/PyTorch/CUDA stacks can coexist on one host. The adapter
script is HydroBench's own thin shim, which is why every method gets an
identical command line regardless of its internal complexity.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_runner import INPUT_MANIFEST, ExecutionRunner, RunnerError


@dataclass
class CondaRunner(ExecutionRunner):
    """Run an adapter via ``conda run -n <env_name> python <adapter>``."""

    kind = "conda"

    env_name: str = ""
    python_exe: str = "python"
    adapter_script: Optional[Path] = None   # path to the adapter entry point
    # Extra path components prepended to the child's PATH (e.g. a CUDA toolkit
    # that a particular env expects but that is not the system default).
    extra_path: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    conda_exe: Optional[str] = None        # override autodetection of `conda`

    def __post_init__(self) -> None:
        # ExecutionRunner is a dataclass: the base __post_init__ normalises
        # workdir, so only validate the conda-specific fields here.
        if not self.env_name:
            raise RunnerError("CondaRunner requires env_name")

    # ---------------------------------------------------------- resolution
    def _find_conda(self) -> str:
        """Locate a conda executable without importing anything conda-side."""
        if self.conda_exe and shutil.which(self.conda_exe):
            return self.conda_exe
        for candidate in ("conda", "mamba", "micromamba"):
            found = shutil.which(candidate)
            if found:
                return found
        # Conda is often on PATH only after ``conda init``; fall back to the
        # well-known installation roots.
        for base in (os.environ.get("CONDA_EXE", ""),
                     os.path.expanduser("~/miniconda3/bin/conda"),
                     os.path.expanduser("~/anaconda3/bin/conda"),
                     "/opt/conda/bin/conda"):
            if base and Path(base).is_file():
                return base
        raise RunnerError(
            "no conda/mamba executable found. Install conda or set conda_exe=."
        )

    def _resolved_env(self) -> str:
        return os.environ.get("HYDROBENCH_CONDA_ENV_OVERRIDE", self.env_name)

    # --------------------------------------------------------------- build
    def _adapter_path(self, job) -> Path:
        if self.adapter_script is not None:
            return Path(self.adapter_script)
        # Convention: <repo>/hydro_bench_adapter.py shipped by the method.
        here = Path(__file__).resolve().parent.parent
        candidate = here / "methods" / "_sandbox" / f"{job.method}__adapter.py"
        if not candidate.is_file():
            raise RunnerError(
                f"CondaRunner has no adapter script for '{job.method}'. Either "
                f"ship {candidate} or pass adapter_script=."
            )
        return candidate

    def _build_command(self, job) -> List[str]:
        conda = self._find_conda()
        adapter = self._adapter_path(job)
        env = self._resolved_env()
        cmd = [
            conda, "run", "--no-capture-output", "-n", env,
            self.python_exe, str(adapter),
            "--input-manifest", str(job.workdir / INPUT_MANIFEST),
        ]
        if self.dry_run:
            cmd.append("--dry-run")
        return cmd

    def run(self, job, show_command: bool = False):
        # Inject PATH/env tweaks for the child only; leave the parent alone.
        prior_env: Dict[str, str] = {}
        if self.extra_path:
            prior_env["PATH"] = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(
                [*self.extra_path, prior_env["PATH"]]
            )
        for key, value in self.extra_env.items():
            prior_env.setdefault(key, os.environ.get(key, ""))
            os.environ[key] = value
        try:
            return super().run(job, show_command=show_command)
        finally:
            for key, value in prior_env.items():
                if value == "":
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def env_sanity_check(self) -> Dict[str, Any]:
        """Report (without executing) whether the target env looks usable."""
        out: Dict[str, Any] = {"kind": "conda", "env": self._resolved_env()}
        try:
            conda = self._find_conda()
        except RunnerError as exc:
            out["conda"] = str(exc)
            out["available"] = False
            return out
        out["conda"] = conda
        out["available"] = True
        return out
