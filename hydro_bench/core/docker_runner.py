"""Docker container runner.

Mounts the job's input/output directories read-only/read-write into a
container and invokes the method's adapter inside it. This is the strongest
isolation available: the child cannot see the host filesystem, the ground
truth directory, or any other method's working tree.

The container-side adapter reads the *same* ``input_manifest.json`` the conda
runner consumes, so a method is fully portable between the two runners by
changing one line in ``configs/methods.yaml``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_runner import INPUT_MANIFEST, ExecutionRunner, RunnerError

# Where the sandbox appears inside the container (fixed by convention).
CONTAINER_ROOT = "/work"
CONTAINER_INPUT = f"{CONTAINER_ROOT}/inputs"
CONTAINER_OUTPUT = f"{CONTAINER_ROOT}/outputs"


@dataclass
class DockerRunner(ExecutionRunner):
    """Run an adapter inside ``docker run --gpus all -v ...``."""

    kind = "docker"

    image: str = ""
    adapter_script: Optional[Path] = None
    docker_exe: Optional[str] = None
    gpus: Optional[str] = "all"     # "all", "0", "0,1", or None to disable
    extra_volumes: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    user: Optional[str] = None      # uid:gid to avoid root-owned outputs
    network: str = "none"           # offline by default; methods must not phone home
    env: Dict[str, str] = field(default_factory=dict)
    shm_size: str = "8g"

    def __post_init__(self) -> None:
        # ExecutionRunner is a dataclass: the base __post_init__ normalises
        # workdir, so only validate the docker-specific fields here.
        if not self.image:
            raise RunnerError("DockerRunner requires image=")

    # ------------------------------------------------------------------ io
    def _find_docker(self) -> str:
        exe = self.docker_exe or "docker"
        resolved = shutil.which(exe)
        if resolved:
            return resolved
        raise RunnerError("no docker executable found on PATH")

    def _adapter_path(self, job) -> Path:
        if self.adapter_script is not None:
            return Path(self.adapter_script)
        here = Path(__file__).resolve().parent.parent
        candidate = here / "methods" / "_sandbox" / f"{job.method}__adapter.py"
        if not candidate.is_file():
            raise RunnerError(
                f"DockerRunner has no adapter script for '{job.method}'. Either "
                f"ship {candidate} or pass adapter_script=."
            )
        return candidate

    def _manifest_inside_container(self, job) -> Path:
        """The manifest path the container sees (workdir is mounted read-write)."""
        return Path(f"{CONTAINER_ROOT}/{job.workdir.name}/{INPUT_MANIFEST}")

    # --------------------------------------------------------------- build
    def _build_command(self, job) -> List[str]:
        docker = self._find_docker()
        adapter = self._adapter_path(job)
        workdir_name = job.workdir.name

        cmd: List[str] = [docker, "run", "--rm"]
        if self.gpus:
            cmd += ["--gpus", self.gpus]
        cmd += ["--network", self.network,
                "--shm-size", self.shm_size]

        # Read-write mount for the sandbox (inputs + outputs + manifests).
        cmd += ["-v", f"{job.workdir.resolve()}:{CONTAINER_ROOT}/{workdir_name}"]
        # Inputs are mounted read-only so a buggy adapter cannot corrupt them.
        cmd += ["-v", f"{job.input_dir.resolve()}:{CONTAINER_INPUT}:ro"]
        cmd += ["-v", f"{job.output_dir.resolve()}:{CONTAINER_OUTPUT}"]
        # NOTE: gt_dir is *deliberately never mounted*. See base_runner docs.
        for vol in self.extra_volumes:
            cmd += ["-v", vol]
        if self.user:
            cmd += ["--user", self.user]
        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += self.extra_args
        cmd += [self.image, "python", str(adapter),
                "--input-manifest",
                str(self._manifest_inside_container(job))]
        if self.dry_run:
            cmd.append("--dry-run")
        return cmd

    # --------------------------------------------------------------- check
    def env_sanity_check(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": "docker", "image": self.image}
        try:
            exe = self._find_docker()
        except RunnerError as exc:
            out["docker"] = str(exc)
            out["available"] = False
            return out
        out["docker"] = exe
        out["available"] = True

        import subprocess

        try:
            proc = subprocess.run([exe, "image", "inspect", self.image],
                                  capture_output=True, text=True, timeout=120)
            out["image_present"] = proc.returncode == 0
            if proc.returncode != 0:
                out["hint"] = (f"pull the image first: {exe} pull {self.image}")
        except Exception as exc:  # pragma: no cover - host-specific
            out["image_present"] = False
            out["hint"] = str(exc)
        return out
