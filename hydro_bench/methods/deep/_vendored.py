"""Shared loader for vendored third-party method code.

HydroBench never reimplements a published network. Deep adapters import the
authors' own modules so the released checkpoints map 1:1 onto the constructed
models, and every architectural choice stays the responsibility of the
original repository.

Those modules are reached in one of two ways:

* through the isolated environment package name (``hydro_funiegan.nets``,
  ``hydro_topal.model``, ...) when the method is executed by the conda or
  docker runner, where the repo is installed as a package of that name;
* otherwise from the on-disk vendored checkout under ``HYDROBENCH_THIRD_PARTY``
  (default: the ``third_party/`` directory beside the ``hydro_bench`` package),
  which lets the same adapter run in-process without a container.

Both paths load identical code; only the location differs.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from ...config_defaults import PACKAGE_ROOT

ENV_VAR = "HYDROBENCH_THIRD_PARTY"

# PyTorch reads this once, when the MPS allocator is first initialised; the
# ceiling it configures cannot be changed mid-run. See prepare_mps_memory_pool.
MPS_HIGH_WATERMARK_VAR = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"


def third_party_root() -> Path:
    """Resolve the directory holding the vendored method repositories."""
    root = os.environ.get(ENV_VAR)
    if root:
        return Path(root).expanduser()
    # Default: <workspace>/third_party, i.e. next to the hydro_bench package.
    return PACKAGE_ROOT.parent / "third_party"


def repo_dir(repo: str) -> Path:
    """Locate one vendored repository, failing loudly if it is absent."""
    path = third_party_root() / repo
    if not path.is_dir():
        raise FileNotFoundError(
            f"vendored repository '{repo}' not found at {path}; clone the "
            f"official repository there or set {ENV_VAR} to the directory "
            "holding it"
        )
    return path


def repo_on_syspath(repo: str, subpath: str = "") -> Path:
    """Prepend a vendored repo (or a sub-directory of it) to ``sys.path``."""
    path = repo_dir(repo)
    if subpath:
        path = path / subpath
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return path


def import_vendored(module: str, *, env_package: Optional[str] = None,
                    repo: Optional[str] = None, subpath: str = "") -> ModuleType:
    """Import a third-party module, preferring the isolated-env package.

    Parameters
    ----------
    module:
        Dotted module name as imported from inside the repository
        (e.g. ``"nets.funiegan"`` for ``funie_gan/PyTorch/nets/funiegan.py``).
    env_package:
        Package name the repo installs in its isolated environment
        (e.g. ``"hydro_funiegan"``). Tried first so a conda/docker runner keeps
        using its own pinned copy of the code.
    repo:
        Directory name under the third-party root. Required unless the module
        already resolves through ``env_package``.
    subpath:
        Sub-directory of the repository to place on ``sys.path`` when falling
        back to the vendored checkout.
    """
    if env_package:
        try:
            return importlib.import_module(f"{env_package}.{module}")
        except ImportError:
            pass  # not running inside the isolated environment
    if repo is None:
        raise ValueError(
            f"cannot import '{module}': no isolated-env package and no "
            "vendored repo name were given"
        )
    repo_on_syspath(repo, subpath)
    return importlib.import_module(module)


def import_module_from_file(path: Path, name: str) -> ModuleType:
    """Import a module whose file name is not a legal Python identifier.

    Several published repos ship files such as ``MSBDN-RDFF.py``; ``import``
    cannot name them directly, so they are loaded from their path instead.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_stub_package(dotted: str, module: ModuleType) -> None:
    """Register ``module`` under a dotted import name, creating fake parents.

    A few vendored repos import dependencies that cannot be installed on this
    host at all -- ``mamba_ssm`` ships only a CUDA build, and modern ``timm``
    pulls in ``huggingface_hub``/``safetensors`` from a package CDN that is
    unreachable here. Where the *part* the architecture actually needs has a
    faithful pure-PyTorch implementation, HydroBench provides it as a compat
    module and installs it under the dotted path the repo imports, so the
    vendored file itself stays byte-identical to its upstream release.

    Only the exact names the architecture imports are shimmed; nothing else
    about the package is pretended to exist.
    """
    parts = dotted.split(".")
    # Create (or reuse) placeholder parent packages up to the leaf.
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            placeholder = ModuleType(parent)
            placeholder.__path__ = []  # marks it as a package
            sys.modules[parent] = placeholder
    sys.modules[dotted] = module


def resolve_device(device: str = "auto") -> str:
    """Resolve an adapter's ``device`` argument to something torch can use.

    ``"auto"`` prefers an Apple GPU (MPS) or CUDA device when one is present,
    since the conv-heavy restoration networks are orders of magnitude faster
    there; on a CPU-only host it degrades to ``"cpu"``. An explicit device name
    is passed through unchanged.
    """
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # torch-free host: the adapter can only fall back
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prepare_mps_memory_pool(unbounded: bool = True) -> bool:
    """Lift Apple's MPS private-memory ceiling, before the first allocation.

    PyTorch reads ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` exactly once, when the
    MPS allocator is initialised -- the first tensor moved to ``"mps"``. The
    value is then baked in for the process: setting the variable later, after
    any MPS tensor has existed, has no effect on the pool's ceiling (verified
    empirically: an already-used allocator still reports
    ``max allowed: 9.07 GiB``). So this must run at startup, before any deep
    adapter moves a weight tensor to the accelerator.

    Why lift it at all: the Apple GPU's private pool is capped far below host
    unified memory (9.07 GiB on this host), and TOPAL's fusion stage
    concatenates five full-resolution feature maps -- a 3 Mpixel input peaks
    at ~18 GiB. With the ceiling in place that frame OOMs and would be dropped
    from the benchmark. Lifting the ceiling lets the driver back the excess
    from unified memory; the arithmetic and the weights are unchanged, so the
    output is still the published model's. ``--cap-mps-memory`` keeps the
    default ceiling for anyone who prefers a hard failure over swap.

    Returns whether the pool is unbounded as a result of this call.
    """
    import os

    if not unbounded:
        return False
    if os.environ.get(MPS_HIGH_WATERMARK_VAR, "1") == "0":
        return True  # the caller (or the environment) already lifted it
    os.environ[MPS_HIGH_WATERMARK_VAR] = "0"
    return True


def forward_with_cpu_fallback(forward, models, device: str, note: dict):
    """Run a network forward, escaping an accelerator OOM without lying.

    The Apple GPU's private memory pool is capped well below host RAM, so a
    large image can exhaust MPS even though the published network's arithmetic
    fits comfortably in main memory -- TOPAL's fusion stage concatenates five
    full-resolution feature maps, and a 3 Mpixel input pushes that past the
    9 GiB ceiling. When that happens the *identical* forward is rerun on CPU,
    which keeps the released weights' output exact; only the execution device
    changes, and that is recorded in ``note`` so the run never silently mixes
    devices. Note that the pool ceiling itself cannot be lifted from inside
    the failing forward -- see :func:`prepare_mps_memory_pool`, which the CLI
    calls at startup precisely so this fallback stays a rare last resort
    rather than the default path for big frames.

    The networks are returned to ``device`` afterwards: the OOM is a property
    of that one oversized image, not of the accelerator, and keeping the
    adapter on CPU permanently would slow the remaining images by orders of
    magnitude. ``models`` may contain ``None`` entries.
    """
    try:
        return forward(device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or device == "cpu":
            raise
        import torch

        def _flush() -> None:
            try:
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

        _flush()

        # Exact CPU rerun, with the accelerator restored afterwards.
        for model in models:
            if model is not None:
                model.cpu()
        _flush()
        try:
            out = forward("cpu")
        finally:
            for model in models:
                if model is not None:
                    model.to(device)
            _flush()
        count = note.get("cpu_fallback_images", 0) + 1
        note["cpu_fallback_images"] = count
        note["cpu_fallback_reason"] = (
            f"{count} image(s) exceeded the '{device}' accelerator's memory "
            "and were reran on CPU (identical weights and arithmetic; "
            "execution device only)")
        return out
