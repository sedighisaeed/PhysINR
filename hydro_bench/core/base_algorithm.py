"""Abstract base class for HydroBench algorithm plugins (Mode A).

A plugin is the *only* interface a new method has to implement in order to be
benchmarked in online mode. The contract is deliberately tiny:

.. code-block:: python

    class MyAlgorithm(BaseAlgorithm):
        name = "MyAlgorithm"

        def __init__(self, checkpoint: PathLike | None = None,
                     device: str = "cpu", **params): ...

        def enhance(self, image: np.ndarray) -> np.ndarray: ...

`enhance` takes a raw underwater image (HxWx3 uint8 RGB) and returns a
restored image with identical shape and dtype. Nothing else is exposed, so
plugins cannot accidentally read the ground truth, peek at other methods, or
touch the evaluation harness.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class AlgorithmCapabilities:
    """What a plugin promises about its own interface.

    HydroBench uses this to decide which metrics are even applicable. A method
    that only operates on 8-bit inputs should declare ``max_bit_depth=8`` so
    the harness does not silently downconvert 16-bit data.
    """

    supports_color: bool = True
    supports_grayscale: bool = True
    max_bit_depth: int = 8
    requires_gpu: bool = False
    # Set of dataset names this method was *developed* on. HydroBench logs a
    # warning if a method is evaluated on a set it was trained on -- a common
    # source of inflated scores.
    training_sets: tuple = field(default_factory=tuple)


class BaseAlgorithm(abc.ABC):
    """Interface every online (Mode A) algorithm must satisfy."""

    # Human-readable name used in tables and figures. Overridden by subclasses.
    name: str = "BaseAlgorithm"

    # Populated by __init_subclass__ so the registry is export-free.
    capabilities: AlgorithmCapabilities = AlgorithmCapabilities()

    # Registry of subclasses keyed by name (populated automatically).
    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is None:
            raise TypeError(f"{cls.__name__} must define a non-None class attribute 'name'")
        BaseAlgorithm._registry[cls.name] = cls

    # ------------------------------------------------------------------ API
    @abc.abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Restore a single image.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 RGB array (row 0 at the top).

        Returns
        -------
        np.ndarray
            Restored image, same shape and dtype as the input.
        """
        raise NotImplementedError

    # --------------------------------------------------------------- helpers
    def enhance_batch(self, images: list) -> list:
        """Default sequential loop; override for parallel/GPU batching."""
        return [self.enhance(img) for img in images]

    def warmup(self, shape: tuple = (256, 256, 3)) -> None:
        """Optional one-shot initialisation (e.g. lazy weight loading)."""

    def cleanup(self) -> None:
        """Release resources after a run completes."""

    # ----------------------------------------------------------- provenance
    def run_metadata(self) -> Dict[str, Any]:
        """What this method actually was, for the results file.

        The default states the method ran as itself with no fallback: a
        classical method has no checkpoint to be missing, so it is always the
        real thing. Adapters that *can* be unrunnable override this to report
        ``used_fallback`` and why.
        """
        return {
            "method": self.name,
            "used_fallback": False,
            "note": "official implementation",
            "checkpoint": None,
        }

    # ------------------------------------------------------------- contract
    def _validate_io(self, image: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Enforce the shape/dtype contract before returning to the harness."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{self.name}: input must be np.ndarray, got {type(image)}")
        if image.ndim != 3 or image.shape[2] not in (1, 3):
            raise ValueError(f"{self.name}: expected HxWx3 uint8 RGB, got {image.shape}")
        if out.shape != image.shape:
            raise ValueError(
                f"{self.name}: output shape {out.shape} != input shape {image.shape}. "
                "Restoration must preserve geometry; resampling pollutes FR metrics."
            )
        if out.dtype != image.dtype:
            raise TypeError(f"{self.name}: output dtype {out.dtype} != input {image.dtype}")
        return out


def _import_all_adapters() -> None:
    """Import every ``adapter.py`` under ``hydro_bench/methods`` once.

    Plugin classes register themselves through ``__init_subclass__``, but that
    only happens when their module is imported. Walking the methods package at
    discovery time makes a method usable without an explicit import, so
    ``run_benchmark.py`` can name methods by string alone.
    """
    if discover_algorithm._imported:  # type: ignore[attr-defined]
        return
    import importlib
    import pkgutil

    import hydro_bench.methods as methods_pkg

    for module_info in pkgutil.walk_packages(methods_pkg.__path__,
                                             prefix=methods_pkg.__name__ + "."):
        if module_info.name.endswith(".adapter"):
            try:
                importlib.import_module(module_info.name)
            except Exception:
                # A broken adapter must not prevent discovery of the others.
                continue
    discover_algorithm._imported = True  # type: ignore[attr-defined]


def discover_algorithm(name: str) -> type:
    """Look up a registered plugin class by its ``name`` attribute."""
    _import_all_adapters()
    try:
        return BaseAlgorithm._registry[name]
    except KeyError:
        available = ", ".join(sorted(BaseAlgorithm._registry)) or "(none registered)"
        raise KeyError(f"unknown algorithm '{name}'. Registered: {available}") from None


def registered_algorithms() -> Dict[str, type]:
    _import_all_adapters()
    return dict(BaseAlgorithm._registry)


# Sentinel attribute used by _import_all_adapters (defined after the function).
discover_algorithm._imported = False


def load_weights(path: Optional[PathLike], must_exist: bool = True) -> Optional[Path]:
    """Resolve a checkpoint path, honouring HYDROBENCH_MODELS as a base dir.

    HydroBench never downloads weights itself. If a checkpoint is missing the
    user must obtain it from the method's official repository.
    """
    if path is None:
        if must_exist:
            raise FileNotFoundError("a checkpoint path is required for this method")
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        base = os.environ.get("HYDROBENCH_MODELS")
        if base:
            p = Path(base) / p
    if must_exist and not p.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {p}. Obtain it from the method's official "
            "repository; HydroBench does not download weights."
        )
    return p
