"""HydroBench core: dataset integrity, algorithm plugins, and execution runners.

Implements the two invariants that make benchmark results defensible in peer
review:

1. Zero data leakage -- every dataset is validated against an immutable
   manifest (sha256 checksums + explicit split lists) before any metric runs,
   and cross-split overlap is a hard error.
2. Environment isolation -- algorithms run behind an ExecutionRunner so that
   methods with conflicting CUDA/PyTorch/Python stacks never interfere.
"""

__version__ = "0.1.0"

__all__ = [
    "base_algorithm",
    "base_runner",
    "conda_runner",
    "dataset_manager",
    "docker_runner",
    "manifest",
]
