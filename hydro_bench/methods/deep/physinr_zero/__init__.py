"""PhysINR-Zero benchmark method: physics-constrained implicit neural field
for zero-shot underwater image enhancement.

This directory is the single home of the PhysINR-Zero method -- implementation,
benchmark adapter and tests -- alongside the other baseline methods under
``hydro_bench/methods/``. It was moved here from the repository-root package
``physinr_zero/`` when PhysINR-Zero became the framework's first published
baseline (see MIGRATION.md at the repository root for the 2026-10-01 move and
the compatibility shim that keeps the old import path and old checkpoints
working).

Submodules
----------
adapter      the HydroBench ``BaseAlgorithm`` wrapper (what methods.yaml loads).
physics      Jerlov water-type priors, image formation, re-degradation, colour.
model        feature extractor + continuous neural field (the ``PhysINR`` net).
losses       the seven loss groups and the progressive weight schedule.
synthesize   physically sampled synthetic degradation tasks (meta-training).
meta_train   FOMAML meta-initialisation from synthetic tasks only.
adapt        single-image test-time adaptation loop.
cli          ``python -m hydro_bench.methods.deep.physinr_zero.cli`` entry point.

The published checkpoints (``weights/physinr_zero_*.pt``) pickle classes under
the old ``physinr_zero.*`` module paths; the repository-root shim package
``physinr_zero/__init__.py`` aliases those paths here so they keep loading.
"""

from .config import PhysINRConfig

__all__ = ["PhysINRConfig"]
__version__ = "0.1.0"
