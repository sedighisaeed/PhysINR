"""Restoration methods: online adapters (Mode A) and offline import (Mode B).

Mode A -- online / plugin
    An adapter implements :class:`~hydro_bench.core.base_algorithm.BaseAlgorithm`
    and runs inside HydroBench (``runner: process``) or inside an isolated
    conda/docker environment (``runner: conda|docker``) via the runners in
    ``hydro_bench/core``.

Mode B -- offline / pre-computed
    :mod:`hydro_bench.methods.offline_importer` reads a directory tree of
    already-enhanced images and presents it through the *identical*
    evaluation interface, so methods whose code cannot be executed still get
    the same metrics, the same splits, and the same statistical treatment.
"""

__all__ = ["offline_importer"]
