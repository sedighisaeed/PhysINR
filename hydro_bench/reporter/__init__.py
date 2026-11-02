"""Reporting layer: LaTeX tables, qualitative figures, and publication plots.

Every artefact is generated from one ``BenchmarkResults`` object, so the
numbers in a table and the numbers in a figure can never diverge.
"""

from . import latex_tables, plots, visualizer

__all__ = ["latex_tables", "plots", "visualizer"]
