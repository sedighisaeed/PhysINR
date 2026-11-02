"""Camera-ready LaTeX table generation (booktabs, with significance tests).

Produces tables that drop straight into an IEEE/ACM submission:

* ``booktabs`` rules only (no vertical lines, no ``\hline``);
* best score **bold**, second-best <u>underlined</u>, computed from the data
  rather than hardcoded;
* a companion significance table reporting the paired Wilcoxon signed-rank
  test of every method against a chosen baseline, with Holm correction;
* ``\pm``-formatted mean/std cells, matching the convention in CVPR/TIP papers.

Everything is generated from the same in-memory results object, so a table can
never disagree with the numbers in a figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..metrics.statistics import (Comparison, MetricComparisonTable,
                                  SampleStats, build_metric_comparison)

# Direction of "better" for metrics HydroBench knows about.
METRIC_BETTER: Dict[str, bool] = {
    "psnr": True,
    "ssim": True,
    "ms_ssim": True,
    "lpips_alex": False,
    "lpips_vgg": False,
    "uciqe": True,
    "uiqm": True,
    "uicm": True,
    "uism": True,
    "uiconm": True,
    "ccf": False,
    "brisque": False,
    "musiq": True,
    "miou": True,
    "pixel_accuracy": True,
    "map_5095": True,
    "map_50": True,
    "mar_5095": True,
    "inlier_ratio": True,
    "repeatability": True,
}

# Human-readable column headers.
METRIC_LABELS: Dict[str, str] = {
    "psnr": "PSNR$\\uparrow$",
    "ssim": "SSIM$\\uparrow$",
    "ms_ssim": "MS-SSIM$\\uparrow$",
    "lpips_alex": "LPIPS$\\downarrow$",
    "lpips_vgg": "LPIPS$_{\\rm vgg}\\downarrow$",
    "uciqe": "UCIQE$\\uparrow$",
    "uiqm": "UIQM$\\uparrow$",
    "uicm": "ICM$\\uparrow$",
    "uism": "SM$\\uparrow$",
    "uiconm": "ConM$\\uparrow$",
    "ccf": "CCF$\\downarrow$",
    "brisque": "BRISQUE$\\downarrow$",
    "musiq": "MUSIQ$\\uparrow$",
    "miou": "mIoU$\\uparrow$",
    "pixel_accuracy": "PixAcc$\\uparrow$",
    "map_5095": "mAP$_{50:95}\\uparrow$",
    "map_50": "mAP$_{50}\\uparrow$",
    "mar_5095": "mAR$_{50:95}\\uparrow$",
    "inlier_ratio": "Inlier$\\uparrow$",
    "repeatability": "Repeat.$\\uparrow$",
}


def _latex_escape(text: str) -> str:
    return (text.replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def _fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "--"
    if math.isinf(value):
        return r"$\infty$"
    if math.isnan(value):
        return "--"
    return f"{value:.{decimals}f}"


def _pm(mean: Optional[float], std: Optional[float], decimals: int = 2) -> str:
    return (f"$_{{{_fmt(mean, decimals)}}}^{{\\pm {_fmt(std, decimals)}}}$"
            if mean is not None and std is not None else "--")


# ------------------------------------------------------------------ results
@dataclass
class BenchmarkResults:
    """All scores from one benchmark run, organised for the reporter."""

    dataset: str
    split: str
    # metric name -> method name -> list of per-sample scores
    scores: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    method_order: List[str] = field(default_factory=list)
    # Optional per-method provenance (e.g. fallback warnings for adapters).
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def methods(self) -> List[str]:
        if self.method_order:
            return list(self.method_order)
        methods: List[str] = []
        for by_method in self.scores.values():
            for m in by_method:
                if m not in methods:
                    methods.append(m)
        return methods

    def add(self, metric: str, method: str, values: List[float]) -> None:
        self.scores.setdefault(metric, {}).setdefault(method, []).extend(values)
        if method not in self.method_order:
            self.method_order.append(method)

    def comparison(self, metric: str, baseline: Optional[str] = None,
                   confidence: float = 0.95) -> Optional[MetricComparisonTable]:
        if metric not in self.scores:
            return None
        higher = METRIC_BETTER.get(metric, True)
        return build_metric_comparison(self.scores[metric], metric=metric,
                                       higher_is_better=higher,
                                       baseline=baseline, confidence=confidence)

    def metric_names(self) -> List[str]:
        return list(self.scores)


# -------------------------------------------------------------- main tables
@dataclass
class LatexTableGenerator:
    """Renders benchmark results as LaTeX."""

    results: BenchmarkResults
    baseline: Optional[str] = None       # reference method for significance tests
    decimals: int = 2
    highlight_best: bool = True
    show_std: bool = True
    confidence: float = 0.95

    # ---------------------------------------------------------------- main
    def render(self, metrics: Optional[Sequence[str]] = None,
               caption: str = "", label: str = "") -> str:
        """One wide table: methods as rows, metrics as columns."""
        metrics = list(metrics) if metrics else self.results.metric_names()
        methods = self.results.methods()
        comparisons = {m: self.results.comparison(m, self.baseline,
                                                  self.confidence)
                       for m in metrics}
        ranks = {m: (comparisons[m].ranks if comparisons[m] else {})
                 for m in metrics}
        n = len(methods)

        lines: List[str] = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"  \centering")
        if caption:
            lines.append(f"  \\caption{{{caption}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")

        # Column spec: method + one per metric.
        spec = "l" + "c" * len(metrics)
        lines.append(f"  \\begin{{tabular}}{{{spec}}}")
        lines.append(r"    \toprule")

        header = "Method & " + " & ".join(
            METRIC_LABELS.get(m, _latex_escape(m)) for m in metrics)
        lines.append(f"    {header} \\\\")
        lines.append(r"    \midrule")

        for method in methods:
            cells = []
            for metric in metrics:
                stat = _stat_for(comparisons[metric], method)
                cells.append(self._cell(stat, ranks[metric].get(method, 0)))
            row = f"    {_latex_escape(method)} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines) + "\n"

    def _cell(self, stat: Optional[SampleStats], rank: int) -> str:
        if stat is None:
            return "--"
        text = _pm(stat.mean, stat.std if self.show_std else None, self.decimals)
        if self.highlight_best:
            if rank == 1:
                text = r"\textbf{" + text + "}"
            elif rank == 2:
                text = r"\underline{" + text + "}"
        return text

    # ------------------------------------------------------- significance
    def render_significance(self, metrics: Optional[Sequence[str]] = None,
                            caption: str = "", label: str = "") -> str:
        """Paired Wilcoxon signed-rank tests vs. the baseline, Holm-corrected."""
        metrics = list(metrics) if metrics else self.results.metric_names()
        if not self.baseline:
            return ("%% significance table requires a baseline method; "
                    "pass baseline= to the generator\n")
        lines: List[str] = [r"\begin{table}[t]", r"  \centering"]
        if caption:
            lines.append(f"  \\caption{{{caption}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
        lines.append(r"  \begin{tabular}{llcccc}")
        lines.append(r"    \toprule")
        lines.append(r"    Metric & Method & $\Delta$ & $p$-value & Sig. & $r$ \\")
        lines.append(r"    \midrule")

        any_rows = False
        for metric in metrics:
            table = self.results.comparison(metric, self.baseline, self.confidence)
            if not table:
                continue
            for comp in table.comparisons:
                if comp.p_value is None:
                    continue
                any_rows = True
                sig = ("\\checkmark" if comp.significant else "--")
                label_m = METRIC_LABELS.get(metric, _latex_escape(metric))
                delta = _fmt(comp.delta_mean, self.decimals)
                p = (f"{comp.p_value:.2e}" if comp.p_value < 0.001
                     else f"{comp.p_value:.3f}")
                eff = _fmt(comp.effect_size, self.decimals)
                lines.append(
                    f"    {label_m} & {_latex_escape(comp.method_b)} & "
                    f"{delta} & {p} & {sig} & {eff} \\\\")
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        if not any_rows:
            return "%% no significance rows were available\n"
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ ranking
    def render_ranking(self, metrics: Optional[Sequence[str]] = None) -> str:
        """Compact 1st/2nd ranking summary, for the rebuttal period."""
        metrics = list(metrics) if metrics else self.results.metric_names()
        out = [r"% Best / second-best summary (auto-generated)"]
        for metric in metrics:
            table = self.results.comparison(metric, self.baseline, self.confidence)
            if not table:
                continue
            best = table.best()
            second = table.second_best()
            out.append(f"% {metric}: best={best.method} ({best.mean:.3f}), "
                       f"runner-up={second.method} ({second.mean:.3f})"
                       if second else f"% {metric}: best={best.method}")
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ io
    def write(self, out_path: Path, metrics: Optional[Sequence[str]] = None,
              caption: str = "", label: str = "",
              include_significance: bool = True) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [self.render(metrics, caption=caption, label=label)]
        if include_significance:
            parts.append(self.render_significance(
                metrics,
                caption=("Paired Wilcoxon signed-rank test vs. "
                         f"{self.baseline} (Holm-corrected)"),
                label=(label + "_sig" if label else "")))
        parts.append(self.render_ranking(metrics))
        out_path.write_text("\n".join(parts), encoding="utf-8")
        return out_path


def _stat_for(table: Optional[MetricComparisonTable], method: str
              ) -> Optional[SampleStats]:
    if table is None:
        return None
    for stat in table.stats:
        if stat.method == method:
            return stat
    return None


# ----------------------------------------------------------- markdown mirror
def render_markdown(results: "BenchmarkResults",
                    metrics: Optional[Sequence[str]] = None,
                    baseline: Optional[str] = None) -> str:
    """A plain-text mirror of the main table for READMEs and CI logs.

    A method that could not run its published model is still shown -- its row
    is real scored output -- but the name is marked ``[fallback]`` so the
    numbers are never mistaken for the baseline's own.
    """
    metrics = list(metrics) if metrics else results.metric_names()
    methods = results.methods()
    comparisons = {m: results.comparison(m, baseline) for m in metrics}
    ranks = {m: (comparisons[m].ranks if comparisons[m] else {}) for m in metrics}

    header = ["Method"] + [m for m in metrics]
    rows = [header, ["---"] * len(header)]
    fallbacks = []
    for method in methods:
        meta = results.metadata.get(method, {})
        label = method
        if meta.get("used_fallback"):
            label = f"{method} [fallback]"
            note = meta.get("note")
            if note:
                fallbacks.append(f"{method}: {note}")
        row = [label]
        for metric in metrics:
            stat = _stat_for(comparisons[metric], method)
            if stat is None:
                row.append("--")
            else:
                mark = {1: "*", 2: "+"}.get(ranks[metric].get(method, 0), "")
                row.append(f"{stat.mean:.3f}±{stat.std:.3f}{mark}")
        rows.append(row)
    width = max(len(c) for c in rows)
    table = "\n".join(
        "| " + " | ".join(cell.ljust(width) for cell in row) + " |"
        for row in rows)
    if fallbacks:
        table += ("\n\n[fallback] rows did not run the published model:\n"
                  + "\n".join(f"- {f}" for f in fallbacks) + "\n")
    return table + "\n"
