"""Publication plots: boxplots, radar charts, and downstream delta curves.

All figures are generated from the same :class:`~hydro_bench.reporter.
latex_tables.BenchmarkResults` object that feeds the tables, so a figure can
never contradict a table in the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..metrics.statistics import build_metric_comparison
from .latex_tables import METRIC_BETTER, METRIC_LABELS, BenchmarkResults


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    # Type-42 (TrueType) fonts in vector output: Elsevier rejects Type-3 fonts
    # in supplied vector artwork.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    return plt


# Artwork contract from paper/00_journal/requirements.md (R12/R13): PNG is not
# among the publisher's accepted formats, so every repository figure is emitted
# a second time as vector PDF (primary) and 500-dpi TIFF (combination art).
SUBMISSION_FORMATS = ("pdf", "tiff")
SUBMISSION_DPI = 500          # combination art, per R12
TIFF_RENDER_DPI = 1000        # 2x supersampling, so the resize below is always a downscale

# The publisher accepts only three widths (R12). A figure is snapped down to the
# largest of these that fits, so nothing exceeds 190 mm and no raster is
# upscaled.
STANDARD_WIDTHS_MM = (190.0, 140.0, 90.0)


def _target_width_mm(natural_mm: float) -> float:
    for width in STANDARD_WIDTHS_MM:
        if natural_mm >= width - 0.5:
            return width
    return STANDARD_WIDTHS_MM[-1]


def save_figure(fig, out_path: Path, dpi: int = 200,
                submission_dir: Optional[Path] = None) -> Path:
    """Save ``fig`` to PNG, and to the publisher's artwork formats too.

    ``bbox_inches="tight"`` crops to the drawn content, so the PNG stays as the
    repository's own rendering unchanged. When ``submission_dir`` is given the
    same render is additionally written there as vector PDF and 500-dpi TIFF,
    both on a white background: the charts are line/combination art, for which
    the publisher asks 500 dpi, and vector output has no resolution limit at
    all.

    Widths are snapped to the publisher's three standard column widths
    (190/140/90 mm). The TIFF is rendered at twice the required dpi and then
    Lanczos-downsampled to the exact pixel count for its target width, so it is
    a genuine 500 dpi at exactly the standard width; the PDF page is scaled
    losslessly instead, since vector art has no fixed resolution.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    if submission_dir is None:
        return out_path

    sub = Path(submission_dir)
    sub.mkdir(parents=True, exist_ok=True)

    import io

    from PIL import Image

    # Measure the natural width of the tight-cropped render.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", dpi=TIFF_RENDER_DPI)
    buf.seek(0)
    natural_px, natural_h = Image.open(buf).size
    natural_mm = natural_px / TIFF_RENDER_DPI * 25.4
    target_mm = _target_width_mm(natural_mm)

    # Vector: scale the page contents losslessly to the exact target width.
    pdf_path = sub / f"{out_path.stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    _scale_pdf(pdf_path, target_mm / natural_mm)

    # Raster: downsample the supersampled render to exactly 500 dpi at the
    # target width. A figure never wider than its natural render, this is
    # always a downscale.
    target_px = int(round(target_mm / 25.4 * SUBMISSION_DPI))
    target_h = max(1, int(round(natural_h * target_px / natural_px)))
    buf.seek(0)
    with Image.open(buf) as im:
        rgb = im.convert("RGB")
        rgb = rgb.resize((target_px, target_h), Image.LANCZOS)
        rgb.save(sub / f"{out_path.stem}.tiff", format="TIFF",
                 dpi=(SUBMISSION_DPI, SUBMISSION_DPI), compression="tiff_deflate")
    return out_path


def _scale_pdf(pdf_path: Path, scale: float) -> None:
    """Scale a one-page vector PDF's contents and page box by ``scale``.

    Without this the PDF page keeps its natural width, which for the wide
    boxplots is a few millimetres over the 190 mm limit.
    """
    if abs(scale - 1.0) < 1e-4:
        return
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        page.scale_by(scale)
        writer.add_page(page)
    with open(pdf_path, "wb") as fh:
        writer.write(fh)


def _bootstrap_ci(values: Sequence[float], confidence: float = 0.95,
                  n: int = 2000) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return (float(arr.mean()) if arr.size else 0.0,) * 2
    gen = np.random.default_rng(0)
    idx = gen.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ------------------------------------------------------------------ boxplot
def boxplot(results: BenchmarkResults, metric: str, out_path: Path,
            dpi: int = 200, order_by: str = "median",
            show_points: bool = True, palette: str = "Set2",
            submission_dir: Optional[Path] = None,
            methods: Optional[Sequence[str]] = None) -> Path:
    """Score distribution per method, sorted best-first.

    Boxplots (not bar charts) are the correct choice here: IQA score
    distributions are skewed, so a mean bar hides the tails a reviewer will
    ask about.
    """
    import seaborn as sns

    plt = _matplotlib()
    if metric not in results.scores:
        raise KeyError(f"metric '{metric}' not found in results")
    by_method = results.scores[metric]
    higher = METRIC_BETTER.get(metric, True)

    # Callers may restrict which methods are plotted (the UIEB figures
    # exclude the ablation variants, which were scored on a smaller image
    # set); the default is every method that has a score for the metric.
    shown = [m for m in (methods if methods is not None else by_method)]
    order = sorted(shown,
                   key=lambda m: np.median(by_method[m]),
                   reverse=higher)
    data, labels = [], []
    for method in order:
        values = [v for v in by_method[method] if v is not None and np.isfinite(v)]
        if values:
            data.append(values)
            labels.append(method)
    if not data:
        raise ValueError(f"no finite scores for metric '{metric}'")

    fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(labels) + 2.0), 3.2),
                           dpi=dpi)
    sns.boxplot(data=data, ax=ax, palette=palette, width=0.55,
                showfliers=not show_points)
    if show_points:
        sns.stripplot(data=data, ax=ax, color="0.25", size=1.6, alpha=0.45)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{results.dataset} / {results.split} — "
                 f"{METRIC_LABELS.get(metric, metric)}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, submission_dir=submission_dir)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------- radar
def radar_chart(results: BenchmarkResults, metrics: Sequence[str], out_path: Path,
                methods: Optional[Sequence[str]] = None,
                dpi: int = 200, normalize: bool = True,
                palette: str = "tab10",
                submission_dir: Optional[Path] = None) -> Path:
    """One polygon per method across several normalised metrics.

    Metrics are min-max normalised across methods (with direction corrected so
    "outward is better" for every axis), which is the only honest way to put
    PSNR and LPIPS on one chart.
    """
    plt = _matplotlib()
    methods = list(methods) if methods else results.methods()
    metrics = [m for m in metrics if m in results.scores]
    if len(metrics) < 3:
        raise ValueError("radar chart needs at least 3 metrics")

    # Collect means and normalise per metric with direction correction.
    raw = {m: {meth: _mean(results.scores[m].get(meth, [])) for meth in methods}
           for m in metrics}
    norm = {}
    for m in metrics:
        values = np.array([raw[m][meth] for meth in methods], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = values[finite].min(), values[finite].max()
        higher = METRIC_BETTER.get(m, True)
        span = max(hi - lo, 1e-12)
        for meth in methods:
            v = raw[m][meth]
            if not np.isfinite(v):
                norm[(m, meth)] = np.nan
                continue
            scaled = (v - lo) / span
            norm[(m, meth)] = scaled if higher else (1.0 - scaled)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"},
                           dpi=dpi)
    colors = plt.get_cmap(palette)(np.linspace(0, 1, len(methods)))
    for meth, color in zip(methods, colors):
        values = [norm[(m, meth)] for m in metrics]
        if all(np.isnan(values)):
            continue
        values = [0.0 if np.isnan(v) else v for v in values]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.6, label=meth)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=8)
    ax.set_yticks([])
    if normalize:
        ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{results.dataset} / {results.split}", fontsize=10, pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.06), fontsize=7)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, submission_dir=submission_dir)
    plt.close(fig)
    return out_path


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ------------------------------------------------------------- delta curves
def delta_curve(results: BenchmarkResults, metric: str, out_path: Path,
                baseline: str, dpi: int = 200,
                show_confidence: bool = True,
                submission_dir: Optional[Path] = None) -> Path:
    """Per-sample improvement of each method over the baseline.

    Curves are ordered by descending per-sample delta and the area under each
    curve is reported in the legend -- the summary statistic a reviewer wants
    when asking "is the gain consistent or driven by a few images?".
    """
    plt = _matplotlib()
    if metric not in results.scores or baseline not in results.scores[metric]:
        raise KeyError(f"metric '{metric}' or baseline '{baseline}' missing")
    base_values = results.scores[metric][baseline]
    higher = METRIC_BETTER.get(metric, True)
    sign = 1.0 if higher else -1.0

    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=dpi)
    cmap = plt.get_cmap("tab10")
    for idx, method in enumerate(results.methods()):
        if method == baseline:
            continue
        values = results.scores[metric].get(method, [])
        n = min(len(values), len(base_values))
        if n < 2:
            continue
        deltas = np.array([sign * (v - b) for v, b in zip(values[:n], base_values[:n])])
        finite = np.isfinite(deltas)
        if finite.sum() < 2:
            continue
        deltas = deltas[finite]
        ordered = np.sort(deltas)[::-1]
        auc = float(np.trapezoid(ordered, dx=1.0)) / max(len(ordered), 1)
        ax.plot(np.linspace(0, 1, len(ordered)), ordered,
                color=cmap(idx % 10), linewidth=1.4,
                label=f"{method} (AUC={auc:+.3f})")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("fraction of samples (sorted by improvement)")
    ax.set_ylabel(f"$\\Delta${METRIC_LABELS.get(metric, metric)} vs. {baseline}")
    ax.set_title(f"{results.dataset} / {results.split} — {metric}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, submission_dir=submission_dir)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- downstream
def downstream_delta_bars(results: BenchmarkResults, metric: str, out_path: Path,
                          baseline: str, dpi: int = 200,
                          submission_dir: Optional[Path] = None) -> Path:
    """mAP improvement bars (raw vs. enhanced) with bootstrap confidence ints."""
    plt = _matplotlib()
    if metric not in results.scores or baseline not in results.scores[metric]:
        raise KeyError(f"metric '{metric}' or baseline '{baseline}' missing")
    base = results.scores[metric][baseline]
    methods = [m for m in results.methods() if m != baseline]

    means, errs, labels = [], [], []
    for method in methods:
        values = results.scores[metric].get(method, [])
        n = min(len(values), len(base))
        if n < 2:
            continue
        deltas = [v - b for v, b in zip(values[:n], base[:n])]
        lo, hi = _bootstrap_ci(deltas)
        means.append(float(np.mean(deltas)))
        errs.append([float(np.mean(deltas)) - lo, hi - float(np.mean(deltas))])
        labels.append(method)
    if not means:
        raise ValueError("no methods to compare against the baseline")

    fig, ax = plt.subplots(figsize=(max(3.6, 0.6 * len(labels) + 1.6), 3.0), dpi=dpi)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=np.array(errs).T, capsize=3, color="#4c72b0", alpha=0.85)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"$\\Delta${METRIC_LABELS.get(metric, metric)}")
    ax.set_title(f"Downstream gain vs. {baseline}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi, submission_dir=submission_dir)
    plt.close(fig)
    return out_path
