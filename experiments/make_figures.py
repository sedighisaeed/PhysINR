"""Generate the manuscript's figures and tables from the merged results.

Every artefact this produces is read out of a results file that the benchmark
itself wrote; nothing here computes a score. If an input is missing the script
names it and moves on, so a partial run yields a partial deliverable rather
than a fabricated one.

Inputs  : experiments/merged_results/<DATASET>__test/results.json
Outputs : paper/03_figures/*.png, paper/04_tables/*.tex, plus a provenance
          manifest that pins every figure to the results file behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
# Running this file directly puts experiments/ on sys.path, not the repo root,
# where hydro_bench lives.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS_PAIRED = ["EUVP", "LSUI", "UIEB"]      # have reference images
DATASETS_UNPAIRED = ["UCCS", "RUIE"]
FR_METRICS = ["psnr", "ssim", "ms_ssim", "lpips_alex"]
NR_METRICS = ["uciqe", "uiqm", "uicm", "uism", "uiconm"]

FIG_DIR = REPO / "paper" / "03_figures"
TAB_DIR = REPO / "paper" / "04_tables"

# The one number the reader is asked to trust most, so it gets its own trace.
PRIMARY_METHOD = "PhysINR-Zero"

ABLATION_PREFIX = PRIMARY_METHOD + " ("


def _is_ablation(method: str) -> bool:
    """Same rule as make_uieb_tables.py: variants are named 'PhysINR-Zero (...)'.

    On UIEB the ablation variants were scored on a 40-image subset, so a
    comparison figure that mixed them with the 134-image baselines would be
    comparing unmatched samples. The main tables exclude them for that reason;
    the main comparison figures must agree with the tables they sit beside.
    """
    return method.startswith(ABLATION_PREFIX)


# Ablation labels appear on figure axes, where a bare underscore would be drawn
# as a matplotlib subscript. Spell the symbols so the label reads as physics
# rather than as an identifier.
_VARIANT_PRETTY = {
    "tau_max = 2": r"$\tau_{\max} = 2$",
    "tau_max = 4": r"$\tau_{\max} = 4$",
    "delta_max = 0.5": r"$\delta_{\max} = 0.5$",
    "delta_max = 4": r"$\delta_{\max} = 4$",
    "coord stride 1": "coord stride 1",
    "coord stride 3": "coord stride 3",
    "5 adaptation steps": "5 adaptation steps",
    "20 adaptation steps": "20 adaptation steps",
}


def _pretty_variant(name: str) -> str:
    """Render an ablation label for a figure axis (no raw underscores)."""
    return _VARIANT_PRETTY.get(name, name.replace("_", " "))


# ------------------------------------------------------------------ loading
def load_merged(dataset: str, merged_root: Path) -> Optional[Any]:
    path = merged_root / f"{dataset}__test" / "results.json"
    if not path.is_file():
        print(f"  skip {dataset}: {path} not found")
        return None
    from hydro_bench.reporter.latex_tables import BenchmarkResults

    blob = json.loads(path.read_text(encoding="utf-8"))
    # The merged file mixes the main comparison with the ablation variants,
    # which were scored on a 40-image subset and cannot be paired with the
    # full-split columns. The comparison tables and boxplots here are the
    # main-comparison views; make_uieb_tables.py renders the ablation tables.
    # Ablation columns are labelled "PhysINR-Zero (<variant>)".
    scores = blob.get("scores", {})
    primary = [m for m in scores.get(
        next(iter(scores)), {}) if "(" not in m]
    if primary and any("(" in m for m in scores.get(next(iter(scores)), {})):
        scores = {metric: {m: v for m, v in by.items() if m in primary}
                  for metric, by in scores.items()}
    res = BenchmarkResults(dataset=blob.get("dataset", dataset),
                           split=blob.get("split", "test"),
                           scores=scores,
                           # The renderer wants the flat method -> provenance
                           # mapping, not the whole metadata envelope: passing
                           # the envelope hides every "used_fallback" row, so
                           # the table would print fallback numbers unlabelled.
                           metadata=blob.get("metadata", {}).get(
                               "method_metadata", {}))
    # Preserve the column order of the merged file, which is the baseline order
    # followed by PhysINR-Zero and its ablations.
    order: List[str] = []
    for by_method in res.scores.values():
        for m in by_method:
            if m not in order:
                order.append(m)
    res.method_order = order
    return res


def load_ablation_curve(log_dir: Path) -> Optional[Dict[str, Any]]:
    """The meta-training loss curve, if the run recorded one.

    The curve lives in the checkpoint's provenance file (``history`` field),
    written by ``MetaTrainer`` so the figure is one file read away rather than
    a re-run.
    """
    for cand in sorted((REPO / "weights").glob("physinr_zero_meta.provenance.json"),
                       reverse=True):
        try:
            blob = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if blob.get("history"):
            return blob
    for cand in sorted(log_dir.glob("meta_train_*.json"), reverse=True):
        try:
            blob = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if blob.get("history"):
            return blob
    return None


# ------------------------------------------------------------------ figures
def fig_boxplots(results: Any, dataset: str, metrics: List[str],
                 better_note: str, provenance: Dict[str, Any],
                 submission_dir: Optional[Path] = None) -> List[Path]:
    from hydro_bench.reporter.plots import boxplot

    out: List[Path] = []
    for metric in metrics:
        if metric not in results.scores:
            continue
        if len(results.scores[metric]) < 2:
            continue
        path = FIG_DIR / f"box_{dataset}_{metric}.png"
        boxplot(results, metric, path, dpi=300, submission_dir=submission_dir,
                methods=[m for m in results.methods() if not _is_ablation(m)])
        out.append(path)
        provenance[str(path)] = {
            "source": f"{dataset}__test/results.json",
            "metric": metric,
            "n_methods": len(results.scores[metric]),
        }
    if not out:
        print(f"  {dataset}: no {better_note} metrics with >=2 methods scored")
    return out


def fig_radar(results: Any, dataset: str, metrics: List[str],
              provenance: Dict[str, Any],
              submission_dir: Optional[Path] = None) -> Optional[Path]:
    have = [m for m in metrics if m in results.scores
            and len(results.scores[m]) >= 2]
    if len(have) < 3:
        print(f"  {dataset}: radar needs >=3 metrics, have {len(have)}")
        return None
    from hydro_bench.reporter.plots import radar_chart

    path = FIG_DIR / f"radar_{dataset}.png"
    radar_chart(results, have, path, dpi=300, submission_dir=submission_dir,
                methods=[m for m in results.methods() if not _is_ablation(m)])
    provenance[str(path)] = {
        "source": f"{dataset}__test/results.json", "metrics": have}
    return path


def fig_ablation_bars(merged_root: Path, dataset: str,
                      provenance: Dict[str, Any],
                      submission_dir: Optional[Path] = None) -> Optional[Path]:
    """Per-variant mean shift on the primary metric, relative to the full model.

    Read from the merged file, where every variant is already its own column.
    """
    # load_merged() drops the ablation columns (they are not length-matched to
    # the main comparison), so read the raw file here: this function is the one
    # consumer that needs the variants alongside the full model's column.
    path = merged_root / f"{dataset}__test" / "results.json"
    if not path.is_file():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    scores = blob.get("scores", {})
    for metric in ("psnr", "uciqe"):
        if metric in scores and PRIMARY_METHOD in scores[metric]:
            break
    else:
        print(f"  {dataset}: no primary metric for the ablation bars")
        return None

    variants = [m for m in scores[metric]
                if m.startswith(PRIMARY_METHOD) and m != PRIMARY_METHOD]
    if len(variants) < 2:
        print(f"  {dataset}: need >=2 ablation variants, have {len(variants)}")
        return None

    import numpy as np

    from hydro_bench.reporter import plots

    plt = plots._matplotlib()

    # The variants were scored on a --limit N subset, but the full model's
    # column runs over the whole split. Truncate to the first N so the bars
    # are deltas against the same images the variants saw; load_split takes
    # ids[:limit], so the front of the full column is exactly that subset.
    full_col = scores[metric][PRIMARY_METHOD]
    n = min(len(scores[metric][v]) for v in variants)
    if len(full_col) > n:
        print(f"  {dataset}: truncating full-model column "
              f"({len(full_col)}) to the variant subset (n={n})")
        full_col = full_col[:n]
    base = float(np.mean(full_col))
    means = [(v, float(np.mean(scores[metric][v])) - base)
             for v in variants]
    means.sort(key=lambda kv: kv[1])
    names = [m.replace(PRIMARY_METHOD + " ", "").strip("()") for m, _ in means]
    # Underscores would render as matplotlib subscripts in a text label; render
    # the physics symbols properly and spell out the rest.
    names = [_pretty_variant(n) for n in names]
    deltas = [d for _, d in means]

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(names) + 1.2))
    colors = ["#c0392b" if d < 0 else "#16a085" for d in deltas]
    ax.barh(names, deltas, color=colors, height=0.62)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"mean {metric} change vs full model")
    ax.set_title(f"Ablation, {dataset} (n={n})", fontsize=9)
    for i, d in enumerate(deltas):
        ax.text(d, i, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7)
    ax.margins(x=0.18)
    fig.tight_layout()
    path = FIG_DIR / f"ablation_{dataset}_{metric}.png"
    from hydro_bench.reporter.plots import save_figure

    save_figure(fig, path, dpi=300, submission_dir=submission_dir)
    plt.close(fig)
    provenance[str(path)] = {
        "source": f"{dataset}__test/results.json", "metric": metric,
        "baseline_mean": base, "variants": dict(means)}
    return path


def fig_meta_train_curve(log_dir: Path,
                         provenance: Dict[str, Any],
                         submission_dir: Optional[Path] = None) -> Optional[Path]:
    curve = load_ablation_curve(log_dir)
    if not curve or not curve.get("history"):
        print("  meta-training curve not found; skipping")
        return None
    hist = curve["history"]
    if len(hist) < 2:
        return None

    from hydro_bench.reporter import plots

    plt = plots._matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs = [h.get("task", i) for i, h in enumerate(hist)]
    ys = [h.get("meta_loss") for h in hist]
    ax.plot(xs, ys, color="#1f4e79", linewidth=1.4)
    ax.set_xlabel("synthetic task")
    ax.set_ylabel("outer-loop meta loss")
    ax.set_title("Meta-training convergence", fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / "meta_train_curve.png"
    from hydro_bench.reporter.plots import save_figure

    save_figure(fig, path, dpi=300, submission_dir=submission_dir)
    plt.close(fig)
    provenance[str(path)] = {"source": str(log_dir / "meta_train_*.json"),
                             "n_points": len(hist)}
    return path


# ------------------------------------------------------------------- tables
def render_tables(results: Any, dataset: str,
                  provenance: Dict[str, Any]) -> List[Path]:
    from hydro_bench.reporter.latex_tables import render_markdown

    out: List[Path] = []
    try:
        md = render_markdown(results)
    except Exception as exc:  # the significance tests need paired columns
        # A length mismatch means the run is incomplete (or a column was scored
        # on a different image set) -- say so rather than producing a table that
        # looks complete next to numbers that are not comparable.
        counts = {m: len(v) for metric, by in results.scores.items()
                  for m, v in by.items()}
        print(f"  {dataset}: table not rendered ({exc}); "
              f"per-method sample counts: {counts}")
        return out
    path = TAB_DIR / f"comparison_{dataset}.md"
    path.write_text(md, encoding="utf-8")
    out.append(path)
    provenance[str(path)] = {
        "source": f"{dataset}__test/results.json",
        "methods": results.methods(),
    }
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", default="experiments/merged_results")
    ap.add_argument("--log-dir", default="experiments/logs")
    ap.add_argument("--fig-dir", default="paper/03_figures")
    ap.add_argument("--table-dir", default="paper/04_tables")
    ap.add_argument("--submission-dir", default="paper/06_submission/figures",
                    help="also write each figure to the publisher's "
                         "artwork formats (vector PDF + 500-dpi TIFF) here")
    args = ap.parse_args()

    merged_root = Path(args.merged_dir)
    global FIG_DIR, TAB_DIR
    FIG_DIR = Path(args.fig_dir)
    TAB_DIR = Path(args.table_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    provenance: Dict[str, Any] = {}
    n_figs = n_tabs = 0

    print("=== figures and tables from merged results ===")
    fig_meta_train_curve(Path(args.log_dir), provenance,
                         submission_dir=Path(args.submission_dir))

    for dataset in DATASETS_PAIRED + DATASETS_UNPAIRED:
        res = load_merged(dataset, merged_root)
        if res is None:
            continue
        print(f"-- {dataset}: {len(res.methods())} methods, "
              f"{len(res.metric_names())} metrics")
        is_paired = dataset in DATASETS_PAIRED
        n_figs += len(fig_boxplots(res, dataset,
                                   FR_METRICS if is_paired else NR_METRICS,
                                   "FR" if is_paired else "NR", provenance,
                                   submission_dir=Path(args.submission_dir)))
        fig_radar(res, dataset,
                  FR_METRICS if is_paired else NR_METRICS, provenance,
                  submission_dir=Path(args.submission_dir))
        n_tabs += len(render_tables(res, dataset, provenance))

    # Ablation figures read the paired set that the ablations ran on.
    for dataset in DATASETS_PAIRED:
        if fig_ablation_bars(merged_root, dataset, provenance,
                             submission_dir=Path(args.submission_dir)) is not None:
            n_figs += 1

    n_figs = sum(1 for p in provenance if str(FIG_DIR) in p)
    manifest = TAB_DIR.parent / "07_provenance" / "figures_provenance.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"generated": str(Path(__file__).resolve()),
         "merged_dir": str(merged_root),
         "artefacts": provenance},
        indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n{len(provenance)} artefact(s) described in {manifest}")
    print(f"figures: {n_figs}   tables: {n_tabs}")
    if not provenance:
        print("nothing generated: no merged results found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
