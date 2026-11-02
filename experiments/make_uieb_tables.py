"""Render the two UIEB tables the general figure script cannot.

``experiments/make_figures.py`` renders one table per dataset straight from the
merged file, and refuses to pair columns of unequal length -- which is the
right instinct. On UIEB that guard fires: the baselines and PhysINR-Zero were
scored on all 134 test images, while every ablation variant was scored on the
first 40 (``--limit 40``, ``experiments/run_physinr_zero.sh``). Mixing them in
one table would either crash the significance test or, worse, compare
unmatched samples.

The two are comparable, just not identical: ``load_split`` takes the front of
the split (``ids[:limit]``, ``dataset_manager.py``), so the first 40 entries of
a full-split column are exactly the images a ``--limit 40`` variant saw. This
script relies on that and on nothing else:

* the main table keeps every method scored on all 134 images;
* the ablation table truncates the full model's column to its first 40 values
  so the paired Wilcoxon tests have matched samples, and states the image
  subset in its caption.

Outputs go next to the other tables (``paper/04_tables/``) and are recorded in
the figures-provenance manifest, which is the mechanism the manuscript uses to
trace a printed number back to the results file behind it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MERGED = REPO / "experiments" / "merged_results" / "UIEB__test" / "results.json"
TAB_DIR = REPO / "paper" / "04_tables"
PROVENANCE = REPO / "paper" / "07_provenance" / "figures_provenance.json"

ABLATION_N = 40
PRIMARY = "PhysINR-Zero"
ABLATION_PREFIX = PRIMARY + " ("


def _is_ablation(method: str) -> bool:
    return method.startswith(ABLATION_PREFIX)


def load() -> Dict[str, Any]:
    if not MERGED.is_file():
        print(f"missing {MERGED}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(MERGED.read_text(encoding="utf-8"))


def _results(scores: Dict[str, Any], order: List[str], metadata: Dict[str, Any]):
    from hydro_bench.reporter.latex_tables import BenchmarkResults

    # The renderer expects the flat method -> provenance mapping, not the
    # whole metadata envelope; passing the envelope makes every "used_fallback"
    # row print unlabelled, which contradicts the manuscript's claim that
    # fallback rows are marked in the tables.
    res = BenchmarkResults(dataset="UIEB", split="test", scores=scores,
                           metadata=metadata.get("method_metadata", {}))
    res.method_order = order
    return res


def truncate(scores: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Restrict every column to its first ``n`` per-image values."""
    return {
        metric: {m: list(v[:n]) for m, v in by_method.items()}
        for metric, by_method in scores.items()
    }


def main() -> int:
    blob = load()
    scores = blob.get("scores", {})
    metadata = blob.get("metadata", {})
    if not scores:
        print("no scores in the merged UIEB file", file=sys.stderr)
        return 1

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    # Artefact keys are repo-relative so the manifest stays stable across
    # machines; make_figures.py records paths the same way.
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    provenance: Dict[str, Any] = {}
    if PROVENANCE.is_file():
        try:
            provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
    artefacts = provenance.setdefault("artefacts", {})

    from hydro_bench.reporter.latex_tables import render_markdown

    # ---- main comparison: methods scored on the full 134-image split only
    main_scores = {m: {k: v for k, v in by.items() if not _is_ablation(k)}
                   for m, by in scores.items()}
    main_order = [m for m in metadata.get("method_order", [])
                  if not _is_ablation(m)] or list(main_scores["psnr"])
    n_main = len(main_scores["psnr"][main_order[0]])
    main_res = _results(main_scores, main_order, metadata)
    md = render_markdown(main_res, baseline=PRIMARY)
    md = (f"<!-- UIEB main comparison, n={n_main} per method. Ablation "
          f"variants are in ablation_UIEB.md, not here: they were scored on a "
          f"40-image subset and are not length-matched to these columns. -->\n"
          + md)
    path = TAB_DIR / "comparison_UIEB.md"
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path} ({len(main_order)} methods, n={n_main})")
    artefacts[rel(path)] = {"source": "experiments/merged_results/UIEB__test/"
                                      "results.json",
                            "methods": main_order, "n_per_method": n_main}

    # ---- ablation: full model truncated to the same first 40 images
    ablation_scores = {m: {k: v[:ABLATION_N] for k, v in by.items()}
                       for m, by in scores.items()}
    ablation_order = [PRIMARY] + [m for m in metadata.get("method_order", [])
                                  if _is_ablation(m)]
    if not any(_is_ablation(m) for m in ablation_order):
        ablation_order = [PRIMARY] + [m for m in ablation_scores["psnr"]
                                      if _is_ablation(m)]
    n_abl = len(ablation_scores["psnr"][PRIMARY])
    abl_res = _results(ablation_scores, ablation_order, metadata)
    md = render_markdown(abl_res, baseline=PRIMARY)
    md = (f"<!-- UIEB ablation, n={n_abl} per method: the first {n_abl} test "
          f"images, the same subset every --limit 40 variant was scored on. "
          f"The full model's column is truncated to that subset so the paired "
          f"Wilcoxon tests (Holm-corrected) compare matched samples; its mean "
          f"here therefore differs from the n={n_main} main table. -->\n" + md)
    path = TAB_DIR / "ablation_UIEB.md"
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path} ({len(ablation_order)} methods, n={n_abl})")
    artefacts[rel(path)] = {
        "source": "experiments/merged_results/UIEB__test/results.json",
        "methods": ablation_order, "n_per_method": n_abl,
        "baseline_truncated_to": n_abl,
        "significance": "paired Wilcoxon signed-rank vs PhysINR-Zero, "
                        "Holm-corrected"}

    # The significance table the manuscript prints alongside the ablation.
    try:
        from hydro_bench.reporter.latex_tables import LatexTableGenerator
        gen = LatexTableGenerator(abl_res, baseline=PRIMARY)
        sig = gen.render_significance()
        path = TAB_DIR / "ablation_UIEB_significance.tex"
        path.write_text(sig, encoding="utf-8")
        print(f"wrote {path}")
        artefacts[rel(path)] = {
            "source": "experiments/merged_results/UIEB__test/results.json",
            "test": "paired Wilcoxon signed-rank, Holm-corrected",
            "baseline": PRIMARY, "n_per_method": n_abl}
    except Exception as exc:  # significance is a refinement, not a blocker
        print(f"significance table not rendered: {exc}")

    provenance["artefacts"] = artefacts
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(provenance, indent=2, sort_keys=True),
                          encoding="utf-8")
    print(f"recorded {len(artefacts)} artefact(s) in {PROVENANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
