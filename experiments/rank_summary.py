"""Derive the every rank claim the manuscript makes, from the merged results.

`paper/02_manuscript/results.md` cites specific ranks and means ("second of
eleven on UCIQE on LSUI", "ranks fifth, within 1.31 of the best"). Those were
written against the pre-rerun tables, where four baselines ran analytic
fallbacks; three of them have since been re-scored as their published models,
so the numbers must be re-derived, not transcribed. This script prints, for
every dataset and metric, the leader and PhysINR-Zero's rank and mean, in the
form the prose uses:

    LSUI  uciqe  HigherBetter  n=120
      rank  method                 mean
      1     Retinex                31.497
      ...
      2     PhysINR-Zero           28.933      gap to leader: 2.56
    PhysINR-Zero rank: 2 of 11; distance to leader: 2.56

Usage:
    python3 experiments/rank_summary.py [--merged-dir experiments/merged_results]
        [--primary PhysINR-Zero]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bench.metrics.statistics import build_metric_comparison  # noqa: E402
from hydro_bench.reporter.latex_tables import METRIC_BETTER  # noqa: E402


PRIMARY_DEFAULT = "PhysINR-Zero"
DATASETS = ["UIEB", "LSUI", "EUVP", "RUIE", "UCCS"]
PAIRED = {"UIEB", "LSUI", "EUVP"}
FR = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg"]
NR = ["uciqe", "uiqm", "uicm", "uism", "uiconm", "ccf"]
# Ablation variants are scored on a 40-image subset and are not part of the
# main comparison; the manuscript ranks "of eleven" among non-variants only.
ABLATION_PREFIX = "PhysINR-Zero ("


def _is_ablation(method: str) -> bool:
    return method.startswith(ABLATION_PREFIX)


def _load(merged_dir: Path, dataset: str) -> Optional[dict]:
    path = merged_dir / f"{dataset}__test" / "results.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ranks_for_metric(scores: Dict[str, List[float]], metric: str,
                     ) -> List[Tuple[int, str, float, int]]:
    """(rank, method, mean, n) best-first, using the same rank computation the
    tables print, so a rank quoted in prose cannot disagree with its table."""
    table = build_metric_comparison(scores, metric=metric,
                                    higher_is_better=METRIC_BETTER.get(metric, True))
    out = []
    for stat in sorted(table.stats,
                       key=lambda s: (s.mean if table.higher_is_better else -s.mean),
                       reverse=table.higher_is_better):
        out.append((table.ranks.get(stat.method, 0), stat.method,
                    stat.mean, stat.n))
    return out


def summarise(merged_dir: Path, primary: str) -> int:
    found = 0
    for dataset in DATASETS:
        blob = _load(merged_dir, dataset)
        if blob is None:
            print(f"{dataset}: no merged results at {merged_dir}/{dataset}__test")
            continue
        found += 1
        scores = blob.get("scores", {})
        meta = (blob.get("metadata", {}) or {}).get("method_metadata", {}) or {}
        metrics = FR + NR if dataset in PAIRED else NR
        print(f"\n{'=' * 72}\n{dataset}\n{'=' * 72}")
        for metric in metrics:
            if metric not in scores:
                continue
            # Main comparison only: ablation variants are scored on a subset
            # and rank separately in the ablation table.
            main_scores = {m: v for m, v in scores[metric].items()
                           if not _is_ablation(m)}
            rows = ranks_for_metric(main_scores, metric)
            higher = METRIC_BETTER.get(metric, True)
            fallbacks = {m for m in main_scores
                         if meta.get(m, {}).get("used_fallback")}
            primary_row = next(((r, m, x, n) for r, m, x, n in rows
                                if m == primary), None)
            n = max((n_ for _, _, _, n_ in rows), default=0)
            print(f"\n{metric}  {'higher-better' if higher else 'lower-better'}"
                  f"  n={n} per method")
            for rank, method, mean, _ in rows:
                mark = " [fallback]" if method in fallbacks else ""
                star = " <--" if method == primary else ""
                print(f"  {rank:>3}  {method:24s} {mean:9.3f}{mark}{star}")
            if primary_row is None:
                print(f"  {primary} absent from this metric")
                continue
            prank, _, pmean, _ = primary_row
            lead = next((x for r, _, x, _ in rows if r == 1), None)
            gap = abs(lead - pmean) if lead is not None else float("nan")
            top = len(rows)
            print(f"  {primary} rank: {prank} of {top}; "
                  f"distance to leader: {gap:.2f}")
    return 0 if found else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", default="experiments/merged_results")
    ap.add_argument("--primary", default=PRIMARY_DEFAULT)
    args = ap.parse_args()
    return summarise(Path(args.merged_dir), args.primary)


if __name__ == "__main__":
    raise SystemExit(main())
