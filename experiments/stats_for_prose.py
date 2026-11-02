#!/usr/bin/env python3
"""Derive every number the §4.2 prose cites, from the merged results.

Outputs, per dataset and metric:
  * mean ± std per method (main comparison only, no ablation variants)
  * PhysINR-Zero's rank, the leader, and the gap to the leader
  * paired Wilcoxon signed-rank test vs PhysINR-Zero: p, Holm-corrected p,
    and the effect size r = |Z|/sqrt(N) per baseline
  * the bottom-of-column neighbours (closest below / above PhysINR-Zero)

Usage:
    python3 experiments/stats_for_prose.py [--merged-dir experiments/merged_results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRIMARY = "PhysINR-Zero"
ABLATION_PREFIX = PRIMARY + " ("
DATASETS = ["UIEB", "LSUI", "EUVP", "RUIE", "UCCS"]
PAIRED = {"UIEB", "LSUI", "EUVP"}
FR = ["psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg"]
NR = ["uciqe", "uiqm", "uicm", "uism", "uiconm", "ccf"]
DIRECTION = {"lpips_alex": "lower", "lpips_vgg": "lower"}
FALLBACK = "WF-Diff [fallback]"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (returns array in input order)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    running = 0.0
    for i in range(n - 1, -1, -1):
        val = min(1.0, (n - i) * ranked[i])
        running = val if running == 0.0 else min(running, val)
        adjusted[order[i]] = running if running else val
    # enforce monotonicity from the largest down
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return adjusted


def r_effect(diffs):
    """r = |Z| / sqrt(N) with Z from the Wilcoxon normal approximation."""
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs[diffs != 0]
    n = len(nz)
    if n == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    mean = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (max(w_plus, w_minus) - mean) / sd
    return z / np.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", default="experiments/merged_results")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--out", default="experiments/logs/prose_stats.txt")
    args = ap.parse_args()

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    for ds in DATASETS:
        path = Path(args.merged_dir) / f"{ds}__test" / "results.json"
        if not path.is_file():
            say(f"## {ds}: MISSING {path}")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        scores = blob.get("scores", {})
        say(f"\n{'=' * 78}\n{ds}\n{'=' * 78}")
        metrics = FR + NR if ds in PAIRED else NR
        for metric in metrics:
            if metric not in scores:
                continue
            by = {m: v for m, v in scores[metric].items()
                  if not m.startswith(ABLATION_PREFIX)}
            if args.primary not in by:
                continue
            pvals = np.asarray(by[args.primary], dtype=float)
            means = {m: float(np.mean(v)) for m, v in by.items() if len(v)}
            lower_better = DIRECTION.get(metric) == "lower"
            order = sorted(means, key=means.get, reverse=not lower_better)
            prank = order.index(args.primary) + 1
            leader = order[0]
            n_vals = min(len(v) for v in by.values() if len(v))

            say(f"\n--- {metric} ({'lower' if lower_better else 'higher'} better, n={n_vals}) ---")
            for i, m in enumerate(order, 1):
                mark = "  <== PRIMARY" if m == args.primary else ""
                fb = "  [fallback]" if m == FALLBACK else ""
                say(f"  {i:2d}. {m:26s} {means[m]:9.3f}{fb}{mark}")

            say(f"\n  rank {prank}/{len(order)}; leader {leader} "
                f"({means[leader]:.3f}); gap {abs(means[leader] - means[args.primary]):.3f}")

            raw = []
            names = []
            for m, v in by.items():
                if m == args.primary or len(v) != len(pvals):
                    continue
                raw.append(wilcoxon(np.asarray(v, dtype=float), pvals).pvalue)
                names.append((m, np.asarray(v, dtype=float) - pvals))
            adj = holm(raw) if raw else []
            say("\n  paired Wilcoxon vs PhysINR-Zero (raw p, Holm p, r):")
            for (m, diff), p_raw, p_adj in zip(names, raw, adj):
                say(f"    {m:26s} p={p_raw:.2e}  holm={p_adj:.2e}  r={r_effect(diff):+.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
