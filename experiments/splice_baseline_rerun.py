#!/usr/bin/env python3
"""Splice the real-model baseline scores into the committed baseline results.

UIEC2-Net, Five A+ and WaterMamba originally ran as one shared analytic
fallback reported under three names (the checkpoints arrived 2026-09-29, after
the sets were already scored). They were re-scored from their released
checkpoints into ``hydro_bench_results_merged/``, which carries all five sets;
this script merges the three re-scored columns back into the committed baseline
file, replacing the fallback columns in place:

- per-image score vectors (``scores[metric][method]``),
- the per-method provenance (``method_metadata[method]``), and
- the evaluated-method list.

It refuses to splice anything whose image count differs from the target, and it
verifies image-for-image alignment by checking that the methods *not* being
spliced are byte-identical between the two files -- a misaligned vector would
silently corrupt every downstream table.

Usage:
    python3 experiments/splice_baseline_rerun.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "hydro_bench_results"
RERUN_DIR = REPO / "hydro_bench_results_merged"

METHODS = ["UIEC2-Net", "Five-A+", "WaterMamba"]
DATASETS = ["LSUI__test", "EUVP__test", "RUIE__test", "UCCS__test"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _splice(target: Path, rerun: Path, methods: list[str],
            dataset: str) -> list[str]:
    tgt = _load(target)
    new = _load(rerun)

    # scores[metric][method] = per-image list, so the sample count is the
    # length of the first method's vector under the first metric.
    first_metric = next(iter(tgt["scores"].values()))
    n_tgt = len(next(iter(first_metric.values())))

    # Alignment guard: the methods not being replaced must already agree
    # image-for-image between the two files. If they differ, the split or the
    # image order differs and a splice would corrupt the paired tests.
    for metric, vectors in tgt["scores"].items():
        for meth, vec in vectors.items():
            if meth in methods or meth not in new["scores"].get(metric, {}):
                continue
            other = new["scores"][metric][meth]
            if len(other) != len(vec) or not all(
                    abs(a - b) < 1e-12 for a, b in zip(vec, other)):
                raise RuntimeError(
                    f"{dataset}/{metric}/{meth}: the un-spliced columns differ "
                    "between the two files, so the image order or the split "
                    "does not match; refusing to splice")

    changed = []

    for metric, vectors in new["scores"].items():
        tgt_scores = tgt["scores"].setdefault(metric, {})
        for meth in methods:
            if meth not in vectors:
                continue
            vec = vectors[meth]
            if len(vec) != n_tgt:
                raise RuntimeError(
                    f"{dataset}/{metric}/{meth}: rerun has {len(vec)} values, "
                    f"target has {n_tgt}; refusing to splice a mismatched "
                    "vector -- the image order or the split differs")
            tgt_scores[meth] = vec
            if meth not in changed:
                changed.append(meth)

    new_meta = new.get("metadata", {}).get("method_metadata", {})
    tgt_meta = tgt.setdefault("metadata", {}).setdefault("method_metadata", {})
    for meth in methods:
        if meth in new_meta:
            tgt_meta[meth] = new_meta[meth]

    tgt["metadata"]["methods_evaluated"] = sorted(
        set(tgt["metadata"].get("methods_evaluated", [])) | set(methods))

    target.write_text(json.dumps(tgt, indent=2, default=str), encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--rerun-dir", default=str(RERUN_DIR))
    args = ap.parse_args()

    base_root = Path(args.baseline_dir)
    rerun_root = Path(args.rerun_dir)

    for ds in DATASETS:
        rerun = rerun_root / ds / "results.json"
        target = base_root / ds / "results.json"
        if not rerun.is_file():
            print(f"SKIP {ds}: no rerun results at {rerun}")
            continue
        if not target.is_file():
            print(f"SKIP {ds}: no baseline results at {target}")
            continue

        before = _load(target)
        already_real = all(
            not before.get("metadata", {}).get("method_metadata", {})
            .get(m, {}).get("used_fallback", True)
            for m in METHODS)
        if already_real:
            print(f"SKIP {ds}: all three methods already carry real-model "
                  "metadata")
            continue

        changed = _splice(target, rerun, METHODS, ds)
        after = _load(target)
        still_fallback = [m for m in METHODS
                          if after["metadata"]["method_metadata"][m]
                          .get("used_fallback")]
        print(f"{ds}: spliced {len(changed)} methods -> {changed}")
        print(f"  still flagged fallback: {still_fallback or 'none'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
