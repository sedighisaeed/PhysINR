#!/usr/bin/env python3
"""Union several benchmark result trees into one.

The evaluator writes ``results.json`` containing only the methods of the run
that produced it, so a programme that scores methods one invocation at a time
ends up with one tree per method. This folds those trees back together: every
method column is copied into a single output tree, which is what the table and
figure builders consume.

Later inputs win for a given (metric, method) pair, so an explicit re-run of a
method overrides an older column rather than silently coexisting with it.
Every method column is length-checked against the mode of the tree, because a
merge that mixes n=134 with n=3 produces a table that looks complete and is
not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def collect(input_roots: List[Path], dataset_key: str) -> Dict[str, Any]:
    spine: Optional[Dict[str, Any]] = None
    merged_scores: Dict[str, Dict[str, List[float]]] = {}
    sources: Dict[str, str] = {}
    for root in input_roots:
        path = root / f"{dataset_key}/results.json"
        if not path.is_file():
            continue
        blob = load(path)
        if blob is None:
            continue
        if spine is None:
            spine = {k: v for k, v in blob.items() if k != "scores"}
        for metric, by_method in blob.get("scores", {}).items():
            for method, values in by_method.items():
                merged_scores.setdefault(metric, {})[method] = list(map(float, values))
                sources[method] = str(path)
    if spine is None:
        return {}
    spine["scores"] = merged_scores
    spine.setdefault("metadata", {})["merge_sources"] = sources
    return spine


def check_lengths(merged: Dict[str, Any]) -> List[str]:
    """Report methods whose column length differs from the tree's modal length."""
    scores = merged.get("scores", {})
    if not scores:
        return ["no scores merged"]
    lens: Dict[int, int] = {}
    for by_method in scores.values():
        for method, values in by_method.items():
            lens[len(values)] = lens.get(len(values), 0) + 1
    if not lens:
        return []
    modal = max(lens, key=lens.get)
    bad = []
    for metric, by_method in scores.items():
        for method, values in by_method.items():
            if len(values) != modal:
                bad.append(f"{metric}/{method}: n={len(values)} (mode {modal})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="dataset key, e.g. UIEB or UIEB-890")
    ap.add_argument("--split", default="test")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="result-tree roots, each holding <DATASET>__<split>/")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    key = f"{args.dataset}__{args.split}"
    merged = collect([Path(p) for p in args.inputs], key)
    if not merged:
        print(f"nothing merged for {key}", file=sys.stderr)
        return 1
    bad = check_lengths(merged)
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{key}" / "results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    n_methods = len(merged["scores"].get("psnr", merged["scores"]))
    print(f"merged {n_methods} methods into {out_file}")
    for metric, by_method in merged["scores"].items():
        counts = sorted({len(v) for v in by_method.values()})
        print(f"  {metric}: {len(by_method)} methods, n={counts}")
    if bad:
        print("LENGTH MISMATCHES:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
