"""Merge baseline results with PhysINR-Zero and its ablation variants.

HydroBench's runner overwrites ``results.json`` when a run completes, so each
variant is run into its own directory (see ``run_physinr_zero.sh``). This
module reads those per-directory results, gives every variant a distinct
column name, and writes one combined table per dataset that the manuscript's
figures and tables are generated from.

Nothing here invents a number: every value is copied from a results file that
the benchmark itself produced, and the merge records which file each column
came from so a table entry can be traced back to a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "hydro_bench_results"
PHYSINR_DIR = REPO / "experiments" / "physinr_results"
ABLATION_DIR = REPO / "experiments" / "physinr_ablations"

# Human-readable column names for the ablation variants. The directory name is
# the source of truth; this only controls how the column is labelled.
ABLATION_LABELS = {
    "no_phys": "PhysINR-Zero (no physical-model loss)",
    "no_cycle": "PhysINR-Zero (no cycle consistency)",
    "no_color": "PhysINR-Zero (no colour balance)",
    "no_struct": "PhysINR-Zero (no structure loss)",
    "no_spatial": "PhysINR-Zero (no spatial smoothness)",
    "no_jerlov": "PhysINR-Zero (no Jerlov prior)",
    "no_prior": "PhysINR-Zero (no anti-trivial constraints)",
    "no_cast": "PhysINR-Zero (no colour-cast loss)",
    "no_bound": "PhysINR-Zero (no attenuation bound)",
    "stride_1": "PhysINR-Zero (coord stride 1)",
    "stride_3": "PhysINR-Zero (coord stride 3)",
    "steps_5": "PhysINR-Zero (5 adaptation steps)",
    "steps_20": "PhysINR-Zero (20 adaptation steps)",
    "tau_max_2.0": "PhysINR-Zero (tau_max = 2)",
    "tau_max_4.0": "PhysINR-Zero (tau_max = 4)",
    "delta_max_0.5": "PhysINR-Zero (delta_max = 0.5)",
    "delta_max_4.0": "PhysINR-Zero (delta_max = 4)",
}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None


def _merge_scores(target: Dict[str, Any], source: Dict[str, Any],
                  method_from: str, method_to: str) -> int:
    """Copy one method's column from ``source`` into ``target``, renaming it.

    Returns the number of (metric, per-image value) entries copied, so a merge
    that silently added nothing is visible in the summary.
    """
    copied = 0
    for metric, by_method in source.get("scores", {}).items():
        values = by_method.get(method_from)
        if not values:
            continue
        target.setdefault(metric, {})[method_to] = list(map(float, values))
        copied += len(values)
    return copied


def merge_dataset(dataset: str, with_ablations: bool, out_root: Path,
                  baseline_dir: Path, physinr_dir: Path,
                  ablation_dir: Path) -> Optional[Dict[str, Any]]:
    key = f"{dataset}__test"
    baseline = _load(baseline_dir / key / "results.json")
    physinr = _load(physinr_dir / key / "results.json")
    if baseline is None and physinr is None:
        return None

    # The baseline file is the spine: it already carries the baselines, the
    # metric list and the metadata from the original run.
    if baseline is not None:
        merged = dict(baseline)
        merged["scores"] = {
            metric: {method: list(map(float, values))
                     for method, values in by_method.items()}
            for metric, by_method in baseline.get("scores", {}).items()
        }
        provenance = {"baselines": str(baseline_dir / key / "results.json")}
    else:
        merged = {"dataset": dataset, "split": "test", "scores": {},
                  "config": {}, "metadata": {}}
        provenance = {"baselines": None}

    if physinr is not None:
        n = _merge_scores(merged["scores"], physinr, "PhysINR-Zero",
                          "PhysINR-Zero")
        provenance["PhysINR-Zero"] = str(physinr_dir / key / "results.json")
        print(f"  {dataset}: PhysINR-Zero, {n} metric values")

    if with_ablations and ablation_dir.is_dir():
        for variant_dir in sorted(ablation_dir.iterdir()):
            blob = _load(variant_dir / key / "results.json")
            if blob is None:
                continue
            label = ABLATION_LABELS.get(variant_dir.name,
                                        f"PhysINR-Zero ({variant_dir.name})")
            n = _merge_scores(merged["scores"], blob, "PhysINR-Zero", label)
            provenance[label] = str(variant_dir / key / "results.json")
            print(f"  {dataset}: {label}, {n} metric values")

    merged["merge_provenance"] = provenance
    merged["merge_note"] = (
        "Baselines copied unchanged from the original HydroBench run; "
        "PhysINR-Zero and each ablation variant ran in their own output "
        "directories and are merged here. See merge_provenance for the exact "
        "file behind every column.")
    out = out_root / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(merged, indent=2),
                                      encoding="utf-8")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["UIEB", "LSUI", "EUVP", "RUIE", "UCCS"])
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--out-dir", default="experiments/merged_results")
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR),
                    help="Directory holding the original baseline runs.")
    ap.add_argument("--physinr-dir", default=str(PHYSINR_DIR),
                    help="Directory holding the main PhysINR-Zero run.")
    ap.add_argument("--ablation-dir", default=str(ABLATION_DIR),
                    help="Directory holding the ablation variant runs, one "
                         "subdirectory per variant.")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    n = 0
    for dataset in args.datasets:
        if merge_dataset(dataset, not args.no_ablations, out_root,
                         Path(args.baseline_dir), Path(args.physinr_dir),
                         Path(args.ablation_dir)) is not None:
            n += 1
    if n == 0:
        print("nothing to merge: no results found for the requested datasets")
        return 1
    print(f"\nmerged {n} dataset(s) -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
