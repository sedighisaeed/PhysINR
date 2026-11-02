"""Splice the re-scored baseline columns into the original run's results.

The three baselines whose published models only became usable after the
original benchmark had already been recorded -- UIEC^2-Net, Five A+ and
WaterMamba -- were re-run into ``hydro_bench_results_rerun/`` (see
``run_rerun.sh``). This script copies those columns over the same methods'
columns in the original run, leaving every other method's scores byte-identical,
and writes the combined file to a new directory that ``merge_results.py`` then
consumes as its baseline spine.

Only the three re-run methods are touched: their per-image values in every
metric, and their provenance entries (``used_fallback``, note, checkpoint, and
WaterMamba's scanner record). Nothing else is invented or recomputed -- the
other seven baselines' numbers are copied through unchanged, which is what
keeps the comparison apples-to-apples with the earlier PhysINR-Zero runs.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent.parent
# Importing the adapters needs the repo root on sys.path; this file is also
# run as a script, where only its own directory would be there.
sys.path.insert(0, str(REPO))
ORIGINAL_DIR = REPO / "hydro_bench_results"
RERUN_DIR = REPO / "hydro_bench_results_rerun"

# The methods run_rerun.sh re-scored. Anything else found in the re-run file is
# reported rather than silently trusted.
RERUN_METHODS = ("Five-A+", "UIEC2-Net", "WaterMamba")


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_meta(method: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a method's run metadata with the adapter's current provenance.

    The re-run spans hours and an adapter's metadata string was revised
    mid-run (WaterMamba's scanner note), so the results files otherwise carry
    two different wordings of the same provenance fact. Overwriting with the
    adapter's present wording keeps one note everywhere; it never touches a
    score, and only keys the adapter actually emits are rewritten.
    """
    out = copy.deepcopy(meta)
    try:
        mod = importlib.import_module(
            f"hydro_bench.methods.deep.{_MODULE_FOR[method]}.adapter")
    except Exception:
        # A method whose adapter no longer imports cleanly keeps its own
        # recorded metadata -- provenance prose degrades, scores never do.
        return out
    for meta_key, constant in _META_CONSTANTS.items():
        if meta_key in out:
            value = getattr(mod, constant, None)
            if isinstance(value, str):
                out[meta_key] = value
    return out


_MODULE_FOR = {"Five-A+": "five_a_plus", "UIEC2-Net": "uiec2_net",
               "WaterMamba": "water_mamba"}
# metadata key -> adapter module constant that produces it.
_META_CONSTANTS = {"scanner": "SCANNER_NOTE", "alpha": "ALPHA"}


def overlay_dataset(dataset: str, original_dir: Path, rerun_dir: Path,
                    out_dir: Path, force: bool) -> bool:
    key = f"{dataset}__test"
    original = _load(original_dir / key / "results.json")
    rerun = _load(rerun_dir / key / "results.json")
    if original is None:
        print(f"  {dataset}: no original run, nothing to overlay")
        return False
    if rerun is None:
        print(f"  {dataset}: no re-run yet (expected at "
              f"{rerun_dir / key / 'results.json'}); skipping")
        return False

    rerun_meta = rerun["metadata"].get("method_metadata", {})
    unexpected = set(rerun_meta) - set(RERUN_METHODS)
    if unexpected:
        print(f"  {dataset}: WARNING re-run also contains {sorted(unexpected)}; "
              "those are not overlaid")

    merged = json.loads(json.dumps(original))  # deep copy, original untouched
    spliced, skipped = [], []
    for metric, by_method in merged.get("scores", {}).items():
        for method in RERUN_METHODS:
            new_values = rerun.get("scores", {}).get(metric, {}).get(method)
            if new_values is None:
                # A metric absent from the re-run (e.g. full-reference metrics
                # on an unpaired set) has nothing to splice.
                continue
            old = by_method.get(method)
            if old is not None and len(old) != len(new_values) and not force:
                skipped.append((metric, method, len(old), len(new_values)))
                continue
            by_method[method] = [float(v) for v in new_values]
            spliced.append((metric, method))

    # Provenance must follow the scores, or a table would mark the new numbers
    # with the old "fallback" label.
    meta = merged["metadata"]
    for method in RERUN_METHODS:
        if method in rerun_meta:
            meta["method_metadata"][method] = _normalise_meta(method, rerun_meta[method])
    meta["rerun_note"] = (
        "UIEC^2-Net, Five A+ and WaterMamba were re-scored with their published "
        "models after that became possible (see run_rerun.sh); their columns "
        "and provenance entries here come from the re-run, every other value "
        "is the original run's. WaterMamba's earlier full-resolution scores "
        "were discarded: its tiled inference had an uncovered edge strip.")

    merged["merge_provenance"] = merged.get("merge_provenance", {})
    merged["merge_provenance"]["rerun"] = str(rerun_dir / key / "results.json")
    merged["merge_provenance"]["original"] = str(original_dir / key / "results.json")

    out = out_dir / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(merged, indent=2),
                                      encoding="utf-8")
    print(f"  {dataset}: spliced {len(spliced)} (metric, method) columns"
          + (f"; skipped {len(skipped)} on length mismatch: {skipped[:3]}"
             if skipped else ""))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["UIEB", "LSUI", "EUVP", "RUIE", "UCCS"])
    ap.add_argument("--original-dir", default=str(ORIGINAL_DIR))
    ap.add_argument("--rerun-dir", default=str(RERUN_DIR))
    ap.add_argument("--out-dir", default="hydro_bench_results_merged",
                    help="directory the overlay writes; pass this to "
                         "merge_results.py as --baseline-dir")
    ap.add_argument("--force", action="store_true",
                    help="splice even where the re-run has a different number "
                         "of per-image values than the original")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    n = 0
    for dataset in args.datasets:
        if overlay_dataset(dataset, Path(args.original_dir), Path(args.rerun_dir),
                           out_dir, args.force):
            n += 1
    if n == 0:
        print("nothing overlaid; run run_rerun.sh first")
        return 1
    print(f"\noverlaid {n} dataset(s) -> {out_dir}\n"
          f"next: python3 experiments/merge_results.py "
          f"--baseline-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
