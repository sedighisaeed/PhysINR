#!/usr/bin/env python3
"""HydroBench CLI: reproducible benchmarking for underwater image restoration.

Subcommands
-----------
verify     Validate a dataset against its manifest (hashes + split disjointness).
run        Run the full evaluation across methods and metrics.
report     Render LaTeX tables and figures from a saved run.
smoke-test Build a synthetic benchmark and exercise the whole pipeline end to end.
manifest   Build a manifest for a dataset tree you have just obtained.
env        Report which optional components are usable in this environment.

Examples
--------
    HYDROBENCH_DATA=/data python run_benchmark.py verify --dataset UIEB
    HYDROBENCH_DATA=/data python run_benchmark.py run --dataset UIEB \
        --methods UDCP Retinex --metrics psnr ssim uciqe uiqm
    python run_benchmark.py smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the package importable whether it is installed or run from the repo.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from hydro_bench.config_defaults import (DEFAULT_FR_METRICS, DEFAULT_NR_METRICS,
                                         DATASETS_CONFIG, METHODS_CONFIG,
                                         DOWNSTREAM_CONFIG, load_configs)
from hydro_bench.core.base_runner import RunnerJob, fresh_workdir
from hydro_bench.core.dataset_manager import DatasetError, make_manager
from hydro_bench.evaluation import BenchmarkRunner, RunConfig
from hydro_bench.reporter.latex_tables import (BenchmarkResults,
                                               LatexTableGenerator,
                                               render_markdown)


# ------------------------------------------------------------------ helpers
def _resolve_env(env_var: str) -> Optional[str]:
    value = os.environ.get(env_var)
    return value or None


def _resolve_placeholder(pathish: str) -> Path:
    """Resolve a ``${ENV:NAME}/...`` config path to a concrete path.

    The shipped configs use a ``${ENV:HYDROBENCH_MODELS}`` placeholder that
    ``os.path.expandvars`` does not understand, so it is resolved here against
    the environment (falling back to the repo's own ``weights/`` directory,
    which is the layout every run script in this repository exports).
    """
    configured = pathish.strip()
    prefix = "${ENV:"
    if prefix not in configured:
        return Path(os.path.expandvars(configured))
    name = configured[len(prefix):].split("}", 1)[0]
    suffix = configured.split("}", 1)[1] if "}" in configured else ""
    base = os.environ.get(name)
    if not base:
        base = str(_HERE.parent / "weights")
    return Path(base) / Path(suffix.lstrip("/"))


def _default_detection_weights() -> Optional[Path]:
    """The frozen detector named in ``downstream.yaml``, if it is on disk."""
    try:
        downstream = load_configs().get("downstream") or {}
    except Exception:
        return None
    configured = (downstream.get("detector") or {}).get("weights")
    if not configured:
        return None
    resolved = _resolve_placeholder(str(configured))
    return resolved if resolved.is_file() else None


def _default_detection_label_dir(dataset: str) -> Optional[Path]:
    """The label directory the dataset's own config declares, if present."""
    try:
        datasets = load_configs().get("datasets") or {}
    except Exception:
        return None
    entry = datasets.get(dataset) or {}
    structure = entry.get("structure") or {}
    configured = structure.get("labels")
    if not configured:
        return None
    root = entry.get("root")
    if not root:
        return None
    resolved = _resolve_placeholder(str(root))
    label_dir = resolved / configured
    return label_dir if label_dir.is_dir() else None


def _configure_mps_memory_pool(args: argparse.Namespace) -> None:
    """Lift Apple's MPS memory ceiling before any weight reaches the GPU.

    Has to happen before the first MPS allocation, so it runs at the very top
    of ``cmd_run`` rather than when an OOM is already in flight: PyTorch reads
    the ratio once, at allocator initialisation. See
    :func:`hydro_bench.methods.deep._vendored.prepare_mps_memory_pool`.
    """
    from hydro_bench.methods.deep._vendored import (MPS_HIGH_WATERMARK_VAR,
                                                   prepare_mps_memory_pool)

    if getattr(args, "cap_mps_memory", False):
        return
    prepare_mps_memory_pool(unbounded=True)
    # Surface the setting: a silent memory-regime change is exactly the kind
    # of thing a reproducibility-focused benchmark must state out loud.
    print(f"MPS memory pool : unbounded ({MPS_HIGH_WATERMARK_VAR}=0) -- "
          f"oversized frames draw on unified memory instead of failing "
          f"(use --cap-mps-memory to keep Apple's default ceiling)",
          file=sys.stderr)


def _print(msg: str) -> None:
    print(msg, file=sys.stderr if False else sys.stdout)


# ------------------------------------------------------------- subcommands
def cmd_verify(args: argparse.Namespace) -> int:
    try:
        manager = make_manager(args.dataset, data_root=args.dataset_root,
                               verify=True, verify_checksums=not args.skip_hashes)
    except DatasetError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 2
    spec = manager.spec
    print(f"dataset      : {spec.name}")
    print(f"root         : {spec.root}")
    print(f"type         : {spec.type}")
    print(f"description  : {spec.description}")
    print(f"manifest     : {spec.manifest_path}")
    for split in manager.splits():
        ids = manager.sample_ids(split)
        print(f"  split {split:8s}: {len(ids)} samples")
        if args.show_samples:
            print(f"    e.g. {ids[:3]}")
    print("checksums    : verified" if not args.skip_hashes else "checksums    : skipped")
    print("RESULT       : OK - dataset matches its manifest")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    from hydro_bench.metrics import full_reference as fr
    from hydro_bench.metrics import no_reference as nr

    print("HydroBench environment report")
    print("  full-reference metrics:")
    for name, ok in fr.available_metrics().items():
        print(f"    {name:12s} {'available' if ok else 'MISSING (installable via pip)'}")
    print("  no-reference metrics:")
    for name, ok in nr.available_nr_metrics().items():
        print(f"    {name:12s} {'available' if ok else 'MISSING (needs model weights)'}")
    for component, module in (("ultralytics (detection)", "ultralytics"),
                              ("lpips", "lpips"),
                              ("torch", "torch")):
        try:
            __import__(module)
            print(f"  {component}: available")
        except ImportError:
            print(f"  {component}: MISSING")
    for tool in ("conda", "docker", "nvidia-smi"):
        print(f"  runner '{tool}': "
              f"{'found' if shutil.which(tool) else 'not found'}")
    return 0


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    metrics = args.metrics or []
    fr_metrics = [m for m in metrics if m in DEFAULT_FR_METRICS]
    nr_metrics = [m for m in metrics if m in DEFAULT_NR_METRICS]
    unknown = [m for m in metrics
               if m not in DEFAULT_FR_METRICS and m not in DEFAULT_NR_METRICS]
    if unknown:
        print(f"WARNING: ignoring unrecognised metrics: {unknown}", file=sys.stderr)
    if not fr_metrics and not nr_metrics and not args.keypoints:
        fr_metrics, nr_metrics = list(DEFAULT_FR_METRICS), list(DEFAULT_NR_METRICS)

    offline_dirs: Dict[str, Path] = {}
    for pair in args.offline_dir or []:
        name, _, path = pair.partition("=")
        offline_dirs[name] = Path(path)

    # Per-method parameters declared in configs/methods.yaml (checkpoint names
    # and the like) are the defaults; an explicit --params on the command line
    # takes precedence over them.
    method_params: Dict[str, Dict[str, Any]] = {}
    try:
        methods_cfg = load_configs().get("methods", {})
    except Exception:  # a missing/broken config must not block the run
        methods_cfg = {}
    for method in args.methods or []:
        declared = methods_cfg.get(method, {}).get("params") or {}
        if declared:
            method_params[method] = dict(declared)
    for method, params in _parse_params(args.params).items():
        method_params[method] = {**method_params.get(method, {}), **params}

    # --detection alone means "run the downstream mAP pass with the frozen
    # detector this benchmark declares", so the weights and label directory
    # default from the configs rather than demanding both be restated on the
    # command line. An explicit argument always wins; a bare --detection whose
    # configured weights are absent is reported loudly, not silently skipped.
    detection_requested = bool(getattr(args, "detection", False))
    detection_weights = (Path(args.detection_weights)
                         if getattr(args, "detection_weights", None) else None)
    detection_label_dir = (Path(args.detection_label_dir)
                           if getattr(args, "detection_label_dir", None)
                           else None)
    if detection_requested and detection_weights is None:
        detection_weights = _default_detection_weights()
        if detection_weights is None:
            print("WARNING: --detection given but the detector weights named in "
                  "downstream.yaml are not on disk; the mAP pass will be skipped",
                  file=sys.stderr)
    if detection_requested and detection_label_dir is None:
        detection_label_dir = _default_detection_label_dir(args.dataset)

    return RunConfig(
        dataset=args.dataset,
        split=args.split,
        methods=list(args.methods or []),
        fr_metrics=fr_metrics,
        nr_metrics=nr_metrics,
        keypoint_detectors=list(args.keypoints or []),
        detection=bool(args.detection),
        segmentation=bool(args.segmentation),
        limit=args.limit,
        baseline=args.baseline,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        offline_dirs=offline_dirs,
        method_params=method_params,
        verify_checksums=not args.skip_hashes,
        show_progress=not args.quiet,
        resume=bool(args.resume),
        cap_mps_memory=bool(getattr(args, "cap_mps_memory", False)),
        save_images=bool(getattr(args, "save_images", False)),
        detection_weights=detection_weights,
        detection_imgsz=int(getattr(args, "detection_imgsz", 640)),
        detection_label_dir=detection_label_dir,
    )


def _parse_params(pairs: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs or []:
        method, _, rest = pair.partition(":")
        params: Dict[str, Any] = {}
        for item in rest.split(","):
            if not item:
                continue
            key, _, value = item.partition("=")
            params[key] = _coerce(value)
        out[method] = params
    return out


def _coerce(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def cmd_run(args: argparse.Namespace) -> int:
    _configure_mps_memory_pool(args)
    config = _build_run_config(args)
    runner = BenchmarkRunner(config=config)
    try:
        results = runner.run()
    except RuntimeError as exc:
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        return 3
    _emit_reports(args, results, runner)
    if runner.warnings:
        print(f"\n{len(runner.warnings)} warnings recorded "
              f"(see results.json):", file=sys.stderr)
        for warning in runner.warnings[:10]:
            print(f"  - {warning}", file=sys.stderr)
    return 0


def _emit_reports(args: argparse.Namespace, results: BenchmarkResults,
                  runner: BenchmarkRunner) -> None:
    out = config_out_root(args)
    generator = LatexTableGenerator(results=results,
                                    baseline=args.baseline,
                                    decimals=args.decimals)
    latex_path = generator.write(out / "results_table.tex",
                                 caption=(f"Quantitative comparison on "
                                          f"{results.dataset}/{results.split} "
                                          f"({results.metric_names()})"),
                                 label=f"tab:{results.dataset}")
    print(f"latex table  : {latex_path}")
    md = render_markdown(results, baseline=args.baseline)
    (out / "results_table.md").write_text(md, encoding="utf-8")
    print(f"markdown     : {out / 'results_table.md'}")
    print(_table_preview(md))

    from hydro_bench.reporter import plots

    for metric in results.metric_names():
        try:
            plots.boxplot(results, metric, out / f"boxplot_{metric}.png")
        except (KeyError, ValueError) as exc:
            print(f"  skipped boxplot for {metric}: {exc}", file=sys.stderr)
    if len(results.metric_names()) >= 3:
        try:
            plots.radar_chart(results, results.metric_names(),
                              out / "radar.png")
        except (KeyError, ValueError) as exc:
            print(f"  skipped radar: {exc}", file=sys.stderr)
    if args.baseline:
        for metric in ("psnr", "ssim", "uciqe", "uiqm"):
            if metric in results.scores and args.baseline in results.scores[metric]:
                try:
                    plots.delta_curve(results, metric,
                                      out / f"delta_{metric}.png",
                                      baseline=args.baseline)
                except (KeyError, ValueError) as exc:
                    print(f"  skipped delta curve {metric}: {exc}",
                          file=sys.stderr)
                break


def config_out_root(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) / f"{args.dataset}__{args.split}"


def _table_preview(md: str, rows: int = 6) -> str:
    lines = md.strip().splitlines()[:rows]
    return "\n".join(f"  {line}" for line in lines)


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.results_json)
    if not path.is_file():
        print(f"results file not found: {path}", file=sys.stderr)
        return 4
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = BenchmarkResults(dataset=payload["dataset"],
                               split=payload["split"],
                               scores=payload["scores"])
    results.method_order = payload["metadata"].get("methods_evaluated", [])
    # Restore per-method provenance so a re-rendered table still marks the
    # rows that ran a fallback instead of the published model.
    results.metadata = payload["metadata"].get("method_metadata", {})
    args.methods = results.method_order
    # The output directory is derived from the run's own dataset/split, so a
    # re-render lands beside the run it came from rather than under the
    # CLI placeholder defaults.
    args.dataset = results.dataset
    args.split = results.split
    _emit_reports(args, results, None)
    return 0


def _list_image_ids(directory: Path) -> List[str]:
    """Sample ids of every image in *directory*, sorted for determinism."""
    return sorted(p.name for p in directory.iterdir()
                  if p.suffix.lower() in _IMAGE_SUFFIXES)


def _parse_split_specs(specs: List[str], root: Path, raw_subdir: str,
                       seed: int) -> Dict[str, List[str]]:
    """Resolve ``NAME=ALL|COUNT|FRACTION|FILE`` specs into split lists.

    A *fraction* (e.g. ``train=0.7``) means "this share of the dataset". The
    fractions are applied to a single seeded permutation of the full id list,
    one partition after another, so the resulting splits are disjoint by
    construction -- resolving each fraction against the unshuffled list, or
    shuffling per split, would let the same sample land in two splits and the
    manifest's leakage check would only catch it afterwards.

    Splits given as ``ALL``, a count, or an explicit id file are taken as-is
    and may be mixed with fractions only when they name disjoint sets; the
    manifest's disjointness check rejects anything that actually overlaps.
    """
    parsed = []
    for pair in specs:
        name, _, spec = pair.partition("=")
        if not name:
            raise ValueError(f"malformed split spec '{pair}' (expected NAME=...)")
        if spec == "ALL":
            kind = "all"
            value: Any = None
        elif spec.isdigit():
            kind = "count"
            value = int(spec)
        else:
            try:
                fraction = float(spec)
            except ValueError:
                kind = "file"
                value = spec
            else:
                if not 0.0 < fraction < 1.0:
                    raise ValueError(
                        f"split fraction '{pair}' is out of range; use a value "
                        "strictly between 0 and 1 (or COUNT, or ALL)")
                kind = "fraction"
                value = fraction
        parsed.append((name, kind, value))

    all_ids = _list_image_ids(root / raw_subdir)
    if not all_ids:
        raise ValueError(f"no images found under {root / raw_subdir}")

    fractions = [(n, v) for n, k, v in parsed if k == "fraction"]
    splits: Dict[str, List[str]] = {}

    if fractions:
        # One permutation, partitioned in declaration order. The final
        # fractional split absorbs the rounding remainder so every id lands
        # somewhere.
        import numpy as np

        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(all_ids)))
        shuffled = [all_ids[i] for i in order]
        start = 0
        for index, (name, fraction) in enumerate(fractions):
            if index == len(fractions) - 1:
                ids = shuffled[start:]
            else:
                # Truncate, matching the protocol used for the UIEB manifest
                # (890 images -> 623/133/134); the last split takes whatever
                # rounding leaves behind, so no id is dropped.
                count = int(fraction * len(all_ids))
                ids = shuffled[start:start + count]
                start += count
            splits[name] = sorted(ids)

    for name, kind, value in parsed:
        if kind == "fraction":
            continue  # already partitioned above
        if kind == "all":
            splits[name] = list(all_ids)
        elif kind == "count":
            splits[name] = list(all_ids[:value])
        else:
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"split id file not found: {path}")
            splits[name] = [line.strip() for line in path.read_text().splitlines()
                            if line.strip()]
    return splits


def cmd_manifest(args: argparse.Namespace) -> int:
    from hydro_bench.core.manifest import build_manifest

    root = Path(args.root)
    if not root.is_dir():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 5
    try:
        splits = _parse_split_specs(args.splits or ["test=ALL"], root,
                                    args.raw_subdir, args.seed)
    except ValueError as exc:
        print(f"cannot build manifest: {exc}", file=sys.stderr)
        return 5
    if not splits:
        print("no samples found to hash", file=sys.stderr)
        return 5
    out_path = Path(args.out) if args.out else Path("datasets/manifests") / f"{args.dataset.lower()}.json"
    manifest = build_manifest(dataset=args.dataset, root=root, splits=splits,
                              raw_subdir=args.raw_subdir,
                              gt_subdir=args.gt_subdir or None,
                              out_path=out_path)
    print(f"manifest written: {out_path}")
    print(f"  {len(manifest.files)} files hashed, splits: "
          f"{({k: len(v) for k, v in manifest.splits.items()})}")
    print("  commit this file to version control to pin the dataset")
    return 0


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """End-to-end test on a synthetic dataset: proves the pipeline is sound."""
    import subprocess

    data_root = Path(args.data_root or "/tmp/hydro_data")
    script = _HERE / "scripts" / "make_synthetic_benchmark.py"
    print(f"[smoke] generating synthetic benchmark in {data_root}")
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(data_root),
         "--n-test", str(args.n_test), "--n-train", str(args.n_train),
         "--n-val", str(args.n_val), "--width", str(args.width),
         "--height", str(args.height)],
        check=False)
    if result.returncode != 0:
        print("[smoke] dataset generation failed", file=sys.stderr)
        return 6
    os.environ["HYDROBENCH_DATA"] = str(data_root)

    manifest_dir = data_root / "SynthUIB" / "manifests"
    target = _HERE / "datasets" / "manifests" / "synthuib.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_dir / "synthuib.json", target)

    print("[smoke] verifying dataset against its manifest")
    verify_args = argparse.Namespace(
        dataset="SynthUIB", dataset_root=None, skip_hashes=False,
        show_samples=False)
    if cmd_verify(verify_args) != 0:
        print("[smoke] verification failed", file=sys.stderr)
        return 2

    print("[smoke] running the full evaluation")
    run_args = argparse.Namespace(
        dataset="SynthUIB", split="test", methods=["UDCP", "Retinex"],
        metrics=["psnr", "ssim", "uciqe", "uiqm", "ccf"],
        keypoints=[], detection=False, segmentation=False,
        limit=args.limit, baseline="UDCP", out_dir=args.out_dir,
        seed=0, offline_dir=[], params=[], skip_hashes=False, quiet=False,
        decimals=2)
    code = cmd_run(run_args)
    if code != 0:
        print("[smoke] evaluation failed", file=sys.stderr)
        return code
    print("[smoke] OK - the full pipeline ran end to end")
    return 0


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydro-bench",
        description="Reproducible benchmarking for underwater image restoration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_verify = sub.add_parser("verify", help="Validate a dataset manifest.")
    p_verify.add_argument("--dataset", required=True)
    p_verify.add_argument("--dataset-root", default=None)
    p_verify.add_argument("--skip-hashes", action="store_true")
    p_verify.add_argument("--show-samples", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_env = sub.add_parser("env", help="Report usable components.")
    p_env.set_defaults(func=cmd_env)

    p_run = sub.add_parser("run", help="Run the evaluation.")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--split", default="test")
    p_run.add_argument("--methods", nargs="+", default=None)
    p_run.add_argument("--metrics", nargs="+", default=None,
                       help="subset of: psnr ssim ms_ssim lpips_alex lpips_vgg "
                            "uciqe uiqm uicm uism uiconm ccf brisque musiq")
    p_run.add_argument("--keypoints", nargs="+", choices=["SIFT", "ORB"],
                       default=None)
    p_run.add_argument("--detection", action="store_true")
    p_run.add_argument("--detection-weights", default=None,
                       help="frozen detector checkpoint for the downstream "
                            "mAP pass (train on the dataset's train split)")
    p_run.add_argument("--detection-label-dir", default=None,
                       help="YOLO-format label directory for the split's "
                            "ground truth")
    p_run.add_argument("--detection-imgsz", type=int, default=640)
    p_run.add_argument("--segmentation", action="store_true")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--baseline", default=None)
    p_run.add_argument("--out-dir", default="hydro_bench_results")
    p_run.add_argument("--offline-dir", nargs="*", default=[],
                       metavar="NAME=PATH",
                       help="pre-computed results for one method (Mode B)")
    p_run.add_argument("--params", nargs="*", default=[],
                       metavar="METHOD:key=val,key=val")
    p_run.add_argument("--skip-hashes", action="store_true")
    p_run.add_argument("--decimals", type=int, default=2)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted run from its per-image checkpoint "
             "(out_dir/<dataset>__<split>/partial.json) instead of restarting")
    p_run.add_argument(
        "--cap-mps-memory", action="store_true",
        help="keep Apple's default MPS memory ceiling instead of lifting it; "
             "frames that exceed the pool then fall back to CPU")
    p_run.add_argument(
        "--save-images", action="store_true",
        help="write each method's enhanced output to "
             "out_dir/<dataset>__<split>/images/<method>/<id>.png for the "
             "manuscript's qualitative comparison grid")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Render tables/figures from a run.")
    p_report.add_argument("--results-json", required=True)
    p_report.add_argument("--baseline", default=None)
    p_report.add_argument("--out-dir", default="hydro_bench_results")
    p_report.add_argument("--decimals", type=int, default=2)
    p_report.add_argument("--dataset", default="dataset")
    p_report.add_argument("--split", default="test")
    p_report.set_defaults(func=cmd_report)

    p_manifest = sub.add_parser("manifest", help="Build a dataset manifest.")
    p_manifest.add_argument("--dataset", required=True)
    p_manifest.add_argument("--root", required=True)
    p_manifest.add_argument("--raw-subdir", default="raw")
    p_manifest.add_argument("--gt-subdir", default=None)
    p_manifest.add_argument("--splits", nargs="+",
                            default=["train=0.7", "val=0.15", "test=0.15"],
                            help="NAME=ALL | NAME=<count> | NAME=<fraction> "
                                 "| NAME=<file with ids>. Fractions are taken "
                                 "from one seeded permutation so the splits "
                                 "are disjoint")
    p_manifest.add_argument("--seed", type=int, default=20260926,
                            help="seed for the fractional split permutation")
    p_manifest.add_argument("--out", default=None)
    p_manifest.set_defaults(func=cmd_manifest)

    p_smoke = sub.add_parser("smoke-test",
                             help="Full pipeline test on synthetic data.")
    p_smoke.add_argument("--data-root", default=None)
    p_smoke.add_argument("--out-dir", default="hydro_bench_results")
    p_smoke.add_argument("--n-train", type=int, default=12)
    p_smoke.add_argument("--n-val", type=int, default=4)
    p_smoke.add_argument("--n-test", type=int, default=8)
    p_smoke.add_argument("--width", type=int, default=96)
    p_smoke.add_argument("--height", type=int, default=72)
    p_smoke.add_argument("--limit", type=int, default=None)
    p_smoke.set_defaults(func=cmd_smoke_test)

    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
