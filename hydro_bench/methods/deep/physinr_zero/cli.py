"""Command-line entry point: ``python -m hydro_bench.methods.deep.physinr_zero <command>``.

Subcommands
-----------
meta-train   Build the meta-initialisation from synthetic tasks only.
enhance      Apply test-time adaptation to one image or a directory.
compare      Score a pre/post-refactoring pair of checkpoints on the same set.
info         Report the parameter budget and the resolved configuration.
smoke        A fast end-to-end check that the pipeline runs and inverts.

``--legacy`` (on meta-train and enhance) switches the configuration back to the
pre-refactoring objective and physics, which is how the published numbers are
reproduced and how a before/after comparison isolates the colour-cast refactor
from an unrelated regression.

Every command writes a provenance record next to its outputs, because a
manuscript claim is only as good as the configuration that produced it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .config import PhysINRConfig
from .model import PhysINR

# Four levels below the repository root (hydro_bench/methods/deep/this_pkg/).
REPO = Path(__file__).resolve().parents[4]

# Image extensions accepted by the directory-walking commands.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------------ helpers --
def _config_from_args(args) -> PhysINRConfig:
    """Build the configuration, honouring ``--legacy`` and the scalar overrides.

    ``--legacy`` selects the pre-refactoring objective and physics so the
    published numbers stay reproducible; the scalar overrides (tasks, seed,
    device) are applied on top of whichever base was chosen.
    """
    overrides: Dict[str, object] = {}
    if getattr(args, "tasks", None) is not None:
        overrides["meta_tasks"] = args.tasks
    if getattr(args, "meta_batch", None) is not None:
        overrides["meta_batch"] = args.meta_batch
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "device", None) is not None:
        overrides["device"] = args.device
    base = PhysINRConfig.legacy() if getattr(args, "legacy", False) \
        else PhysINRConfig()
    return replace(base, **overrides) if overrides else base


def _checkpoint_config(ckpt: Path, legacy: bool) -> PhysINRConfig:
    """The configuration a checkpoint was trained under, with legacy overrides.

    The configuration travels inside the checkpoint blob; rebuilding it from
    the current defaults would pair a trained state with hyperparameters no
    run ever described -- and after the type-head fix, an architecture whose
    weights cannot even load.  ``legacy`` still forces the pre-refactoring
    fields on top, so the published checkpoint reproduces its published
    behaviour.
    """
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(PhysINRConfig)}
    cfg = PhysINRConfig(**{k: v for k, v in blob.get("config", {}).items()
                           if k in known})
    if legacy:
        lg = PhysINRConfig.legacy()
        cfg = replace(cfg, **{k: getattr(lg, k) for k in
                              ("use_spectral_beta", "ambient_adaptive",
                               "yiq_align", "w_cast", "w_bound",
                               "type_head_features", "w_type_sup",
                               "t_min", "tau_data_weight",
                               "tau_jerlov_centered", "level_head",
                               "w_level_sup")})
    return cfg


def _cast_report(image: np.ndarray) -> Dict[str, float]:
    """Colour-cast diagnostics of one HxWx3 image in [0, 1].

    The cast this refactor targets is a *global* chroma bias, so the
    diagnostics are the CIELAB chroma means -- a* < 0 is green, b* < 0 is
    blue, and a green/cyan cast drives both negative -- together with their
    magnitude, and the YIQ chroma means the alignment stage acts on.  The
    magnitude is the headline number: it is what "the cast got better" means.
    """
    import torch

    from . import physics

    x = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
    lab = physics.rgb_to_lab(x)
    mu_a = float(lab[..., 1].mean())
    mu_b = float(lab[..., 2].mean())
    mu, _sigma = physics.yiq_chroma_stats(x)
    return {
        "lab_mu_a": round(mu_a, 4),
        "lab_mu_b": round(mu_b, 4),
        "lab_cast_magnitude": round(math.hypot(mu_a, mu_b), 4),
        "yiq_mu_i": round(float(mu[0, 0]), 5),
        "yiq_mu_q": round(float(mu[0, 1]), 5),
    }


def _find_reference(input_path: Path, reference_dir: Path,
                    input_root: Path) -> Optional[Path]:
    """Locate the ground-truth image matching an input, or None."""
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        cand = reference_dir / (input_path.stem + ext)
        if cand.is_file():
            return cand
    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        return None
    cand = reference_dir / rel
    return cand if cand.is_file() else None


def _dataset_test_images(dataset: str, split: str = "test") -> List[Path]:
    """Test-split image paths of a HydroBench dataset, from its manifest.

    Used by the deduplication pass so that the meta-training source cannot
    contain content that later appears in an evaluation split.
    """
    manifest = REPO / "hydro_bench" / "datasets" / "manifests" / f"{dataset.lower()}.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no manifest for {dataset}: {manifest}")
    with open(manifest, encoding="utf-8") as fh:
        blob = json.load(fh)
    ids = blob.get("splits", {}).get(split, [])
    root = blob.get("root")
    if root:
        root = Path(root)
        if not root.is_absolute():
            root = (REPO / "datasets" / dataset).resolve()
    else:
        root = (REPO / "datasets" / dataset).resolve()

    files = blob.get("files", {})
    if isinstance(files, dict) and files:
        # Manifest maps dataset-relative paths to hashes.
        out = []
        for rel in files:
            p = Path(rel)
            if not p.is_absolute():
                p = root / p
            if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".bmp", ".jpeg"}:
                out.append(p)
        if out:
            return out

    # Fall back to enumerating the input directory and matching by stem.
    structure = {"UIEB": "raw_input_imgs", "EUVP": "train_val/underwater_dark",
                 "LSUI": "input", "RUIE": "RUIE_raw", "UCCS": "color_cast"}
    sub = structure.get(dataset)
    if sub is None:
        raise KeyError(f"unknown input layout for dataset {dataset}")
    src = root / sub
    stems = {Path(i).stem for i in ids}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in src.rglob("*")
                  if p.suffix.lower() in exts and p.stem in stems)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hydro_bench.methods.deep.physinr_zero",
        description="PhysINR-Zero: physics-constrained implicit neural field "
                    "for zero-shot underwater image enhancement.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    # ---- meta-train -------------------------------------------------------
    mt = sub.add_parser("meta-train", help="Train the meta-initialisation.")
    mt.add_argument("--source", required=True,
                    help="Clean-image directory for synthetic task generation "
                         "(a path relative to the repo root, e.g. "
                         "datasets/clean_photos). Must NOT be inside a "
                         "HydroBench evaluation dataset: the paper's "
                         "zero-shot claim requires that no real underwater "
                         "image enter meta-training, and a benchmark "
                         "dataset's reference images are real underwater "
                         "photographs. The command refuses such a source "
                         "unless --allow-eval-source is also given.")
    mt.add_argument("--allow-eval-source", action="store_true",
                    help="Override the refusal of a source inside an "
                         "evaluation dataset. Recorded in the provenance, "
                         "and the zero-shot claim must then be dropped.")
    mt.add_argument("--out", default="weights/physinr_zero_meta.pt")
    mt.add_argument("--tasks", type=int, default=None)
    mt.add_argument("--meta-batch", type=int, default=None)
    mt.add_argument("--seed", type=int, default=None)
    mt.add_argument("--device", default=None)
    mt.add_argument("--no-dedup", action="store_true",
                    help="Skip the duplicate check against evaluation test "
                         "splits. Only for experiments that never evaluate "
                         "on the affected datasets.")
    mt.add_argument("--legacy", action="store_true",
                    help="Meta-train with the pre-refactoring objective and "
                         "physics (no spectral parameterisation, no adaptive "
                         "background light, no cast losses, no CDOM). Use this "
                         "to reproduce the published checkpoint.")
    mt.add_argument("--log-dir", default="experiments/logs")

    # ---- enhance ----------------------------------------------------------
    en = sub.add_parser("enhance", help="Enhance images by test-time adaptation.")
    en.add_argument("input", help="Single image file, or a directory of images.")
    en.add_argument("--output", required=True,
                    help="Output image file, or output directory.")
    en.add_argument("--checkpoint", default="weights/physinr_zero_meta.pt")
    en.add_argument("--steps", type=int, default=None)
    en.add_argument("--seed", type=int, default=None)
    en.add_argument("--device", default=None)
    en.add_argument("--legacy", action="store_true",
                    help="Run the checkpoint under the pre-refactoring "
                         "configuration -- the objective and physics the "
                         "published numbers were produced with. A "
                         "pre-refactoring checkpoint run this way reproduces "
                         "its published outputs.")
    en.add_argument("--trace", action="store_true",
                    help="Also write a per-image adaptation trace as JSON.")
    en.add_argument("--limit", type=int, default=None)
    en.add_argument("--log-dir", default="experiments/logs")

    # ---- compare ----------------------------------------------------------
    cm = sub.add_parser(
        "compare",
        help="Score a pre/post-refactoring checkpoint pair on the same images.")
    cm.add_argument("input",
                    help="Single image file, or a directory of images.")
    cm.add_argument("--legacy-checkpoint", required=True,
                    help="Pre-refactoring checkpoint, run in legacy mode.")
    cm.add_argument("--refactored-checkpoint", required=True,
                    help="Post-refactoring checkpoint, run with the current "
                         "configuration.")
    cm.add_argument("--reference", default=None,
                    help="Optional directory of reference (ground-truth) "
                         "images. Where a matching file is found, PSNR and "
                         "SSIM are reported alongside the no-reference "
                         "metrics.")
    cm.add_argument("--out", default="experiments/cast_comparison.json",
                    help="JSON report to write.")
    cm.add_argument("--steps", type=int, default=None)
    cm.add_argument("--seed", type=int, default=None)
    cm.add_argument("--device", default=None)
    cm.add_argument("--limit", type=int, default=None)
    cm.add_argument("--log-dir", default="experiments/logs")

    # ---- info --------------------------------------------------------------
    sub.add_parser("info", help="Show parameter count and configuration.")

    # ---- smoke -------------------------------------------------------------
    sm = sub.add_parser("smoke", help="Fast end-to-end pipeline check.")
    sm.add_argument("--checkpoint", default="weights/physinr_zero_meta.pt")
    sm.add_argument("--device", default=None)
    return p


# --------------------------------------------------------------- commands ----
def cmd_info(args) -> int:
    cfg = PhysINRConfig()
    model = PhysINR(cfg)
    print(f"PhysINR-Zero {__version__}")
    print(f"trainable parameters : {model.count_parameters()} "
          f"(budget {cfg.param_budget})")
    print(f"water types          : {cfg.num_water_types}")
    print(f"fourier levels L     : {cfg.fourier_levels}")
    print(f"adaptation steps     : {cfg.adapt_steps}")
    print(f"lr field / extractor : {cfg.lr_field} / {cfg.lr_extractor}")
    groups = model.parameter_groups()
    print(f"field params         : {sum(p.numel() for p in groups[0]['params'])}")
    print(f"extractor params     : {sum(p.numel() for p in groups[1]['params'])}")
    return 0


def cmd_smoke(args) -> int:
    """Verify the pipeline runs and the physics inverts.

    The check is a *unit* property, not a benchmark: degrading a synthetic
    clean image with a known transmission and recovering it through the
    constrained inverse should reproduce the input to within the ambient-light
    term.  If this fails, the physics implementation is wrong, independent of
    any learned behaviour.
    """
    import torch

    from . import physics
    from .config import PhysINRConfig
    from .model import PhysINR

    cfg = PhysINRConfig()
    model = PhysINR(cfg)
    device = args.device or ("cpu")  # smoke test stays on CPU for determinism
    model = model.to(device)

    torch.manual_seed(0)
    h = w = 64
    img = torch.rand(1, 3, h, w, device=device)
    out = model(img, step=0)

    # Algebraic identity of §8/§9: regrade . physical_inverse = I, up to the
    # eps guard in the transmission denominator.  Tested at moderate
    # transmission, where eps is negligible; at very low transmission eps
    # dominates by design, which is what keeps the inverse finite.
    img_cl = img.permute(0, 2, 3, 1)
    t = (0.5 + 0.5 * torch.rand_like(out.transmission)).clamp(0.5, 1.0)
    a = out.ambient.detach()
    raw = physics.physical_inverse(img_cl, a, t, cfg.eps)
    roundtrip = physics.regrade(raw, t, a)
    err = (roundtrip - img_cl).abs().max().item()

    # The constrained pipeline applies a clamp and the bounded nonlinear
    # correction, so it is checked for *bounds*, not exact inversion.
    tau_ok = bool((out.tau >= 0).all() and (out.tau <= cfg.tau_max).all())
    t_ok = bool((out.transmission > 0).all() and (out.transmission <= 1).all())
    a_ok = bool((out.ambient >= cfg.ambient_min - cfg.kappa_ambient).all()
                and (out.ambient <= cfg.ambient_max + cfg.kappa_ambient).all())
    w_ok = bool(abs(out.water_weights.sum(-1).item() - 1.0) < 1e-4)
    shape_ok = out.restored.shape == img_cl.shape
    restored_ok = bool((out.restored >= 0).all() and (out.restored <= 1).all())

    # ---- coordinate subsampling (§21 step 4a) ----------------------------
    # A strided forward must be (almost) the *same* field evaluated on a
    # subset of the pixels: the per-pixel parts are indexed from the full-grid
    # feature field, so they match exactly, while the two global poolings
    # (the water-type logit and the ambient offset) are estimated from the
    # sampled pixels instead of all of them.  The residual is therefore a
    # sampling error on a global statistic, bounded well away from anything
    # that could change the solution qualitatively.
    stride = 4
    out_s = model(img, step=0, coord_stride=stride)
    hs = out_s.tau.shape[1]
    ws = out_s.tau.shape[2]
    same_pixels = (out.tau[:, ::stride, ::stride] - out_s.tau[:, :hs, :ws])
    stride_err = same_pixels.abs().max().item()
    stride_ok = stride_err < 2e-3           # ~6e-4 relative on a tau range of 8
    # Odd sizes: ceil(H/stride) must line up with the pixel slice.
    img_odd = torch.rand(1, 3, 63, 67, device=device)
    out_odd = model(img_odd, step=0, coord_stride=2)
    odd_ok = (out_odd.tau.shape[1] == 32 and out_odd.tau.shape[2] == 34
              and out_odd.restored.shape[1] == 32)

    checks = {
        "tau in [0, tau_max]": tau_ok,
        "transmission in (0, 1]": t_ok,
        "ambient within bounds": a_ok,
        "water weights sum to 1": w_ok,
        "output geometry preserved": shape_ok,
        "restored image in [0, 1]": restored_ok,
        "regrade . inverse = image": err < 1e-3,
        "strided forward ~ same field on shared pixels": stride_ok,
        "odd sizes: stride grid lines up": odd_ok,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        print(f"round-trip max error: {err:.2e}")
        print(f"stride mismatch: {stride_err:.2e}, "
              f"odd shape: {tuple(out_odd.tau.shape)}")
        return 1
    print(f"round-trip max error: {err:.2e}  (threshold 1e-3)")
    print(f"stride max error: {stride_err:.2e}  (threshold 2e-3, "
          f"global-pooling sampling error)")
    print(f"parameters: {model.count_parameters()}")
    return 0


def _evaluation_datasets() -> List[str]:
    """Datasets HydroBench evaluates on, read from the committed manifests.

    Used to refuse a meta-training source that lives inside one of them.  This
    is derived from the manifests rather than hardcoded so it stays correct as
    datasets are added -- the leakage this guards against is exactly the kind
    of thing a hardcoded list silently goes stale on.
    """
    manifest_dir = REPO / "hydro_bench" / "datasets" / "manifests"
    if not manifest_dir.is_dir():
        return []
    out = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception:
            continue
        if blob.get("splits", {}).get("test"):
            out.append(path.stem)
    return out


def _source_dataset_name(source: Path) -> Optional[str]:
    """The HydroBench dataset a source directory sits inside, or None.

    ``datasets/LSUI/GT`` -> ``lsui``; anything outside ``datasets/`` -> None.
    Returns None rather than raising when the source is elsewhere on disk,
    which is a perfectly legitimate place to keep a clean photo pool.
    """
    # Resolve first: a relative path can never be relative_to() an absolute
    # base, so an un-resolved source would fall through to None and silently
    # bypass the guard entirely.
    source = Path(source).resolve()
    try:
        parts = source.relative_to((REPO / "datasets").resolve()).parts
    except ValueError:
        return None
    if not parts:
        return None
    return parts[0].lower()


def _assert_source_is_clean(source: Path, datasets: List[str],
                            allow_eval_source: bool, logger) -> bool:
    """Refuse a meta-training source drawn from an evaluation dataset.

    The zero-shot claim of the paper is that no real underwater image enters
    meta-training.  A benchmark dataset's *reference* directory is the most
    tempting source -- it is the cleanest imagery in the repo -- and it would
    falsify that claim outright, so this is a hard stop unless the caller
    explicitly overrides it and the override is recorded in the provenance.
    See Assumption A8 in docs/METHODOLOGY_RESOLUTION.md.
    """
    candidate = _source_dataset_name(source)
    if candidate is None:
        return True
    offenders = [d for d in datasets if d.lower() == candidate]
    if not offenders:
        return True
    if allow_eval_source:
        logger.warning(
            f"OVERRIDE: meta-training source {source} is inside the evaluation "
            f"dataset {offenders[0]}; the zero-shot claim does not hold for "
            f"this checkpoint and must not be asserted in the manuscript.")
        return True
    logger.error(
        f"refusing meta-training source {source}: it is inside the evaluation "
        f"dataset {offenders[0]}. The paper's zero-shot claim requires that no "
        f"real underwater image enter meta-training, and the dataset's "
        f"reference images are real underwater photographs. "
        f"Pass --allow-eval-source to proceed anyway (recorded in the "
        f"provenance, and the claim must then be dropped).")
    return False


def cmd_meta_train(args) -> int:
    import torch

    from .meta_train import MetaTrainer
    from .utils import (deduplicate, get_logger, resolve_device,
                        seed_everything, write_provenance)
    from . import physics

    cfg = _config_from_args(args)
    seed_everything(cfg.seed)

    logger = get_logger("physinr.meta", Path(args.log_dir))
    source = REPO / args.source
    if not source.is_dir():
        logger.error(f"meta-training source not found: {source}")
        return 2

    eval_sets = _evaluation_datasets()
    if not _assert_source_is_clean(source, eval_sets,
                                   args.allow_eval_source, logger):
        return 3

    dedup_record: Dict[str, object] = {
        "performed": False,
        "evaluation_datasets": eval_sets,
        "source": str(source),
        "source_inside_evaluation_dataset": _source_dataset_name(source)
        in {d.lower() for d in eval_sets},
    }
    if not args.no_dedup:
        # Dedup against the test split of *every* manifest, not a hardcoded
        # subset, so a new dataset cannot leak through a list that was not
        # updated to mention it.
        refs = []
        for ds in eval_sets:
            try:
                paths = _dataset_test_images(ds)
                refs.append((ds, paths))
                logger.info(f"dedup reference: {ds} ({len(paths)} test images)")
            except Exception as exc:
                logger.warning(f"could not enumerate test images for {ds}: {exc}")
        if refs:
            dedup_result = deduplicate(source, refs, logger=logger)
            sources = [Path(p) for p in dedup_result["kept"]]
            dedup_record.update({"performed": True, **dedup_result,
                                 "reference_sets": {n: len(p) for n, p in refs}})
        else:
            logger.warning("no reference sets available; running without dedup")
            sources = sorted(source.rglob("*"))
    else:
        sources = sorted(source.rglob("*"))
        logger.warning("--no-dedup set: meta-training source is NOT checked "
                       "against evaluation test splits")

    sources = [p for p in sources
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}]
    if not sources:
        logger.error(f"no usable clean images under {source}")
        return 2
    logger.info(f"meta-training on {len(sources)} clean source images, "
                f"{cfg.meta_tasks} synthetic tasks, device={cfg.device}")

    trainer = MetaTrainer(cfg, sources, log_dir=Path(args.log_dir), logger=logger)
    record = trainer.train(out_path=REPO / args.out,
                           provenance=dedup_record)
    logger.info(f"meta-training complete: {record.tasks_completed} tasks in "
                f"{record.duration_seconds:.0f}s, meta_loss={record.final_meta_loss:.5f}")
    return 0


def cmd_enhance(args) -> int:
    from .adapt import TestTimeAdapter, load_meta_checkpoint
    from .config import PhysINRConfig
    from .model import PhysINR
    from .utils import get_logger, load_image, resolve_device, to_uint8, write_provenance

    from PIL import Image

    cfg = _config_from_args(args)

    logger = get_logger("physinr.enhance", Path(args.log_dir))
    ckpt = REPO / args.checkpoint
    if not ckpt.is_file():
        logger.error(f"meta-initialisation not found: {ckpt} "
                     f"(run `python -m hydro_bench.methods.deep.physinr_zero meta-train` first)")
        return 2
    # The checkpoint's own configuration decides the architecture and the
    # objective; --legacy still forces the pre-refactoring fields on top.
    cfg = _checkpoint_config(ckpt, getattr(args, "legacy", False))
    if args.device is not None:
        cfg = replace(cfg, device=args.device)

    model = PhysINR(cfg)
    blob = load_meta_checkpoint(model, ckpt)
    logger.info(f"loaded meta-initialisation from {ckpt} "
                f"({blob.get('param_count')} params, {blob.get('tasks')} tasks)")
    adapter = TestTimeAdapter(cfg, model)

    src = Path(args.input)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if src.is_file():
        items = [src]
        out_dir = Path(args.output).parent
        default_name = Path(args.output).name
    else:
        items = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
        out_dir = Path(args.output)
        default_name = None
        if args.limit:
            items = items[: args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = {}
    total = 0.0
    for i, path in enumerate(items):
        image = load_image(path)
        name = default_name if default_name else f"{path.stem}.png"
        enhanced, trace, dt = enhance_one(adapter, image, path.name, name,
                                          args.steps, args.trace, logger)
        Image.fromarray(to_uint8(enhanced)).save(out_dir / name)
        total += dt
        if args.trace and trace is not None:
            traces[str(path)] = trace.to_dict()

    provenance = {
        "checkpoint": str(ckpt),
        "checkpoint_tasks": blob.get("tasks"),
        "steps": args.steps or cfg.adapt_steps,
        "images": len(items),
        "seconds_per_image": total / max(1, len(items)),
        "config": cfg.to_dict(),
        "device": adapter.device,
    }
    write_provenance(out_dir / "enhance_provenance.json", provenance)
    if traces:
        write_provenance(out_dir / "adaptation_traces.json", traces)
    logger.info(f"enhanced {len(items)} images in {total:.1f}s "
                f"({provenance['seconds_per_image']:.2f}s/image)")
    return 0


def enhance_one(adapter, image, source_name, out_name, steps,
                return_trace=False, logger=None):
    """Adapt one image and log the result.

    Returns ``(enhanced, trace_or_None, seconds)``.  Adaptation always computes
    the loss -- it is the objective being descended -- so the trace is requested
    on every call and the logged ``final_loss`` is a real number even when the
    caller does not want the full term breakdown.  Reporting NaN here would hide
    a diverging run behind a formatting default.
    """
    t0 = time.time()
    enhanced, trace = adapter.adapt(image, steps=steps, return_trace=True)
    dt = time.time() - t0
    if logger is not None:
        logger.info(f"{source_name} -> {out_name} "
                    f"({dt:.2f}s, final_loss="
                    f"{trace.final_loss if trace else float('nan'):.5f})")
    # Discard the trace unless the caller asked for it, so --trace stays opt-in.
    return enhanced, (trace if return_trace else None), dt


def cmd_compare(args) -> int:
    """Score a pre/post-refactoring checkpoint pair on the same image set.

    The legacy checkpoint runs under the pre-refactoring configuration and the
    refactored checkpoint under the current one, so the difference the report
    measures is the colour-cast refactor and not a change of defaults.

    Cast diagnostics are always reported; UCIQE is reported when HydroBench's
    metric implementations are importable, and PSNR/SSIM additionally when a
    matching reference image exists.  A missing or broken metric import must
    not kill the cast comparison, which is the primary output -- hence the
    try/except around the import rather than a hard dependency.
    """
    from .adapt import TestTimeAdapter, load_meta_checkpoint
    from .utils import get_logger, load_image, to_uint8, write_provenance

    logger = get_logger("physinr.compare", Path(args.log_dir))
    src = Path(args.input)
    if src.is_file():
        items, input_root = [src], src.parent
    else:
        items = sorted(p for p in src.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
        input_root = src
        if args.limit:
            items = items[: args.limit]
    if not items:
        logger.error(f"no images found under {src}")
        return 2

    sides: Dict[str, Dict[str, object]] = {}
    for label, ckpt_arg, legacy in (("legacy", args.legacy_checkpoint, True),
                                    ("refactored", args.refactored_checkpoint, False)):
        ckpt = REPO / ckpt_arg
        if not ckpt.is_file():
            logger.error(f"{label} checkpoint not found: {ckpt}")
            return 2
        overrides = {}
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.device is not None:
            overrides["device"] = args.device
        # Each side runs under the configuration its own checkpoint was
        # trained with; legacy then forces the pre-refactoring fields on top.
        cfg = _checkpoint_config(ckpt, legacy)
        cfg = replace(cfg, **overrides) if overrides else cfg
        model = PhysINR(cfg)
        blob = load_meta_checkpoint(model, ckpt)
        sides[label] = {
            "cfg": cfg,
            "adapter": TestTimeAdapter(cfg, model),
            "checkpoint": str(ckpt),
            "tasks": blob.get("tasks"),
            "param_count": blob.get("param_count"),
        }
        logger.info(f"{label}: {ckpt} ({blob.get('tasks')} tasks, "
                    f"{blob.get('param_count')} params)")

    fr = nr = None
    try:
        from hydro_bench.metrics import full_reference as fr_mod
        from hydro_bench.metrics import no_reference as nr_mod
        fr, nr = fr_mod, nr_mod
    except Exception as exc:
        logger.warning(f"HydroBench metrics unavailable ({exc}); reporting "
                       f"cast diagnostics only")
    if args.reference is not None and not Path(args.reference).is_dir():
        logger.warning(f"reference directory not found: {args.reference}; "
                       f"full-reference metrics will be skipped")
        args.reference = None

    records = []
    for i, path in enumerate(items, 1):
        image = load_image(path)
        rec: Dict[str, object] = {"image": path.name,
                                  "input": _cast_report(image)}
        enhanced_of = {}
        for label, side in sides.items():
            enhanced, _trace = side["adapter"].adapt(image, steps=args.steps)
            rec[label] = _cast_report(enhanced)
            enhanced_of[label] = enhanced
            if nr is not None:
                try:
                    rec[label]["uciqe"] = round(
                        nr.evaluate_nr(to_uint8(enhanced),
                                       metrics=("uciqe",)).uciqe, 4)
                except Exception as exc:
                    logger.warning(f"uciqe failed on {path.name}: {exc}")
                    rec[label]["uciqe"] = None
            else:
                rec[label]["uciqe"] = None
        if fr is not None and args.reference is not None:
            ref_path = _find_reference(path, Path(args.reference), input_root)
            if ref_path is None:
                logger.warning(f"no reference image for {path.name}")
            else:
                ref_img = load_image(ref_path)
                if ref_img.shape[:2] != image.shape[:2]:
                    logger.warning(f"reference geometry mismatch on {path.name}; "
                                   f"full-reference metrics skipped")
                else:
                    for label, enhanced in enhanced_of.items():
                        try:
                            res = fr.evaluate_pair(to_uint8(ref_img),
                                                   to_uint8(enhanced),
                                                   metrics=("psnr", "ssim"))
                            rec[label]["psnr"] = round(res.psnr, 4)
                            rec[label]["ssim"] = round(res.ssim, 4)
                        except Exception as exc:
                            logger.warning(f"FR metrics failed on {path.name}: {exc}")
        records.append(rec)
        logger.info(f"[{i}/{len(items)}] {path.name}: |cast| input="
                    f"{rec['input']['lab_cast_magnitude']:.2f} "
                    f"legacy={rec['legacy']['lab_cast_magnitude']:.2f} "
                    f"refactored={rec['refactored']['lab_cast_magnitude']:.2f}")

    # ---- aggregate --------------------------------------------------------
    metric_keys = ("lab_mu_a", "lab_mu_b", "lab_cast_magnitude", "yiq_mu_i",
                   "yiq_mu_q", "uciqe", "psnr", "ssim")

    def mean_of(side: str, key: str) -> Optional[float]:
        vals = [r[side][key] for r in records
                if r.get(side, {}).get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {side: {k: mean_of(side, k) for k in metric_keys}
               for side in ("input", "legacy", "refactored")}

    def paired_delta(key: str) -> Optional[float]:
        vals = [r["refactored"][key] - r["legacy"][key] for r in records
                if r["refactored"].get(key) is not None
                and r["legacy"].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    n_reduced = sum(1 for r in records
                    if r["refactored"]["lab_cast_magnitude"]
                    < r["legacy"]["lab_cast_magnitude"])
    summary["cast_reduced_on"] = f"{n_reduced}/{len(records)} images"
    summary["mean_cast_magnitude_reduction"] = round(float(np.mean([
        r["legacy"]["lab_cast_magnitude"] - r["refactored"]["lab_cast_magnitude"]
        for r in records])), 4)
    summary["mean_psnr_change"] = paired_delta("psnr")
    summary["mean_ssim_change"] = paired_delta("ssim")
    summary["mean_uciqe_change"] = paired_delta("uciqe")

    report = {
        "input": str(src),
        "reference": str(args.reference) if args.reference else None,
        "images": len(records),
        "steps": args.steps or PhysINRConfig().adapt_steps,
        "sides": {label: {"checkpoint": side["checkpoint"],
                          "tasks": side["tasks"],
                          "param_count": side["param_count"],
                          "config": side["cfg"].to_dict()}
                  for label, side in sides.items()},
        "summary": summary,
        "per_image": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(out_path, report)

    s = summary
    logger.info(f"wrote {out_path}")
    logger.info(
        f"mean |chroma cast|:  input={s['input']['lab_cast_magnitude']}  "
        f"legacy={s['legacy']['lab_cast_magnitude']}  "
        f"refactored={s['refactored']['lab_cast_magnitude']}  "
        f"({s['cast_reduced_on']} reduced)")
    logger.info(
        f"mean Lab mu_a:       input={s['input']['lab_mu_a']}  "
        f"legacy={s['legacy']['lab_mu_a']}  "
        f"refactored={s['refactored']['lab_mu_a']}")
    for key, label in (("psnr", "PSNR (dB)"), ("ssim", "SSIM"),
                       ("uciqe", "UCIQE")):
        if s["legacy"][key] is not None and s["refactored"][key] is not None:
            logger.info(f"mean {label:<11s} legacy={s['legacy'][key]}  "
                        f"refactored={s['refactored'][key]}  "
                        f"change={s[f'mean_{key}_change']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return {
        "info": cmd_info,
        "smoke": cmd_smoke,
        "meta-train": cmd_meta_train,
        "enhance": cmd_enhance,
        "compare": cmd_compare,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
