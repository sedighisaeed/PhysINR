"""End-to-end benchmark orchestration.

This is the layer that connects datasets, methods (online or offline), and
metrics into a single reproducible run, then hands the results to the
reporter. It is the only place that sees both the ground truth and a method's
output in the same process, which is why every leakage guard lives upstream
of it in :mod:`hydro_bench.core.dataset_manager`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core.base_algorithm import BaseAlgorithm, discover_algorithm
from .core.dataset_manager import DatasetManager, Sample, make_manager
from .core.manifest import Manifest
from .methods.offline_importer import OfflineImporter
from .metrics import full_reference as fr
from .metrics import no_reference as nr
from .metrics.downstream import keypoint_match
from .metrics.downstream.object_detection import Detection, DetectionReport
from .metrics.downstream.object_detection import evaluate_detections
from .reporter.latex_tables import BenchmarkResults

FR_DEFAULTS = ("psnr", "ssim", "lpips_alex")
NR_DEFAULTS = ("uciqe", "uiqm", "ccf")


@dataclass
class RunConfig:
    """What one benchmark run computes."""

    dataset: str
    split: str = "test"
    methods: List[str] = field(default_factory=list)
    fr_metrics: List[str] = field(default_factory=list)
    nr_metrics: List[str] = field(default_factory=list)
    keypoint_detectors: List[str] = field(default_factory=list)
    detection: bool = False
    segmentation: bool = False
    limit: Optional[int] = None
    baseline: Optional[str] = None
    out_dir: Path = Path("hydro_bench_results")
    seed: int = 0
    offline_dirs: Dict[str, Path] = field(default_factory=dict)
    method_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verify_checksums: bool = True
    show_progress: bool = True
    # Reuse the incremental checkpoint in out_root to continue an interrupted
    # run instead of restarting from the first image.
    resume: bool = False
    # Whether Apple's MPS private-memory ceiling is left in place for the run.
    # Lifted by default at startup (see prepare_mps_memory_pool) so oversized
    # frames draw on unified memory instead of dropping out of the benchmark.
    cap_mps_memory: bool = False
    # Write every method's enhanced output to out_root/images/<method>/<id>.png
    # so the manuscript's qualitative comparison grid is built from the exact
    # pixels the metrics scored, not a re-run that could differ.
    save_images: bool = False
    # Downstream detection: score the raw and every enhanced frame with a
    # frozen detector trained on the dataset's own train split, then report
    # mAP raw-vs-restored per method. Empty path disables the pass.
    detection_weights: Optional[Path] = None
    detection_imgsz: int = 640
    detection_label_dir: Optional[Path] = None

    def out_root(self) -> Path:
        path = Path(self.out_dir) / f"{self.dataset}__{self.split}"
        path.mkdir(parents=True, exist_ok=True)
        return path


# ------------------------------------------------------------------ engines
@dataclass
class BenchmarkRunner:
    """Runs the full evaluation and accumulates a ``BenchmarkResults``."""

    config: RunConfig
    manager: Optional[DatasetManager] = None
    results: BenchmarkResults = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # sample_id -> metric -> method -> value; the incremental checkpoint.
    _partial: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.results = BenchmarkResults(dataset=self.config.dataset,
                                        split=self.config.split)

    # ------------------------------------------------------------- setup
    def load_dataset(self) -> DatasetManager:
        if self.manager is None:
            self.manager = make_manager(self.config.dataset,
                                        verify=True,
                                        verify_checksums=self.config.verify_checksums)
        return self.manager

    def samples(self) -> List[Sample]:
        manager = self.load_dataset()
        return manager.load_split(self.config.split,
                                  limit=self.config.limit,
                                  require_gt=bool(self.config.fr_metrics))

    # ------------------------------------------------------- method sources
    def _method_names(self) -> List[str]:
        return list(dict.fromkeys(self.config.methods))

    def _build_algorithm(self, method: str) -> Optional[BaseAlgorithm]:
        params = self.config.method_params.get(method, {})
        try:
            cls = discover_algorithm(method)
        except KeyError:
            self.warnings.append(
                f"method '{method}' has no registered adapter; "
                "available methods are the offline result directories")
            return None
        try:
            return cls(**params)
        except FileNotFoundError as exc:
            self.warnings.append(f"method '{method}': {exc}")
            return None
        except Exception as exc:  # any construction failure must not kill the run
            self.warnings.append(f"method '{method}' failed to initialise: {exc}")
            return None

    def _offline_importer(self, method: str) -> Optional[OfflineImporter]:
        directory = self.config.offline_dirs.get(method)
        if directory is None:
            return None
        return OfflineImporter(output_dir=Path(directory), method=method)

    @staticmethod
    def _method_metadata(method: str, source: Any) -> Dict[str, Any]:
        """One method's provenance, straight from the adapter or importer.

        ``run_metadata`` is what distinguishes a real published model from a
        labelled analytic fallback, and it carries the execution device and any
        accelerator-OOM count. Without it a results file cannot tell the two
        apart, so it is recorded for every method.
        """
        get = getattr(source, "run_metadata", None)
        if callable(get):
            try:
                meta = get()
            except Exception as exc:  # provenance must never sink a run
                return {"method": method,
                        "note": f"run_metadata unavailable: {exc}"}
            if isinstance(meta, dict):
                return {**meta, "method": meta.get("method", method)}
        return {"method": method, "used_fallback": False,
                "note": "no provenance reported by the adapter"}

    # --------------------------------------------------------------- run
    def run(self) -> BenchmarkResults:
        started = time.time()
        samples = self.samples()
        if not samples:
            raise RuntimeError(
                f"split '{self.config.split}' of dataset "
                f"'{self.config.dataset}' is empty; nothing to evaluate")

        method_names = self._method_names()
        if not method_names:
            raise RuntimeError("no methods were selected for evaluation")

        # Pre-resolve every method so a failure to initialise is reported once,
        # not once per image.
        algorithms: Dict[str, BaseAlgorithm] = {}
        importers: Dict[str, OfflineImporter] = {}
        for method in method_names:
            importer = self._offline_importer(method)
            if importer is not None:
                try:
                    importer.verify_coverage([s.id for s in samples])
                    importers[method] = importer
                except Exception as exc:
                    self.warnings.append(str(exc))
                continue
            algorithm = self._build_algorithm(method)
            if algorithm is not None:
                algorithms[method] = algorithm
                trained_on = algorithm.capabilities.training_sets
                if self.config.dataset in trained_on:
                    # Not a bug in the harness -- a property of the published
                    # model. But a table that silently mixes trained-on and
                    # held-out methods is misleading, so the run says it.
                    self.warnings.append(
                        f"method '{method}' was trained on '{self.config.dataset}'"
                        f"; its scores on this dataset are not held-out")

        active = list(algorithms) + list(importers)
        if not active:
            raise RuntimeError(
                "no method could be initialised or imported; "
                f"warnings: {self.warnings}")

        iterator = samples
        if self.config.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc=f"eval {self.config.dataset}",
                                unit="img")
            except Exception:
                pass

        # Resume support: a previous run on this dataset/split may already have
        # scored some (sample, method) pairs. Replaying them means an
        # interrupted benchmark continues instead of restarting from zero --
        # important when one oversized image costs half an hour. The
        # checkpoint is keyed by method *within* each sample, so an interrupt
        # during the last method of an image replays the earlier ones instead
        # of redoing them.
        done = self._load_partial() if self.config.resume else {}
        # The replayed scores are already complete, so they must stay in the
        # in-memory checkpoint too: _persist_partial writes only _partial, and
        # seeding it from the disk checkpoint keeps the next interrupt from
        # clobbering the very work this run replayed.
        self._partial.update(done)
        self.results = BenchmarkResults(dataset=self.config.dataset,
                                        split=self.config.split)

        for sample in iterator:
            replayed = done.get(sample.id, {})
            for method, scores in replayed.items():
                for metric, value in scores.items():
                    self.results.add(metric, method, [float(value)])
            raw = None
            for method in active:
                if method in replayed:
                    # The scores are already in the checkpoint and are reused
                    # as-is. But if this run is saving outputs, the earlier
                    # run may not have been -- leaving a hole in the grid that
                    # the figure builder has no way to distinguish from a
                    # method that produced nothing. Recompute the image only;
                    # the metrics are not recomputed, so the table is
                    # unaffected by whatever nondeterminism this introduces.
                    if not self.config.save_images:
                        continue
                    if self._image_exists(sample.id, method):
                        continue
                    try:
                        if raw is None:
                            raw = sample.load_raw()
                        if method in importers:
                            enhanced = importers[method].load(sample.id)
                        else:
                            enhanced = algorithms[method].enhance(raw)
                        self._save_image(sample.id, method, enhanced)
                    except Exception as exc:
                        self.warnings.append(
                            f"{method}/{sample.id}: could not backfill image "
                            f"on resume: {exc}")
                    continue
                try:
                    if raw is None:
                        raw = sample.load_raw()
                    if method in importers:
                        enhanced = importers[method].load(sample.id)
                    else:
                        enhanced = algorithms[method].enhance(raw)
                    if self.config.save_images:
                        self._save_image(sample.id, method, enhanced)
                    self._score_pair(sample, raw, enhanced, method)
                except Exception as exc:
                    self.warnings.append(f"{method}/{sample.id}: {exc}")
                # Checkpoint after every method: an interrupt then costs at
                # most the one (image, method) pair in flight.
                self._persist_partial()
            # Release the GPU's caching allocator back to the OS once the frame
            # is done. On Apple silicon the MPS allocator holds freed buffers
            # rather than returning them, so over hundreds of multi-megapixel
            # frames resident memory climbs until a Metal command buffer
            # deadlocks in waitUntilCompleted (observed on an 8 GB host).
            # empty_cache only releases unused memory, so it cannot disturb an
            # in-flight computation.
            self._release_gpu_cache()

        # Per-method provenance: which rows ran the published model and which
        # ran a labelled fallback, on what device. A reader of the results file
        # needs this to interpret the table, since a fallback row is
        # deliberately still present (and scored) rather than absent.
        method_metadata = {
            method: self._method_metadata(method, source)
            for method, source in {**algorithms, **importers}.items()}

        self.metadata.update({
            "n_samples": len(samples),
            "duration_sec": round(time.time() - started, 2),
            "methods_evaluated": active,
            "methods_skipped": sorted(set(method_names) - set(active)),
            "method_metadata": method_metadata,
            "warnings": self.warnings,
            "manifest": self._manifest_summary(),
            "mps_memory_pool": (
                "capped (Apple default ceiling; oversized frames fall back to "
                "CPU)" if self.config.cap_mps_memory
                else "unbounded (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 at "
                     "startup; oversized frames draw on unified memory)"),
        })
        # The reporter reads per-method provenance from the results object, so
        # the table can mark a fallback row instead of presenting it as the
        # method's own published performance.
        self.results.metadata = method_metadata

        if self.config.detection and self.config.detection_weights is not None:
            self._run_detection(samples, algorithms, importers,
                                method_metadata)
        elif self.config.detection:
            self.warnings.append(
                "detection was requested but no detector weights resolved, so "
                "the downstream mAP pass was skipped")

        self._persist()
        return self.results

    # ----------------------------------------------------------- detection
    def _run_detection(self, samples: List[Sample],
                       algorithms: Dict[str, BaseAlgorithm],
                       importers: Dict[str, OfflineImporter],
                       method_metadata: Dict[str, Any]) -> None:
        """Score raw and enhanced frames with a frozen detector.

        The detector is trained on the dataset's own *train* split and frozen
        for the whole pass, so the only thing varying between rows is the
        enhancement applied upstream. Frames are letterboxed by the detector
        itself; HydroBench resizes nothing, because resampling the frame would
        change the box geometry the mAP is computed over.
        """
        weights = Path(self.config.detection_weights)
        if not weights.is_file():
            self.warnings.append(
                f"detection weights not found at {weights}; the downstream "
                "mAP pass was skipped")
            return

        evaluator = DetectionEvaluator(
            weights=weights, imgsz=self.config.detection_imgsz,
            device=self._detection_device())
        evaluator.load()

        # All frames in a split share one geometry for the datasets HydroBench
        # benchmarks (DUO is uniformly 1920x1080), but the loader is written
        # per-image so a mixed-resolution split still de-normalises correctly.
        label_dir = self.config.detection_label_dir

        ground_truth: List[Detection] = []
        cache: Dict[str, np.ndarray] = {}
        raw_ids: List[str] = []
        raw_batch: List[np.ndarray] = []
        all_raw_preds: List[List[Detection]] = []

        def consume(batch, ids, sink):
            # Ultralytics raises on an empty source, and the final flush after
            # a split whose length is an exact multiple of the batch window is
            # legitimately empty -- nothing to score, so it is a no-op.
            if not batch:
                return
            preds = evaluator.predict(batch, image_ids=ids)
            sink.extend(preds)
            batch.clear()
            ids.clear()

        for sample in samples:
            try:
                raw = sample.load_raw()
            except Exception as exc:
                self.warnings.append(f"detection/{sample.id}: {exc}")
                continue
            stem = Path(sample.id).stem
            cache[stem] = raw
            if label_dir is not None and raw.shape[:2]:
                h, w = raw.shape[:2]
                ground_truth.extend(
                    evaluator.load_ground_truth(Path(label_dir), [stem], (w, h)))
            raw_ids.append(stem)
            raw_batch.append(raw)
            # Ultralytics batches internally; feeding a moderate window keeps
            # peak memory bounded on a unified-memory host.
            if len(raw_batch) >= 8:
                consume(raw_batch, raw_ids, all_raw_preds)
        consume(raw_batch, raw_ids, all_raw_preds)

        if ground_truth:
            self.metadata["detection_ground_truth"] = len(ground_truth)
        self._score_detection(all_raw_preds, list(cache), "(raw)", ground_truth)

        # Then each method's enhanced frames against the same ground truth.
        for method, source in {**algorithms, **importers}.items():
            method_ids: List[str] = []
            scored_ids: List[str] = []   # consume() clears method_ids, so the
            method_batch: List[np.ndarray] = []   # scored set is kept separate
            method_preds: List[List[Detection]] = []
            for sample in samples:
                stem = Path(sample.id).stem
                raw = cache.get(stem)
                if raw is None:
                    continue
                try:
                    if method in importers:
                        enhanced = importers[method].load(sample.id)
                    else:
                        enhanced = source.enhance(raw)
                except Exception as exc:
                    self.warnings.append(
                        f"detection/{method}/{sample.id}: {exc}")
                    continue
                method_ids.append(stem)
                method_batch.append(enhanced)
                if len(method_batch) >= 8:
                    scored_ids.extend(method_ids)
                    consume(method_batch, method_ids, method_preds)
            scored_ids.extend(method_ids)
            consume(method_batch, method_ids, method_preds)
            if scored_ids or method_preds:
                self._score_detection(method_preds, scored_ids, method,
                                      ground_truth)
            method_metadata.setdefault(method, {})["detection"] = (
                "scored with frozen detector on this split")

    def _score_detection(self, preds: List[List[Detection]], ids: List[str],
                         method: str,
                         ground_truth: Optional[List[Detection]] = None
                         ) -> None:
        """Fold detector output into the results table.

        Called once per method with *every* prediction on the split, because
        COCO mAP is a global ranking metric: scoring batches of eight
        separately and averaging the per-batch APs is not the same number as
        scoring the split, and the difference is not in our favour to ignore.
        """
        flat = [d for per_image in preds for d in per_image]
        self.results.add("n_detections", method, [float(len(flat))])
        self.results.add("n_images", method, [float(len(ids))])
        if ground_truth:
            report = evaluate_detections(flat, ground_truth)
            self.results.add("map50", method, [report.map_50])
            self.results.add("map5095", method, [report.map_5095])
            self.results.add("map75", method, [report.map_75])
            self.results.add("mar5095", method, [report.mar_5095])

    @staticmethod
    def _detection_device() -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _release_gpu_cache() -> None:
        """Return the GPU caching allocator's freed buffers to the OS.

        On Apple silicon the MPS allocator holds released buffers rather than
        handing them back, so across hundreds of multi-megapixel frames
        resident memory climbs until a Metal command buffer deadlocks inside
        ``waitUntilCompleted`` (observed on an 8 GB host). Called once per
        frame, after the results are checkpointed.
        """
        try:
            import torch
        except Exception:
            return
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # A cache release is best-effort memory hygiene; it must never
            # take a run down with it.
            pass

    # ------------------------------------------------------------- scoring
    def _image_exists(self, sample_id: str, method: str) -> bool:
        """Whether a saved output for this pair is already on disk."""
        import cv2

        if not sample_id.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            return False
        path = self.config.out_root() / "images" / method / sample_id
        if not path.is_file():
            return False
        # A zero-byte or undecodable file would read as a blank tile, so test
        # decode rather than existence alone.
        return cv2.imread(str(path)) is not None

    def _save_image(self, sample_id: str, method: str,
                    enhanced: np.ndarray) -> None:
        """Write one enhanced output next to the run's metrics.

        The grid in the manuscript is assembled from these files, which are the
        same pixels the metrics scored. Re-generating a figure image later from
        a fresh ``enhance()`` call risks a different output (nondeterministic
        kernels, changed defaults) than the number printed in the table.
        """
        import cv2

        if enhanced is None or enhanced.size == 0:
            return
        folder = self.config.out_root() / "images" / method
        folder.mkdir(parents=True, exist_ok=True)
        bgr = enhanced[:, :, ::-1]
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            self.warnings.append(f"{method}/{sample_id}: could not encode PNG")
            return
        # Sample ids may already end in .png/.jpg (SUIM, DUO); append .png only
        # when the id is bare, or the grid builder finds two extensions.
        stem = Path(sample_id).stem if sample_id.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) else sample_id
        buf.tofile(str(folder / f"{stem}.png"))

    def _score_pair(self, sample: Sample, raw: np.ndarray,
                    enhanced: np.ndarray, method: str) -> None:
        # Accumulate into a throwaway dict first: an empty entry must never
        # reach ``_partial``, or --resume would treat the pair as finished even
        # though no metric produced a score for it.
        record: Dict[str, float] = {}
        if self.config.fr_metrics and sample.gt_path is not None:
            gt = sample.load_gt()
            result = fr.evaluate_pair(gt, enhanced, metrics=self.config.fr_metrics)
            for key, value in result.as_dict().items():
                if value is not None:
                    self.results.add(key, method, [float(value)])
                    record[key] = float(value)
        if self.config.nr_metrics:
            result = nr.evaluate_nr(enhanced, metrics=tuple(self.config.nr_metrics))
            for key, value in result.as_dict().items():
                if value is not None:
                    self.results.add(key, method, [float(value)])
                    record[key] = float(value)
        if record:
            self._partial.setdefault(sample.id, {})[method] = record
        if self.config.keypoint_detectors:
            report = keypoint_match.evaluate_both_matchers(
                raw, enhanced, detectors=self.config.keypoint_detectors,
                homography=np.eye(3))
            for detector, rep in report.items():
                self.results.add(f"repeatability", f"{method} ({detector})",
                                 [float(rep.repeatability)])
                self.results.add(f"inlier_ratio", f"{method} ({detector})",
                                 [float(rep.inlier_ratio)])

    # ----------------------------------------------------------- persistence
    def _partial_path(self) -> Path:
        return self.config.out_root() / "partial.json"

    def _persist_partial(self) -> None:
        """Atomically checkpoint the scores collected so far."""
        path = self._partial_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._partial, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_partial(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Read the incremental checkpoint left by an interrupted run."""
        path = self._partial_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"could not read resume checkpoint {path}: {exc}. Delete it "
                "or rerun without --resume") from exc
        if not isinstance(data, dict):
            return {}
        return data

    def _persist(self) -> None:
        out = self.config.out_root()
        (out / "results.json").write_text(
            json.dumps(self._serialise(), indent=2, default=_json_default),
            encoding="utf-8")
        # The run is complete and fully serialised; the checkpoint is stale.
        partial = self._partial_path()
        if partial.is_file():
            partial.unlink()

    def _serialise(self) -> Dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "split": self.config.split,
            "config": {
                "fr_metrics": self.config.fr_metrics,
                "nr_metrics": self.config.nr_metrics,
                "limit": self.config.limit,
                "baseline": self.config.baseline,
                "seed": self.config.seed,
            },
            "scores": {
                metric: {method: list(map(float, values))
                         for method, values in by_method.items()}
                for metric, by_method in self.results.scores.items()
            },
            "metadata": self.metadata,
        }

    def _manifest_summary(self) -> Dict[str, Any]:
        manager = self.manager
        if manager is None or manager.spec.manifest_path is None:
            return {}
        path = Path(manager.spec.manifest_path)
        return {"path": str(path),
                "dataset": manager.spec.name,
                "splits": {k: len(v) for k, v in manager.manifest.splits.items()}}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj)} is not JSON serialisable")


# --------------------------------------------------------------- detection
@dataclass
class DetectionEvaluator:
    """Scores raw vs. enhanced frames with a frozen detector (DUO/URPC)."""

    weights: Path
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.6
    device: str = "cpu"
    model: Any = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "downstream detection requires the ultralytics package "
                "(pip install ultralytics). HydroBench never installs it "
                "implicitly.") from exc
        self.model = YOLO(str(self.weights))

    def predict(self, images: Sequence[np.ndarray],
                image_ids: Optional[Sequence[str]] = None
                ) -> List[List[Detection]]:
        """Run the detector over a batch of frames.

        Parameters
        ----------
        images:
            Frames to score, in the order the caller wants them reported.
        image_ids:
            Stable per-image identifiers (typically the file stem) that the
            predictions are matched against ground truth by. When omitted the
            caller gets the batch position instead, which is *only* safe for a
            single call over the whole split; passing stems is what makes the
            COCO matching join predictions to ground truth at all.
        """
        if self.model is None:
            self.load()
        if image_ids is not None and len(image_ids) != len(images):
            raise ValueError(
                f"{len(image_ids)} image_ids for {len(images)} images; the two "
                "must correspond element-wise")
        results = self.model.predict(list(images), imgsz=self.imgsz,
                                     conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False)
        out: List[List[Detection]] = []
        for index, result in enumerate(results):
            image_id = (image_ids[index] if image_ids is not None
                        else str(index))
            dets: List[Detection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                out.append(dets)
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x0, y0, x1, y1 = xyxy
                dets.append(Detection(image_id=str(image_id),
                                      category_id=int(box.cls[0]),
                                      bbox=(x0, y0, x1 - x0, y1 - y0),
                                      score=float(box.conf[0])))
            out.append(dets)
        return out

    @staticmethod
    def load_ground_truth(label_dir: Path, image_ids: Sequence[str],
                          shape: Tuple[int, int]) -> List[Detection]:
        """Read YOLO-format ``.txt`` labels as absolute-pixel detections.

        Parameters
        ----------
        label_dir:
            Directory holding one ``<stem>.txt`` per image, in the Ultralytics
            layout (``class cx cy w h``, all normalised to [0, 1]).
        image_ids:
            Stems to load; images with no label file contribute no boxes.
        shape:
            ``(width, height)`` of the images the labels describe, used to
            de-normalise the boxes into absolute pixels.
        """
        width, height = shape
        ground_truth: List[Detection] = []
        label_dir = Path(label_dir)
        for stem in image_ids:
            path = label_dir / f"{stem}.txt"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = (float(p) for p in parts[:5])
                abs_w, abs_h = w * width, h * height
                x0, y0 = cx * width - abs_w / 2.0, cy * height - abs_h / 2.0
                ground_truth.append(
                    Detection(image_id=str(stem), category_id=int(cls),
                              bbox=(x0, y0, abs_w, abs_h), score=0.0,
                              area=abs_w * abs_h))
        return ground_truth

    def evaluate(self, predictions: Sequence[Detection],
                 ground_truth: Sequence[Detection]) -> DetectionReport:
        return evaluate_detections(predictions, ground_truth)
