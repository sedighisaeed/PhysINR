"""Semantic segmentation evaluation for SUIM: mIoU and per-class IoU.

Standard intersection-over-union over the class set, plus the aggregate
metrics a paper reports (mIoU, pixel accuracy, frequency-weighted IoU, and
Dice, which is the standard SUIM-reported boundary metric).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# SUIM's published class set, in mask-index order.
SUIM_CLASSES = [
    "background (waterbody)",
    "human/diver",
    "aquatic plant/sea-grass",
    "wreck/ruins",
    "robot/instruments",
    "reef/invertebrates",
    "fish",
    "sea-floor",
]


@dataclass
class SegmentationReport:
    miou: float = 0.0
    pixel_accuracy: float = 0.0
    mean_dice: float = 0.0
    frequency_weighted_iou: float = 0.0
    per_class_iou: Dict[int, float] = field(default_factory=dict)
    per_class_dice: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    n_classes: int = 0
    n_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mIoU": self.miou,
            "pixel_acc": self.pixel_accuracy,
            "mean_dice": self.mean_dice,
            "FWIoU": self.frequency_weighted_iou,
            "per_class_IoU": {str(k): v for k, v in self.per_class_iou.items()},
            "per_class_dice": {str(k): v for k, v in self.per_class_dice.items()},
            "n_samples": self.n_samples,
        }

    def class_names(self, classes: Sequence[str]) -> Dict[str, float]:
        """Per-class IoU keyed by human-readable class name."""
        return {classes[i]: self.per_class_iou.get(i, float("nan"))
                for i in range(len(classes))}


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    """Accumulate a CxC confusion matrix (rows=gt, cols=pred)."""
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch {pred.shape} vs {gt.shape}")
    k = (gt.astype(np.int64) * n_classes + pred.astype(np.int64)).ravel()
    cm = np.bincount(k, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).astype(np.float64)


def metrics_from_confusion(cm: np.ndarray) -> SegmentationReport:
    """Derive every aggregate metric from one confusion matrix."""
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    gt_count = cm.sum(axis=1)      # true instances per class
    pred_count = cm.sum(axis=0)    # predicted instances per class
    union = gt_count + pred_count - tp
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / np.maximum(union, 1e-12), np.nan)
        dice = np.where((gt_count + pred_count) > 0,
                        2 * tp / np.maximum(gt_count + pred_count, 1e-12), np.nan)
    # Classes absent from ground truth are excluded from mIoU (standard).
    present = gt_count > 0
    miou = float(np.nanmean(iou[present])) if present.any() else 0.0
    mean_dice = float(np.nanmean(dice[present])) if present.any() else 0.0
    pixel_acc = float(tp.sum() / total) if total > 0 else 0.0
    freq = gt_count / total if total > 0 else np.zeros_like(gt_count)
    fwiou = float(np.nansum(np.where(np.isfinite(iou), freq * iou, 0.0)))

    return SegmentationReport(
        miou=miou,
        pixel_accuracy=pixel_acc,
        mean_dice=mean_dice,
        frequency_weighted_iou=fwiou,
        per_class_iou={int(i): (float(iou[i]) if np.isfinite(iou[i]) else float("nan"))
                       for i in range(n_classes)},
        per_class_dice={int(i): (float(dice[i]) if np.isfinite(dice[i]) else float("nan"))
                        for i in range(n_classes)},
        confusion_matrix=cm,
        n_classes=n_classes,
    )


def evaluate_segmentation(predictions: Sequence[np.ndarray],
                          ground_truth: Sequence[np.ndarray],
                          n_classes: int,
                          ignore_index: Optional[int] = None) -> SegmentationReport:
    """Aggregate IoU metrics over a whole validation split.

    Masks may be RGB (class index encoded in a channel) or single-channel label
    maps; both are reduced to a class-index array.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("prediction / ground-truth count mismatch")
    total_cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    n_samples = 0
    for pred, gt in zip(predictions, ground_truth):
        p = _to_class_map(pred, n_classes)
        g = _to_class_map(gt, n_classes)
        if ignore_index is not None:
            valid = g != ignore_index
            p, g = p[valid], g[valid]
        p = np.clip(p, 0, n_classes - 1)
        g = np.clip(g, 0, n_classes - 1)
        total_cm += confusion_matrix(p, g, n_classes)
        n_samples += 1
    report = metrics_from_confusion(total_cm)
    report.n_samples = n_samples
    return report


def _to_class_map(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Reduce an RGB or single-channel mask to a class-index map."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        # SUIM encodes classes as distinct RGB triples; reduce via hashing.
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        flat = arr.reshape(-1, arr.shape[2])
        keys = np.unique(flat, axis=0)
        lut = {tuple(int(c) for c in key): idx
               for idx, key in enumerate(keys)}
        out = np.empty(flat.shape[0], dtype=np.int64)
        for i, row in enumerate(flat):
            out[i] = lut[tuple(int(c) for c in row)]
        return out.reshape(arr.shape[:2])
    return arr.astype(np.int64)


# ------------------------------------------------------------------ deltas
def delta_report(baseline: SegmentationReport,
                 candidate: SegmentationReport) -> Dict[str, Any]:
    def d(name: str) -> Dict[str, Any]:
        b, c = getattr(baseline, name), getattr(candidate, name)
        return {"baseline": b, "candidate": c, "delta": c - b}

    out = {name: d(name) for name in ("miou", "pixel_accuracy", "mean_dice",
                                      "frequency_weighted_iou")}
    out["per_class_IoU_delta"] = {
        cls: candidate.per_class_iou.get(cls, float("nan"))
             - baseline.per_class_iou.get(cls, float("nan"))
        for cls in set(baseline.per_class_iou) | set(candidate.per_class_iou)
    }
    return out
