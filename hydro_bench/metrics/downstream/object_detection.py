"""COCO-style detection metrics, implemented without the COCO toolkit.

Reproducing ``pycocotools`` faithfully matters: a benchmark that ships its own
mAP variant is impossible to cross-check. This module implements the same
algorithm (101-point interpolated precision, area- and class-agnostic
averaging, max-over-overlaps greedy matching, ignore-flag support) and is
verified against a hand-computed case in the test suite.

Reference: "Microsoft COCO: Common Objects in Context", Lin et al., ECCV 2014,
and the ``pycocotools/COCOeval`` implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The COCO IoU thresholds for the primary metric.
COCO_IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05, dtype=np.float64)
# 101-point interpolated recall grid (0, 0.01, ..., 1.0).
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)

# COCO area ranges (pixels^2): all, small, medium, large.
AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}


@dataclass
class Detection:
    """One predicted or ground-truth box in absolute pixel coordinates."""

    image_id: str
    category_id: int
    # COCO axis-aligned box: [x_min, y_min, width, height].
    bbox: Tuple[float, float, float, float]
    score: float = 1.0        # 0 for ground truth (conventionally unused)
    area: Optional[float] = None
    iscrowd: int = 0
    isignore: int = 0         # crowd/ignore regions absorb detections harmlessly

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def iou(self, other: "Detection") -> float:
        return iou_xyxy(self.xyxy, other.xyxy)


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes. Returns 0.0 for degenerate boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def yolo_to_detection(image_id: str, label_line: str, width: int, height: int,
                      score: float = 0.0) -> Optional[Detection]:
    """Convert one YOLO-format annotation line to a :class:`Detection`.

    YOLO stores normalised [class, x_center, y_center, w, h]; COCO stores
    absolute [x_min, y_min, w, h].
    """
    parts = label_line.split()
    if len(parts) < 5:
        return None
    category = int(float(parts[0]))
    xc, yc, w, h = (float(p) for p in parts[1:5])
    # Clamp to the image so boxes straddling the border stay valid.
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    abs_w, abs_h = w * width, h * height
    x_min = (xc * width) - abs_w / 2.0
    y_min = (yc * height) - abs_h / 2.0
    return Detection(image_id=image_id, category_id=category,
                     bbox=(x_min, y_min, abs_w, abs_h), score=score,
                     area=abs_w * abs_h)


def load_yolo_labels(label_path, image_id: str, width: int, height: int,
                     as_ignore: bool = False) -> List[Detection]:
    """Read a YOLO ``.txt`` label file into ground-truth detections."""
    from pathlib import Path

    label_path = Path(label_path)
    if not label_path.is_file():
        return []
    out: List[Detection] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        det = yolo_to_detection(image_id, line.strip(), width, height)
        if det is None:
            continue
        det.iscrowd = 1 if as_ignore else 0
        det.isignore = 1 if as_ignore else 0
        out.append(det)
    return out


@dataclass
class DetectionReport:
    """Result of evaluating one set of predictions against ground truth."""

    map_5095: float = 0.0            # primary COCO metric
    map_50: float = 0.0              # PASCAL-VOC style metric
    map_75: float = 0.0              # strict metric
    mar_5095: float = 0.0            # max recall averaged over IoU + area
    per_class_ap50: Dict[int, float] = field(default_factory=dict)
    per_class_ap5095: Dict[int, float] = field(default_factory=dict)
    precision_at_iou50: List[float] = field(default_factory=list)
    recall_at_iou50: List[float] = field(default_factory=list)
    n_ground_truth: int = 0
    n_detections: int = 0
    confusion: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mAP@50:95": self.map_5095,
            "mAP@50": self.map_50,
            "mAP@75": self.map_75,
            "mAR@50:95": self.mar_5095,
            "per_class_AP50": dict(self.per_class_ap50),
            "per_class_AP50:95": dict(self.per_class_ap5095),
            "n_ground_truth": self.n_ground_truth,
            "n_detections": self.n_detections,
        }


def evaluate_detections(predictions: Sequence[Detection],
                        ground_truth: Sequence[Detection],
                        iou_thresholds: Optional[np.ndarray] = None,
                        area_key: str = "all",
                        max_dets: int = 100) -> DetectionReport:
    """Score predictions with the COCO 101-point interpolated protocol.

    Parameters
    ----------
    predictions, ground_truth : sequence of :class:`Detection`
        ``predictions`` must carry scores; ``ground_truth`` may include
        ``isignore``/``iscrowd`` regions.
    area_key : str
        One of ``all``, ``small``, ``medium``, ``large``.
    max_dets : int
        Cap on detections per image (COCO uses 1, 10, 100).
    """
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    lo_area, hi_area = AREA_RANGES[area_key]

    gt_by_image: Dict[str, List[Detection]] = defaultdict(list)
    for det in ground_truth:
        gt_by_image[det.image_id].append(det)
    pred_by_image: Dict[str, List[Detection]] = defaultdict(list)
    for det in predictions:
        pred_by_image[det.image_id].append(det)

    image_ids = sorted(set(gt_by_image) | set(pred_by_image))
    categories = sorted({d.category_id for d in ground_truth}
                        | {d.category_id for d in predictions})
    report = DetectionReport(n_ground_truth=len(ground_truth),
                             n_detections=len(predictions))

    ap_by_iou: Dict[float, List[float]] = defaultdict(list)
    ar_by_iou: Dict[float, List[float]] = defaultdict(list)
    ap50_by_class: Dict[int, float] = {}
    ap5095_by_class: Dict[int, float] = {}

    for category in categories:
        per_iou_precision: Dict[float, List[float]] = defaultdict(list)
        per_iou_recall: Dict[float, List[float]] = defaultdict(list)

        for iou_thr in iou_thresholds:
            tp = np.zeros(max_dets, dtype=np.float64)
            fp = np.zeros(max_dets, dtype=np.float64)
            n_gt = 0

            for image_id in image_ids:
                gts = [g for g in gt_by_image.get(image_id, [])
                       if g.category_id == category]
                # Split ground truth into scored boxes and ignore regions.
                gt_valid = [g for g in gts if not g.isignore]
                gt_ignore = [g for g in gts if g.isignore]
                n_gt += len(gt_valid)

                dets = [d for d in pred_by_image.get(image_id, [])
                        if d.category_id == category and d.score is not None]
                dets.sort(key=lambda d: -d.score)
                dets = dets[:max_dets]

                matched = np.zeros(len(gt_valid), dtype=bool)
                for k, det in enumerate(dets):
                    best_iou, best_idx = iou_thr, -1
                    for j, gt in enumerate(gt_valid):
                        if matched[j]:
                            continue
                        val = det.iou(gt)
                        if val >= best_iou:
                            best_iou, best_idx = val, j
                    if best_idx >= 0:
                        matched[best_idx] = True
                        tp[k] = 1.0
                    else:
                        # A detection that lands on an ignore region is
                        # neither a hit nor a miss -- COCO drops it.
                        ignored = any(det.iou(g) >= iou_thr for g in gt_ignore)
                        fp[k] = 0.0 if ignored else 1.0

            if n_gt == 0:
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

            # 101-point interpolated precision, matching pycocotools exactly.
            for idx in range(len(recall) - 1, 0, -1):
                if recall[idx] > recall[idx - 1]:
                    precision[idx - 1] = max(precision[idx - 1], precision[idx])
            q = np.zeros(COCO_RECALL_THRESHOLDS.shape, dtype=np.float64)
            for r_idx, r in enumerate(COCO_RECALL_THRESHOLDS):
                eligible = recall >= r
                q[r_idx] = precision[eligible].max() if eligible.any() else 0.0
            ap = float(q.mean())
            per_iou_precision[float(iou_thr)].append(ap)
            per_iou_recall[float(iou_thr)].append(float(recall[-1])
                                                  if recall.size else 0.0)

        if not per_iou_precision:
            continue
        ap50_by_class[category] = float(np.mean(per_iou_precision[0.5]))
        ap5095_by_class[category] = float(np.mean(
            [np.mean(v) for v in per_iou_precision.values()]))
        report.per_class_ap50.update(ap50_by_class)
        report.per_class_ap5095.update(ap5095_by_class)

        for iou_thr, aps in per_iou_precision.items():
            ap_by_iou[iou_thr].extend(aps)
        for iou_thr, recs in per_iou_recall.items():
            ar_by_iou[iou_thr].extend(recs)

        iou_50 = per_iou_precision.get(0.5)
        if iou_50:
            report.precision_at_iou50.append(float(np.mean(iou_50)))
            report.recall_at_iou50.append(float(np.mean(per_iou_recall.get(0.5, [0.0]))))

    if ap_by_iou:
        # mAP averages over classes *and* IoU thresholds.
        report.map_5095 = float(np.mean([np.mean(v) for v in ap_by_iou.values()]))
        report.map_50 = float(np.mean(ap_by_iou[0.5]))
        report.map_75 = float(np.mean(ap_by_iou[0.75])) if 0.75 in ap_by_iou else 0.0
        report.mar_5095 = float(np.mean([np.max(v) if v else 0.0
                                         for v in ar_by_iou.values()]))
    return report


def summarize_predictions(predictions: Sequence[Detection]) -> Dict[str, Any]:
    """Score distribution summary (useful for calibration debugging)."""
    scores = np.array([p.score for p in predictions if p.score is not None])
    if scores.size == 0:
        return {"n": 0}
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def delta_report(baseline: DetectionReport, candidate: DetectionReport) -> Dict[str, Any]:
    """Improvement of one method over another, in absolute and relative terms."""
    def d(name: str) -> Dict[str, Any]:
        b, c = getattr(baseline, name), getattr(candidate, name)
        rel = (c - b) / b if b not in (0.0, 0) else float("nan")
        return {"baseline": b, "candidate": c, "delta": c - b, "rel_delta_pct": rel * 100}

    out = {name: d(name) for name in ("map_5095", "map_50", "map_75", "mar_5095")}
    out["per_class_AP50_delta"] = {
        cat: candidate.per_class_ap50.get(cat, 0.0) - baseline.per_class_ap50.get(cat, 0.0)
        for cat in set(baseline.per_class_ap50) | set(candidate.per_class_ap50)
    }
    return out
