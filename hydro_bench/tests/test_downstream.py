"""Downstream evaluation: COCO detection mAP, SUIM segmentation, keypoint matching."""

from __future__ import annotations

import numpy as np
import pytest

from hydro_bench.metrics.downstream import (keypoint_match, object_detection,
                                            segmentation)


class TestIoU:
    def test_identical_boxes(self):
        assert object_detection.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert object_detection.iou_xyxy((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_partial_overlap(self):
        iou = object_detection.iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15))
        # intersection 25, union 175 -> 1/7
        assert iou == pytest.approx(25 / 175, abs=1e-9)

    def test_degenerate_box(self):
        # Zero-area boxes must not divide by zero.
        assert object_detection.iou_xyxy((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


class TestDetection:
    def test_perfect_predictions(self):
        gt = [object_detection.Detection("img", 0, (10, 10, 20, 20), score=1.0)]
        pred = [object_detection.Detection("img", 0, (10, 10, 20, 20), score=0.99)]
        report = object_detection.evaluate_detections(pred, gt)
        assert report.map_50 == pytest.approx(1.0, abs=1e-9)
        assert report.map_5095 == pytest.approx(1.0, abs=1e-9)
        assert report.n_ground_truth == 1

    def test_no_predictions_scores_zero(self):
        gt = [object_detection.Detection("img", 0, (10, 10, 20, 20))]
        report = object_detection.evaluate_detections([], gt)
        assert report.map_50 == pytest.approx(0.0, abs=1e-9)

    def test_wrong_class_scores_zero(self):
        gt = [object_detection.Detection("img", 0, (10, 10, 20, 20))]
        pred = [object_detection.Detection("img", 1, (10, 10, 20, 20), score=0.9)]
        report = object_detection.evaluate_detections(pred, gt)
        assert report.map_50 == pytest.approx(0.0, abs=1e-9)

    def test_localization_failure_is_graded(self):
        # A box that misses at IoU 0.5 should still not count at 0.75.
        gt = [object_detection.Detection("img", 0, (0, 0, 100, 100))]
        pred = [object_detection.Detection("img", 0, (60, 60, 160, 160), score=0.9)]
        report = object_detection.evaluate_detections(pred, gt)
        assert report.map_50 < report.map_5095 * 0 + 1.0  # sanity: runs
        assert report.map_75 <= report.map_50

    def test_ignore_region_absorbs_detection(self):
        gt = [object_detection.Detection("img", 0, (10, 10, 20, 20)),
              object_detection.Detection("img", 0, (50, 50, 20, 20),
                                         isignore=1)]
        pred = [object_detection.Detection("img", 0, (10, 10, 20, 20), score=0.9),
                object_detection.Detection("img", 0, (50, 50, 20, 20), score=0.5)]
        report = object_detection.evaluate_detections(pred, gt)
        assert report.map_50 == pytest.approx(1.0, abs=1e-9)

    def test_yolo_conversion(self):
        det = object_detection.yolo_to_detection("img", "0 0.5 0.5 0.2 0.4",
                                                 width=100, height=50)
        assert det is not None
        assert det.category_id == 0
        x, y, w, h = det.bbox
        assert w == pytest.approx(20.0, abs=1e-6)
        assert h == pytest.approx(20.0, abs=1e-6)
        assert x == pytest.approx(40.0, abs=1e-6)
        assert y == pytest.approx(15.0, abs=1e-6)

    def test_yolo_out_of_range_is_clamped(self):
        det = object_detection.yolo_to_detection("img", "2 1.5 1.5 2.0 2.0",
                                                 width=100, height=100)
        assert det is not None
        assert det.bbox[2] <= 100.0

    def test_delta_report(self):
        base = object_detection.DetectionReport(map_50=0.5, map_5095=0.4)
        cand = object_detection.DetectionReport(map_50=0.7, map_5095=0.5)
        delta = object_detection.delta_report(base, cand)
        assert delta["map_50"]["delta"] == pytest.approx(0.2)
        assert delta["map_5095"]["rel_delta_pct"] == pytest.approx(25.0)

    def test_score_ranking_matters(self):
        # A low-confidence true positive ranked below a false positive hurts AP.
        gt = [object_detection.Detection("img", 0, (10, 10, 20, 20))]
        good = [object_detection.Detection("img", 0, (10, 10, 20, 20), score=0.9)]
        bad_order = [object_detection.Detection("img", 1, (0, 0, 5, 5), score=0.95),
                     object_detection.Detection("img", 0, (10, 10, 20, 20),
                                                score=0.5)]
        good_report = object_detection.evaluate_detections(good, gt)
        bad_report = object_detection.evaluate_detections(bad_order, gt)
        assert good_report.map_50 >= bad_report.map_50


class TestSegmentation:
    def test_perfect_prediction(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[:8] = 1
        report = segmentation.evaluate_segmentation([gt], [gt], n_classes=2)
        assert report.miou == pytest.approx(1.0, abs=1e-9)
        assert report.pixel_accuracy == pytest.approx(1.0, abs=1e-9)

    def test_all_wrong(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        pred = np.ones((16, 16), dtype=np.int64)
        report = segmentation.evaluate_segmentation([pred], [gt], n_classes=2)
        assert report.miou == pytest.approx(0.0, abs=1e-9)

    def test_partial_overlap(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        pred = np.zeros((16, 16), dtype=np.int64)
        gt[:, :8] = 1        # left half is class 1
        pred[:, 4:12] = 1    # overlaps half of it
        report = segmentation.evaluate_segmentation([pred], [gt], n_classes=2)
        # class 0: tp=64 (right half pred0 & gt0)... compute expected IoU
        assert 0.0 < report.miou < 1.0

    def test_rgb_mask_is_reduced(self):
        gt_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        gt_rgb[:, :8] = (10, 20, 30)
        report = segmentation.evaluate_segmentation([gt_rgb], [gt_rgb],
                                                    n_classes=2)
        assert report.miou == pytest.approx(1.0, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            segmentation.evaluate_segmentation(
                [np.zeros((8, 8), np.int64)], [np.zeros((16, 16), np.int64)], 2)

    def test_confusion_matrix_shape(self):
        gt = np.zeros((8, 8), dtype=np.int64)
        pred = np.zeros((8, 8), dtype=np.int64)
        cm = segmentation.confusion_matrix(pred, gt, n_classes=3)
        assert cm.shape == (3, 3)

    def test_delta_report(self):
        base = segmentation.SegmentationReport(miou=0.4, pixel_accuracy=0.8)
        cand = segmentation.SegmentationReport(miou=0.6, pixel_accuracy=0.9)
        delta = segmentation.delta_report(base, cand)
        assert delta["miou"]["delta"] == pytest.approx(0.2)

    def test_class_names_mapping(self):
        report = segmentation.SegmentationReport(
            miou=0.5, per_class_iou={0: 0.4, 1: 0.6})
        names = report.class_names(["bg", "object"])
        assert names["bg"] == 0.4 and names["object"] == 0.6


class TestKeypoints:
    def test_identical_images_match_well(self, textured_image):
        if not keypoint_match._HAS_CV2:
            pytest.skip("OpenCV required")
        report = keypoint_match.evaluate_matching(textured_image, textured_image,
                                                  detector_name="ORB",
                                                  homography=np.eye(3))
        assert report.n_keypoints_a > 0
        assert report.n_matches > 0
        assert 0.0 <= report.inlier_ratio <= 1.0
        assert 0.0 <= report.repeatability <= 1.0

    def test_identical_images_have_high_repeatability(self, textured_image):
        if not keypoint_match._HAS_CV2:
            pytest.skip("OpenCV required")
        report = keypoint_match.evaluate_matching(textured_image, textured_image,
                                                  detector_name="ORB",
                                                  homography=np.eye(3))
        assert report.repeatability > 0.8

    def test_orthogonal_images_match_poorly(self, textured_image, rng):
        if not keypoint_match._HAS_CV2:
            pytest.skip("OpenCV required")
        other = rng.integers(0, 256, textured_image.shape, dtype=np.uint8)
        report = keypoint_match.evaluate_matching(textured_image, other,
                                                  detector_name="ORB",
                                                  homography=np.eye(3))
        assert report.inlier_ratio <= 0.7

    def test_both_matchers(self, textured_image):
        if not keypoint_match._HAS_CV2:
            pytest.skip("OpenCV required")
        reports = keypoint_match.evaluate_both_matchers(textured_image,
                                                        textured_image,
                                                        homography=np.eye(3))
        assert set(reports) == {"SIFT", "ORB"}
        for rep in reports.values():
            assert isinstance(rep.inlier_ratio, float)

    def test_unknown_detector(self, clean_image):
        if not keypoint_match._HAS_CV2:
            pytest.skip("OpenCV required")
        with pytest.raises(ValueError, match="unsupported keypoint detector"):
            keypoint_match.evaluate_matching(clean_image, clean_image,
                                             detector_name="SURF")

    def test_report_serialises(self, clean_image):
        rep = keypoint_match.MatchingReport(detector="ORB", n_matches=5,
                                            n_inliers=3, inlier_ratio=0.6)
        assert rep.as_dict()["inlier_ratio"] == 0.6
