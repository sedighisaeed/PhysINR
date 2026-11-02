"""Feature matching and homography evaluation.

Underwater SLAM and SfM pipelines depend on repeatable, geometrically
verifiable keypoints. A restoration method that boosts UCIQE while destroying
SIFT repeatability is a regression for exactly the robotics applications UIE
claims to serve, so this module quantifies it directly.

Metrics
-------
* :func:`repeatability` -- fraction of keypoints detected in image A that
  have a corresponding keypoint in image B within a pixel tolerance, after
  warping by the known ground-truth homography.
* :func:`inlier_ratio` -- fraction of descriptor matches that survive RANSAC
  geometric verification with a homography model.
* :func:`matching_score` -- matches found normalised by the number of
  keypoints in the reference image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

DEFAULT_MATCHERS = ("SIFT", "ORB")


@dataclass
class MatchingReport:
    detector: str = ""
    n_keypoints_a: int = 0
    n_keypoints_b: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    repeatability: float = 0.0
    matching_score: float = 0.0
    mean_keypoint_response: float = 0.0
    homography_error: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        base = {k: v for k, v in self.__dict__.items() if k != "extra"}
        base.update(self.extra)
        return base


def _make_detector(name: str, n_features: int = 5000):
    name = name.upper()
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for keypoint evaluation")
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=n_features)
    if name == "ORB":
        return cv2.ORB_create(nfeatures=n_features)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"unsupported keypoint detector: {name} "
                     f"(supported: {DEFAULT_MATCHERS + ('AKAZE',)})")


def _matcher_for(detector_name: str):
    name = detector_name.upper()
    if name == "ORB":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def detect_and_describe(image: np.ndarray, detector_name: str,
                        n_features: int = 5000) -> Tuple[Any, Any]:
    """Grey-convert, detect, and describe. Returns (keypoints, descriptors)."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required")
    gray = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2GRAY)
    detector = _make_detector(detector_name, n_features)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return [], None
    return keypoints, descriptors


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    raise TypeError(f"unsupported image dtype: {arr.dtype}")


def _filter_matches(matches, ratio_test: float):
    """Lowe ratio test on the two nearest neighbours of each query point."""
    good = []
    if not matches:
        return good
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio_test * n.distance:
            good.append(m)
    return good


def _repeatability(kp_a, kp_b, homography: np.ndarray, shape: Tuple[int, int],
                   tolerance: float = 3.0) -> float:
    """Fraction of A's keypoints that survive into B under the known warp."""
    if not kp_a or not kp_b:
        return 0.0
    pts_a = np.float32([k.pt for k in kp_a]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts_a, homography).reshape(-1, 2)
    pts_b = np.float32([k.pt for k in kp_b]).reshape(-1, 2)
    h, w = shape[:2]
    inside = ((warped[:, 0] >= 0) & (warped[:, 0] < w)
              & (warped[:, 1] >= 0) & (warped[:, 1] < h))
    if not inside.any():
        return 0.0
    # Vectorised nearest-neighbour search among B's keypoints.
    diff = warped[inside][:, None, :] - pts_b[None, :, :]
    distances = np.linalg.norm(diff, axis=-1).min(axis=1)
    return float((distances <= tolerance).mean())


def evaluate_matching(image_a: np.ndarray, image_b: np.ndarray,
                      detector_name: str = "SIFT",
                      ratio_test: float = 0.75,
                      ransac_threshold: float = 5.0,
                      homography: Optional[np.ndarray] = None,
                      min_inliers_for_estimate: int = 8,
                      confidence: float = 0.999) -> MatchingReport:
    """Full matching evaluation between two views of the same scene.

    Parameters
    ----------
    image_a, image_b : np.ndarray
        Two views. For restoration benchmarking, ``image_a`` is the raw
        underwater frame and ``image_b`` is the enhanced one; the geometric
        relationship is identity, so pass ``homography=np.eye(3)``.
    homography : optional np.ndarray
        Ground-truth 3x3 warp mapping A into B. Required for a meaningful
        repeatability score.
    """
    report = MatchingReport(detector=detector_name)
    kp_a, desc_a = detect_and_describe(image_a, detector_name)
    kp_b, desc_b = detect_and_describe(image_b, detector_name)
    report.n_keypoints_a = len(kp_a)
    report.n_keypoints_b = len(kp_b)
    if desc_a is None or desc_b is None or not kp_a or not kp_b:
        return report

    matcher = _matcher_for(detector_name)
    knn_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good = _filter_matches(knn_matches, ratio_test)
    report.n_matches = len(good)
    report.matching_score = float(len(good) / max(1, len(kp_a)))
    report.mean_keypoint_response = float(
        np.mean([k.response for k in kp_b]))

    if homography is not None:
        report.repeatability = _repeatability(kp_a, kp_b, homography,
                                              image_a.shape)

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if len(good) >= min_inliers_for_estimate:
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                         ransac_threshold, confidence=confidence)
        if matrix is not None and mask is not None:
            inliers = np.asarray(mask).ravel().astype(bool)
            report.n_inliers = int(inliers.sum())
            report.inlier_ratio = float(report.n_inliers / max(1, len(good)))
            if homography is not None:
                # Frobenius distance between the estimated and true warp,
                # normalised so scale changes do not dominate.
                denom = max(np.linalg.norm(homography), 1e-9)
                report.homography_error = float(
                    np.linalg.norm(matrix - homography) / denom)
    return report


def evaluate_both_matchers(image_a: np.ndarray, image_b: np.ndarray,
                           detectors: Sequence[str] = DEFAULT_MATCHERS,
                           **kwargs) -> Dict[str, MatchingReport]:
    """Run SIFT and ORB (or any subset) and return one report per detector."""
    return {name: evaluate_matching(image_a, image_b, detector_name=name, **kwargs)
            for name in detectors}
