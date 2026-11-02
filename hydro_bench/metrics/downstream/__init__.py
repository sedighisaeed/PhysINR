"""Application-driven downstream evaluation.

The load-bearing claim for reviewers: *does restoration improve utility in a
real robotic task?* A method that gains 2 dB PSNR but breaks a detector's
recall is not an improvement for an AUV.

Protocol
--------
The detector / segmenter is **frozen** and was trained only on raw imagery
(``frozen_raw``). Enhanced images are scored by the same frozen weights, so
any delta measures utility the enhancement adds rather than model capacity.
Enhanced images never enter training, which is what keeps the comparison
honest.

Modules
-------
* :mod:`object_detection` -- COCO-style mAP@[.5:.95] and mAP@50, with a
  self-contained evaluator so results do not depend on the COCO toolkit.
* :mod:`segmentation` -- mIoU and per-class IoU on SUIM.
* :mod:`keypoint_match` -- SIFT/ORB repeatability and RANSAC inlier ratio
  under geometric verification.
"""

__all__ = ["object_detection", "segmentation", "keypoint_match"]
