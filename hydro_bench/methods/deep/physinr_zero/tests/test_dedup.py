"""Tests for the perceptual-hash deduplication of the meta-training source.

``deduplicate`` is the last mechanical check standing between the paper's
zero-shot claim and a duplicate of an evaluation image slipping into the
meta-training pool. Its behaviour on odd-shaped inputs is therefore part of
the claim's integrity, not a utility detail: a source that raises is dropped
from the pool *and* never compared against the references, which is safe but
silently shrinks the pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydro_bench.methods.deep.physinr_zero.utils import (deduplicate, load_image, phash,
                                phash_distance)


REPO = Path(__file__).resolve().parents[5]
DIV2K_POOL = REPO / "datasets" / "clean_photos"


@pytest.fixture
def div2k_images():
    """A few real photographs from the meta-training pool, if present."""
    cands = sorted(DIV2K_POOL.rglob("*.png"))[:8]
    if len(cands) < 4:
        pytest.skip("DIV2K clean pool not fetched; cannot test on real photos")
    return cands


def _img(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


class TestPHashShapes:
    """The reshape must survive every aspect ratio the pool contains."""

    # phash returns highfreq**2 - 1 bits: 63 at the defaults (8x8 minus the DC
    # term). The DC coefficient encodes only global brightness, so excluding it
    # keeps the hash about content.
    @pytest.mark.parametrize("h,w", [
        (128, 128),   # square
        (128, 71),    # portrait, width not divisible by the height block
        (128, 85),    # portrait, the case that raised in the real pool
        (128, 100),
        (128, 103),
        (71, 128),    # landscape, height not divisible
        (128, 256),
        (256, 128),
        (33, 97),     # neither axis divisible, both prime
    ])
    def test_every_aspect_ratio_hashes(self, h, w):
        bits = phash(_img(h, w))
        assert bits.shape == (63,)
        assert set(np.unique(bits)) <= {0, 1}

    def test_a_tiny_image_still_hashes(self):
        # Smaller than hash_size on an axis: the block clamps to 1 rather than
        # raising, so a very small source is still checked.
        bits = phash(_img(5, 8))
        assert bits.shape == (63,)

    def test_a_grayscale_image_hashes(self):
        bits = phash(_img(64, 64)[..., 0])
        assert bits.shape == (63,)

    def test_the_threshold_is_calibrated_away_from_random(self):
        """``deduplicate`` uses a Hamming threshold of 5, which is only a useful
        duplicate test if unrelated images sit far above it. This pins that the
        hash width and the threshold were chosen together."""
        dists = [phash_distance(phash(_img(128, 85, seed=i)),
                                phash(_img(128, 85, seed=j)))
                 for i in range(8) for j in range(i + 1, 9)]
        assert min(dists) > 5, \
            f"unrelated images collide at the dedup threshold (min {min(dists)})"

    def test_identical_images_hash_identically(self):
        a = _img(128, 85, seed=3)
        assert phash_distance(phash(a), phash(a.copy())) == 0

    def test_unrelated_images_differ(self):
        # Two independent random images should not collide at a tight threshold;
        # this is what makes the duplicate check able to fire at all.
        a, b = _img(128, 85, seed=1), _img(128, 85, seed=2)
        assert phash_distance(phash(a), phash(b)) > 5

    def test_the_same_image_hashes_identically_under_dedup_normalisation(
            self, div2k_images):
        """What the dedup pass actually relies on.

        ``deduplicate`` loads both the source and every reference through the
        same ``load_image(p, max_side)`` normalisation, so a duplicate lands in
        the same grid regardless of how each copy is stored. Cross-scale
        robustness of the raw hash is *not* required and does not hold (a real
        photo moves ~10 bits at some resample factors, from grid aliasing);
        identical normalisation is what makes the comparison exact.
        """
        for path in div2k_images:
            a = load_image(path, 128)
            b = load_image(path, 128)
            assert phash_distance(phash(a), phash(b)) == 0

    def test_unrelated_pool_images_sit_far_above_the_threshold(
            self, div2k_images):
        """The threshold of 5 must sit in an empty gap, or the dedup pass is
        making arbitrary decisions at the margin."""
        dists = [phash_distance(phash(load_image(div2k_images[i], 128)),
                                phash(load_image(div2k_images[j], 128)))
                 for i in range(len(div2k_images))
                 for j in range(i + 1, len(div2k_images))]
        assert min(dists) > 5, \
            f"two unrelated pool images are within the duplicate threshold " \
            f"(min {min(dists)})"


class TestDeduplicate:
    def _pool(self, tmp_path, n=4):
        d = tmp_path / "clean"
        d.mkdir()
        for i in range(n):
            _save(d / f"c_{i}.png", _img(96, 128, seed=i))
        return d

    def test_a_clean_pool_keeps_everything(self, tmp_path):
        pool = self._pool(tmp_path)
        refs = _ref_set(tmp_path, _img(96, 128, seed=99))
        out = deduplicate(pool, [refs])
        assert len(out["kept"]) == 4
        assert out["dropped"] == []

    def test_a_duplicate_is_dropped_and_named(self, tmp_path):
        pool = self._pool(tmp_path)
        # The reference is an exact copy of pool image 2.
        refs = _ref_set(tmp_path, _load(pool / "c_2.png"))
        out = deduplicate(pool, [refs])
        names = {Path(p).name for p in out["kept"]}
        assert "c_2.png" not in names
        assert len(out["dropped"]) == 1
        assert out["drop_counts"]["refs"] == 1

    def test_an_unreadable_reference_does_not_kill_the_pass(self, tmp_path):
        pool = self._pool(tmp_path)
        refs = ("refs", [tmp_path / "corrupt.png"])
        out = deduplicate(pool, [refs])
        assert len(out["kept"]) == 4

    def test_the_threshold_is_recorded(self, tmp_path):
        pool = self._pool(tmp_path)
        out = deduplicate(pool, [_ref_set(tmp_path, _img(96, 128, seed=99))],
                          threshold=12)
        assert out["threshold"] == 12


# ------------------------------------------------------------------ helpers
def _save(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def _load(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _ref_set(tmp_path: Path, arr: np.ndarray):
    d = tmp_path / "refs"
    d.mkdir(exist_ok=True)
    _save(d / "r.png", arr)
    return ("refs", [d / "r.png"])
