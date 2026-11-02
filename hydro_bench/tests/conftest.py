"""Shared fixtures for the HydroBench test suite."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from hydro_bench.scripts.make_synthetic_benchmark import generate  # noqa: E402


@pytest.fixture(scope="session")
def synth_dataset(tmp_path_factory) -> Path:
    """A small synthetic paired dataset, generated once for the whole session."""
    root = tmp_path_factory.mktemp("synth_data")
    # generate() already appends the "SynthUIB" subdirectory.
    return generate(root=root, n_train=4, n_val=2, n_test=6,
                    width=64, height=48, seed=7)


@pytest.fixture(scope="session")
def synth_root(synth_dataset, tmp_path_factory) -> Path:
    """The HYDROBENCH_DATA base directory containing ``synth_dataset``."""
    base = tmp_path_factory.mktemp("hydro_data")
    shutil.copytree(synth_dataset, base / "SynthUIB")
    return base


@pytest.fixture(autouse=True)
def hydrobench_env(synth_root, monkeypatch):
    """Point every test at the session's synthetic dataset."""
    monkeypatch.setenv("HYDROBENCH_DATA", str(synth_root))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def clean_image(rng) -> np.ndarray:
    return rng.integers(40, 220, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def noisy_image(clean_image, rng) -> np.ndarray:
    noise = rng.integers(-25, 26, clean_image.shape)
    return np.clip(clean_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def hazy_image(rng) -> np.ndarray:
    """A blue-cast, low-contrast image: what an underwater frame looks like."""
    base = np.full((64, 64, 3), (62, 104, 124), np.int16)
    return np.clip(base + rng.integers(-10, 11, (64, 64, 3)),
                   0, 255).astype(np.uint8)


@pytest.fixture
def textured_image(rng) -> np.ndarray:
    """A structured, high-contrast image with enough texture for keypoint detection.

    Random noise gives ORB almost nothing to latch onto, so keypoint tests need
    real edges and corners.
    """
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:96, 0:96]
    # Checkerboard + discs: plenty of corners at multiple scales.
    image[(yy // 16 + xx // 16) % 2 == 0] = (200, 180, 160)
    for _ in range(4):
        cx, cy = rng.integers(16, 80, 2)
        r = int(rng.integers(6, 14))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        image[mask] = rng.integers(0, 120, 3)
    return image
