"""Fixture data generators for the test suite.

Covers the shapes the brief calls out: separated Gaussian clusters, a curved 1-D
manifold in high dimensions, collinear points, identical points, a disconnected kNN
graph, and n barely above the minimum.
"""

from __future__ import annotations

import numpy as np

from fastumap._types import FloatArray, IntArray


def gaussian_clusters(
    n: int = 600, n_clusters: int = 8, dim: int = 64, sep: float = 10.0, seed: int = 0
) -> tuple[FloatArray, IntArray]:
    """`n_clusters` well-separated isotropic Gaussians in `dim` dimensions."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    data = centers[labels] + rng.normal(size=(n, dim))
    return np.ascontiguousarray(data, dtype=np.float64), labels.astype(np.int64)


def curved_manifold(n: int = 400, dim: int = 50, seed: int = 0) -> FloatArray:
    """A 1-D swiss-roll-like curve embedded linearly into `dim` dimensions."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 3.0 * np.pi, n)
    base = np.stack([t * np.cos(t), t * np.sin(t), t], axis=1)
    projection = rng.normal(size=(3, dim))
    data = base @ projection + 0.01 * rng.normal(size=(n, dim))
    return np.ascontiguousarray(data, dtype=np.float64)


def collinear(n: int = 100, dim: int = 10, seed: int = 0) -> FloatArray:
    """Points on a single line through the origin."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim)
    t = np.linspace(-1.0, 1.0, n)[:, None]
    return np.ascontiguousarray(t * direction[None, :], dtype=np.float64)


def identical(n: int = 50, dim: int = 10) -> FloatArray:
    """All points at the same location — every pairwise distance is 0."""
    return np.ones((n, dim), dtype=np.float64)


def disconnected_blobs(n: int = 200, dim: int = 20, seed: int = 0) -> FloatArray:
    """Two blobs so far apart that no kNN edge crosses between them."""
    rng = np.random.default_rng(seed)
    first = rng.normal(size=(n // 2, dim))
    second = rng.normal(size=(n - n // 2, dim)) + 1.0e6
    return np.ascontiguousarray(np.vstack([first, second]), dtype=np.float64)


def barely_above_minimum(dim: int = 8, seed: int = 0) -> FloatArray:
    """n only just above what a 2-D projection needs (5 points)."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(5, dim)), dtype=np.float64)
