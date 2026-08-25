"""Density clustering (DBSCAN) — roadmap Tier 2."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import cluster, dbscan
from fastumap._types import FloatArray, IntArray


def _blobs(n: int = 800, d: int = 8, k: int = 4, seed: int = 0) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * 12.0
    y = rng.integers(0, k, n)
    x = (centers[y] + rng.standard_normal((n, d))).astype(np.float64)
    return x, y.astype(np.int64)


def _purity(true: IntArray, pred: IntArray) -> float:
    mask = pred != -1
    if not mask.any():
        return 0.0
    total = sum(
        int(np.bincount(true[mask][pred[mask] == c]).max()) for c in set(pred[mask].tolist())
    )
    return total / int(mask.sum())


def test_dbscan_recovers_blob_count_without_being_told_k() -> None:
    x, _ = _blobs()
    labels = dbscan(x, min_samples=5)
    assert len(set(labels.tolist()) - {-1}) == 4


def test_dbscan_finds_non_convex_clusters_kmeans_cannot() -> None:
    datasets = pytest.importorskip("sklearn.datasets")
    xm, ym = datasets.make_moons(n_samples=600, noise=0.05, random_state=0)
    labels = dbscan(np.ascontiguousarray(xm, dtype=np.float64), eps=0.2, min_samples=5)
    assert len(set(labels.tolist()) - {-1}) == 2
    assert _purity(ym.astype(np.int64), labels) > 0.95


def test_dbscan_labels_outliers_as_noise() -> None:
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((300, 4)) * 0.3
    outliers = rng.standard_normal((20, 4)) * 20.0
    x = np.ascontiguousarray(np.vstack([dense, outliers]), dtype=np.float64)
    labels = dbscan(x, min_samples=5)
    assert (labels[-20:] == -1).mean() > 0.5  # most far-flung points flagged -1


def test_dbscan_is_deterministic() -> None:
    x, _ = _blobs(n=400)
    assert np.array_equal(dbscan(x, min_samples=5), dbscan(x, min_samples=5))


def test_dbscan_via_cluster_matches_standalone() -> None:
    x, _ = _blobs(n=400)
    assert np.array_equal(cluster(x, method="dbscan", min_samples=5), dbscan(x, min_samples=5))


def test_dbscan_rejects_return_centroids_and_bad_min_samples() -> None:
    x, _ = _blobs(n=200)
    with pytest.raises(ValueError, match="return_centroids is not supported"):
        cluster(x, method="dbscan", return_centroids=True)
    with pytest.raises(ValueError, match="min_samples"):
        dbscan(x, min_samples=0)
