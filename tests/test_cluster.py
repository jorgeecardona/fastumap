"""Clustering — our numpy k-means and spectral, no external clustering libraries."""

import numpy as np
import pytest

from fastumap import cluster
from fastumap.clustering import kmeans


def _blobs(n: int = 1500, d: int = 20, k: int = 5, spread: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * spread
    y = rng.integers(0, k, n)
    x = (centers[y] + rng.standard_normal((n, d))).astype(np.float64)
    return x, y


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    total = sum(int(np.bincount(true[pred == c]).max()) for c in np.unique(pred))
    return total / len(true)


@pytest.mark.parametrize("method", ["kmeans", "spectral"])
def test_recovers_separated_blobs(method: str) -> None:
    x, y = _blobs()
    labels = cluster(x, method=method, n_clusters=5, random_state=0)
    assert labels.shape == (x.shape[0],)
    assert _purity(y, labels) > 0.95


def test_kmeans_is_deterministic() -> None:
    x, _ = _blobs()
    a = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    b = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    assert np.array_equal(a, b)


def test_cosine_metric_runs() -> None:
    x, _ = _blobs()
    labels = cluster(x, method="kmeans", n_clusters=5, metric="cosine", random_state=0)
    assert labels.shape == (x.shape[0],)


def test_kmeans_standalone_returns_centroids() -> None:
    x, y = _blobs()
    labels, centroids = kmeans(x, 5, random_state=0)
    assert centroids.shape == (5, x.shape[1])
    assert _purity(y, labels) > 0.95


def test_no_n_clusters_raises() -> None:
    x, _ = _blobs(n=100)
    with pytest.raises(ValueError, match="n_clusters"):
        cluster(x, method="kmeans")


def test_bad_method_raises() -> None:
    x, _ = _blobs(n=100)
    with pytest.raises(ValueError, match="unknown method"):
        cluster(x, method="dbscan", n_clusters=3)


def test_too_many_clusters_raises() -> None:
    x, _ = _blobs(n=10)
    with pytest.raises(ValueError, match="exceeds"):
        cluster(x, method="kmeans", n_clusters=20)


def test_accel_path_recovers_clusters() -> None:
    # n >= _ACCEL_MIN (50k) routes k-means through the native fastumap-accel kernel when present;
    # it gives different (f32) labels than numpy but must still recover well-separated clusters.
    pytest.importorskip("fastumap_accel")
    x, y = _blobs(n=60_000, d=32, k=8)
    labels = cluster(x, method="kmeans", n_clusters=8, random_state=0)
    assert labels.shape == (60_000,)
    assert _purity(y, labels) > 0.95
