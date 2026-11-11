"""``cluster(pca_dim=...)``: cluster in a PCA-reduced space to cut the kNN and k-means cost."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import assign_clusters, cluster


def _blobs(n: int = 600, d: int = 120, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((k, d)) * 6.0
    truth = rng.integers(0, k, n)
    return centers[truth] + rng.standard_normal((n, d)), truth


def _agree(a: np.ndarray, b: np.ndarray) -> bool:
    # Same partition up to a relabelling: every predicted cluster maps to one true cluster.
    pairs = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("method", ["kmeans", "spectral", "dbscan", "hdbscan"])
def test_pca_dim_recovers_blobs(method: str) -> None:
    x, truth = _blobs()
    labels = cluster(x, method=method, n_clusters=4, pca_dim=16)
    assert labels.shape == (600,)
    kept = labels >= 0
    assert kept.mean() > 0.75  # dbscan auto-eps marks the blob edges as noise
    assert _agree(labels[kept], truth[kept])


def test_pca_dim_none_and_too_large_are_noops() -> None:
    x, _ = _blobs()
    plain = cluster(x, n_clusters=4)
    assert np.array_equal(plain, cluster(x, n_clusters=4, pca_dim=None))
    assert np.array_equal(plain, cluster(x, n_clusters=4, pca_dim=500))


def test_pca_dim_centroids_live_in_input_space() -> None:
    x, _ = _blobs()
    result = cluster(x, n_clusters=4, pca_dim=16, return_centroids=True)
    assert result.centroids.shape == (4, 120)
    assert np.array_equal(assign_clusters(x, result.centroids), result.labels)


def test_pca_dim_rejects_bad_values() -> None:
    x, _ = _blobs()
    with pytest.raises(ValueError, match="pca_dim"):
        cluster(x, n_clusters=4, pca_dim=0)
    with pytest.raises(ValueError, match="return_centroids"):
        cluster(x, method="hdbscan", pca_dim=16, return_centroids=True)
