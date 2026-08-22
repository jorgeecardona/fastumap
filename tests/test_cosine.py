"""Cosine metric: end-to-end, deterministic, and better than euclidean where direction
(not magnitude) carries the structure — the text/CLS-embedding case.
"""

from __future__ import annotations

import numpy as np

from fastumap import spectral_project, umap_project
from fastumap._types import FloatArray, IntArray
from fastumap.neighbors import knn_cosine


def directional_clusters(
    n: int = 600, n_clusters: int = 6, dim: int = 50, seed: int = 0
) -> tuple[FloatArray, IntArray]:
    """Clusters defined by direction, with magnitudes spanning two orders of magnitude.

    Same-cluster points share a direction but have wildly different lengths, so euclidean
    distance is dominated by magnitude (mixing clusters) while cosine sees the clusters.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_clusters, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    labels = rng.integers(0, n_clusters, size=n)
    directions = centroids[labels] + 0.05 * rng.normal(size=(n, dim))
    magnitudes = rng.uniform(0.1, 10.0, size=(n, 1))
    return np.ascontiguousarray(directions * magnitudes, dtype=np.float64), labels.astype(np.int64)


def _label_purity(emb: FloatArray, labels: IntArray, k: int) -> float:
    from fastumap.neighbors import knn_euclidean

    idx, _ = knn_euclidean(emb, k + 1)
    n = emb.shape[0]
    total = 0.0
    for i in range(n):
        neighbours = idx[i][idx[i] != i][:k]
        total += float(np.mean(labels[neighbours] == labels[i]))
    return total / n


def test_knn_cosine_self_is_nearest_and_distances_in_range() -> None:
    x, _ = directional_clusters(n=200)
    idx, dist = knn_cosine(x, 10)
    assert idx.shape == (200, 10)
    assert bool(np.all(dist >= 0.0)) and bool(np.all(dist <= 2.0 + 1e-9))
    # each point is its own nearest neighbour at cosine distance ~0
    assert bool(np.all(idx[:, 0] == np.arange(200)))
    assert bool(np.all(dist[:, 0] < 1e-9))


def test_cosine_projection_deterministic() -> None:
    x, _ = directional_clusters(n=300)
    first = umap_project(x, 2, metric="cosine", random_state=1, n_epochs=100)
    second = umap_project(x, 2, metric="cosine", random_state=1, n_epochs=100)
    assert np.array_equal(first, second)
    assert bool(np.all(np.isfinite(first)))


def test_cosine_beats_euclidean_on_directional_clusters() -> None:
    x, labels = directional_clusters(n=600, seed=0)
    cosine_emb = umap_project(x, 2, metric="cosine", random_state=0, n_epochs=200)
    euclidean_emb = umap_project(x, 2, metric="euclidean", random_state=0, n_epochs=200)
    cosine_purity = _label_purity(cosine_emb, labels, 15)
    euclidean_purity = _label_purity(euclidean_emb, labels, 15)
    assert cosine_purity > euclidean_purity


def test_unknown_metric_rejected() -> None:
    x, _ = directional_clusters(n=100)
    import pytest

    with pytest.raises(ValueError, match="unknown metric"):
        umap_project(x, 2, metric="manhattan")
    with pytest.raises(ValueError, match="unknown metric"):
        spectral_project(x, 2, metric="manhattan")
