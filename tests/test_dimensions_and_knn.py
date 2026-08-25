"""Output dimensions > 3, and precomputed-kNN input (roadmap Tier 1)."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import cluster, umap_project
from fastumap._types import FloatArray
from fastumap.neighbors import knn_graph


def _data(n: int = 300, d: int = 24, seed: int = 0) -> FloatArray:
    return np.random.default_rng(seed).standard_normal((n, d))


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_projects_to_any_dimension(dim: int) -> None:
    x = _data()
    xy = umap_project(x, dim, random_state=0)
    assert xy.shape == (x.shape[0], dim)
    assert bool(np.all(np.isfinite(xy)))


def test_reduce_to_ten_dims_then_cluster() -> None:
    # the target workflow: reduce to ~10D, then cluster there (umap-learn's own advice).
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 24)) * 8.0
    y = rng.integers(0, 5, 600)
    x = (centers[y] + rng.standard_normal((600, 24))).astype(np.float64)
    reduced = umap_project(x, 10, random_state=0)
    labels = cluster(reduced, method="kmeans", n_clusters=5, random_state=0)
    assert reduced.shape == (600, 10)
    assert len(np.unique(labels)) == 5


def test_dimensions_bounds() -> None:
    x = _data(d=8)
    with pytest.raises(ValueError, match="dimensions must be >= 1"):
        umap_project(x, 0)
    with pytest.raises(ValueError, match="cannot exceed n_features"):
        umap_project(x, 20)


def test_precomputed_knn_matches_builtin_bit_for_bit() -> None:
    # feeding fastumap's own kNN back in must reproduce the built-in path exactly.
    x = _data()
    builtin = umap_project(x, 2, n_neighbors=15, random_state=0)
    idx, dist = knn_graph(x, 15, metric="euclidean", mode="exact")
    pre = umap_project(x, 2, precomputed_knn=(idx, dist), random_state=0)
    assert np.array_equal(builtin, pre)


def test_precomputed_knn_validation() -> None:
    x = _data()
    idx, dist = knn_graph(x, 15, metric="euclidean", mode="exact")
    with pytest.raises(ValueError, match="rows"):
        umap_project(x, 2, precomputed_knn=(idx[:100], dist[:100]))
    with pytest.raises(ValueError, match="same shape"):
        umap_project(x, 2, precomputed_knn=(idx, dist[:, :5]))
    with pytest.raises(ValueError, match="not both"):
        umap_project(x, 2, precomputed_knn=(idx, dist), pca_dim=10)
