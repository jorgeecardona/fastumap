"""choose_k on high-dimensional blobs: same k at every seed, inertia never rises with k (#30)."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import _kmeans
from fastumap.clustering import choose_k, kmeans


def _unit_blobs(n: int = 4000, d: int = 384, k: int = 7) -> np.ndarray:
    # Noise per dimension close to the centre spread: within- and between-cluster distances are
    # near each other, the case where plain k-means++ seeds two centroids in one cluster.
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(k, d))
    x = centers[rng.integers(0, k, size=n)] + 0.9 * rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", [42, 0, 7])
def test_choose_k_finds_planted_k_and_curve_never_rises(seed: int) -> None:
    result = choose_k(
        _unit_blobs(), k_min=2, k_max=14, metric="cosine", random_state=seed, return_evidence=True
    )
    assert result.k == 7
    assert result.inertia_rises == 0


def test_kmeans_uses_every_cluster() -> None:
    labels, _ = kmeans(_unit_blobs(), 12, batch_size=0, random_state=42)
    assert len(np.unique(labels)) == 12


def test_empty_cluster_attracts_no_points() -> None:
    # Cluster 2 is empty: its all-zero centroid must not win the final assignment.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8)) + 5.0
    labels = (np.arange(200) % 2).astype(np.int64)
    new_labels, centroids = _kmeans.nearest_centroid_fit(x, labels, 3)
    assert not np.any(new_labels == 2)
    assert centroids.shape == (3, 8)
