"""Public projection entry points: ``umap_project`` and ``spectral_project``."""

from __future__ import annotations

import numpy as np

from ._types import FloatArray
from .graph import find_ab_params, fuzzy_simplicial_set
from .layout import optimize_layout, spectral_layout
from .neighbors import knn_euclidean


def _default_epochs(n: int) -> int:
    # umap-learn's rule: more epochs when the graph is small enough to afford them.
    return 500 if n <= 10_000 else 200


def _validate(matrix: FloatArray, dimensions: int) -> FloatArray:
    x = np.ascontiguousarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D (n_samples, n_features)")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 points to project")
    return x


def umap_project(
    matrix: FloatArray,
    dimensions: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int | None = None,
    negative_sample_rate: int = 5,
    random_state: int = 42,
    metric: str = "euclidean",
) -> FloatArray:
    """Project ``matrix`` (n_samples, n_features) into ``dimensions`` (2 or 3).

    Deterministic given ``random_state``: the spectral start vector is pinned and the
    negative sampler is seeded, so the same input and seed give bit-identical output.
    """
    if metric != "euclidean":
        raise NotImplementedError(f"metric={metric!r} not supported yet; only 'euclidean'")
    x = _validate(matrix, dimensions)
    n = x.shape[0]
    k = min(n_neighbors, n)

    knn_indices, knn_dists = knn_euclidean(x, k)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    rng = np.random.default_rng(random_state)
    init = spectral_layout(graph, dimensions, rng=rng)
    a, b = find_ab_params(spread, min_dist)
    epochs = _default_epochs(n) if n_epochs is None else n_epochs
    return optimize_layout(
        init,
        graph,
        epochs,
        a=a,
        b=b,
        rng=rng,
        negative_sample_rate=negative_sample_rate,
    )


def spectral_project(
    matrix: FloatArray,
    dimensions: int = 2,
    *,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> FloatArray:
    """Just the spectral initialisation: kNN graph, fuzzy set, spectral embedding.

    Useful on its own as a fast, deterministic layout, and as the seed the full
    optimisation refines.
    """
    x = _validate(matrix, dimensions)
    n = x.shape[0]
    k = min(n_neighbors, n)
    knn_indices, knn_dists = knn_euclidean(x, k)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    rng = np.random.default_rng(random_state)
    return spectral_layout(graph, dimensions, rng=rng)
