"""Clustering, implemented in pure numpy + scipy — no new dependencies.

fastumap reimplements UMAP from scratch; clustering is the same. ``cluster()`` deliberately does
NOT compute a 2-D/3-D layout first (that layout is an iterative SGD and takes minutes at a million
points; clustering does not need it). Two methods, both ours:

- ``method="kmeans"`` — k-means (k-means++ seeding, Lloyd iterations, blocked assignment so the
  n-by-k distance matrix is never materialised). Fastest; you pick ``n_clusters``. This is the
  path aimed at a ~10 s budget at 1M points; when the numpy inner loop is the wall, the Rust
  accelerator is the place to speed it up (as it did for the SGD), not a third-party library.
- ``method="spectral"`` — spectral clustering: build the fuzzy kNN graph, take the leading
  normalised-adjacency eigenvectors (fastumap's own spectral machinery), and k-means those. Finds
  non-convex clusters k-means alone misses. Needs ``fastumap[ann]`` only above ~16k points, where
  the kNN switches to the approximate backend.

Everything here is float32-friendly and blocked, so it holds to the low-memory / low-CPU
constraints; the best shape at 1M (iteration count, a compiled k-means kernel) is a tuning
question to settle against a benchmark.
"""

from __future__ import annotations

import importlib

import numpy as np

from ._types import FloatArray, IntArray

_BLOCK = 4096  # rows per block in the assignment step; bounds peak memory to block-by-k
_ACCEL_MIN = 50_000  # below this the numpy path is fast and deterministic; above, use the accel

try:
    # Native k-means, built into the package as fastumap._accel: a matrixmultiply-GEMM + rayon
    # Lloyd that clusters 1M x 128 in ~10s (multi-core) vs the numpy path's minutes. Same API;
    # different (f32) results. Falls back to numpy only if the extension is absent (unbuilt checkout).
    _accel_kmeans = importlib.import_module("fastumap._accel").kmeans
except (ImportError, AttributeError):
    _accel_kmeans = None


def _unit_rows(x: FloatArray) -> FloatArray:
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    return x / np.where(norms > 0.0, norms, 1.0)[:, None]


def cluster(
    embeddings: FloatArray,
    *,
    method: str = "kmeans",
    n_clusters: int | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    n_iter: int = 50,
    random_state: int = 42,
) -> IntArray:
    """Cluster ``embeddings`` ``(n, d)`` into integer labels ``(n,)``.

    ``method`` is ``"kmeans"`` or ``"spectral"``; both take ``n_clusters``. ``metric`` is
    ``"euclidean"`` or ``"cosine"`` (cosine unit-normalises the rows). Deterministic given
    ``random_state``. Does NOT compute a layout — for a picture call ``umap_project`` separately.
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    if n_clusters is None:
        raise ValueError(f"method={method!r} requires n_clusters")
    work = _unit_rows(x) if metric == "cosine" else x
    if method == "kmeans":
        return kmeans(work, n_clusters, n_iter=n_iter, random_state=random_state)[0]
    if method == "spectral":
        return _spectral(work, n_clusters, n_neighbors, metric, n_iter, random_state)
    raise ValueError(f"unknown method {method!r}; use 'kmeans' or 'spectral'")


def _assign(x: FloatArray, centroids: FloatArray) -> IntArray:
    """Nearest-centroid label per row, computed one block at a time (bounded memory)."""
    n = x.shape[0]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, _BLOCK):
        block = x[start : start + _BLOCK]
        # argmin over ||b - c||^2 = ||b||^2 - 2 b.c + ||c||^2; the ||b||^2 term is constant per row.
        d2 = c_sq[None, :] - 2.0 * (block @ centroids.T)
        labels[start : start + _BLOCK] = np.argmin(d2, axis=1)
    return labels


def _kmeanspp_init(x: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    """k-means++ seeding: spread the initial centroids by distance-squared sampling."""
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    centroids[0] = x[rng.integers(n)]
    closest = np.sum((x - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        probs = closest / total if total > 0.0 else np.full(n, 1.0 / n)
        centroids[i] = x[rng.choice(n, p=probs)]
        closest = np.minimum(closest, np.sum((x - centroids[i]) ** 2, axis=1))
    return centroids


def _centroids_from_labels(x: FloatArray, labels: IntArray, k: int) -> FloatArray:
    """Per-cluster mean — the accel returns labels only, so recover centroids here."""
    dim = x.shape[1]
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    centroids = np.zeros((k, dim), dtype=np.float64)
    for d in range(dim):
        centroids[:, d] = np.bincount(labels, weights=x[:, d], minlength=k)
    nonempty = counts > 0
    centroids[nonempty] /= counts[nonempty, None]
    return centroids


def kmeans(
    x: FloatArray, n_clusters: int, *, n_iter: int = 50, random_state: int = 42
) -> tuple[IntArray, FloatArray]:
    """Lloyd's k-means with k-means++ init. Returns ``(labels, centroids)``.

    Uses the built-in native kernel (fastumap._accel) above ``_ACCEL_MIN`` points when present
    (~10s at 1M x 128 vs minutes for numpy); otherwise the pure-numpy path here, which is
    deterministic to the byte. Both converge to the same objective within the usual tolerance.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > x.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} exceeds n={x.shape[0]}")
    if _accel_kmeans is not None and x.shape[0] >= _ACCEL_MIN:
        data = np.ascontiguousarray(x, dtype=np.float64)
        labels = np.asarray(
            _accel_kmeans(data, n_clusters, n_iter, int(random_state)), dtype=np.int64
        )
        return labels, _centroids_from_labels(data, labels, n_clusters)
    rng = np.random.default_rng(random_state)
    centroids = _kmeanspp_init(x, n_clusters, rng)
    labels = _assign(x, centroids)
    for _ in range(n_iter):
        # new centroids = per-cluster mean; a bincount scatter, one pass over the dims.
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        sums = np.zeros_like(centroids)
        for d in range(x.shape[1]):
            sums[:, d] = np.bincount(labels, weights=x[:, d], minlength=n_clusters)
        nonempty = counts > 0
        new = centroids.copy()
        new[nonempty] = sums[nonempty] / counts[nonempty, None]  # keep empty clusters put
        moved = _assign(x, new)
        if np.array_equal(moved, labels) and np.allclose(new, centroids):
            centroids = new
            break
        centroids, labels = new, moved
    return labels, centroids


def _spectral(
    x: FloatArray, k: int, n_neighbors: int, metric: str, n_iter: int, random_state: int
) -> IntArray:
    """Spectral clustering: fuzzy kNN graph -> leading normalised eigenvectors -> k-means."""
    from . import _scipy
    from .graph import fuzzy_simplicial_set
    from .neighbors import knn_graph

    n = x.shape[0]
    kk = min(n_neighbors, n)
    indices, dists = knn_graph(x, kk, metric=metric, mode="auto")
    graph = fuzzy_simplicial_set(indices, dists)
    v0 = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
    # Leading (k) eigenvectors of the normalised adjacency — the same routine spectral_layout
    # uses, asking for k components instead of the embedding dimension.
    _, vecs = _scipy.normalized_adjacency_eigenvectors(
        n, graph.head, graph.tail, graph.weight, k, v0
    )
    embedding = _unit_rows(
        np.ascontiguousarray(vecs, dtype=np.float64)
    )  # row-normalise (Ng-Jordan-Weiss)
    return kmeans(embedding, k, n_iter=n_iter, random_state=random_state)[0]
