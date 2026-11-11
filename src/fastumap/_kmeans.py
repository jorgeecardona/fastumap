"""k-means internals: the native kernel calls, blocked nearest-centroid assignment, inertia.

The public functions (``kmeans``, ``cluster``, ``assign_clusters``, ``choose_k``) live in
``fastumap.clustering`` and call into this module. Tests patch names here, not there.
"""

from __future__ import annotations

import numpy as np

from ._native import load_accel
from ._types import FloatArray, IntArray

BLOCK = 4096  # rows per block in the assignment step; bounds peak memory to block-by-k
# Above this, full Lloyd's per-iteration O(n*k*d) pass is the wall on a small container, so the
# default switches to mini-batch. Chosen by n ALONE, never by core count: gating on cores would
# make the same input and seed give different labels on different machines, which is exactly the
# guarantee this library sells. batch_size=0 forces full Lloyd back on at any size.
MINIBATCH_MIN = 200_000
MINIBATCH_BATCH = 10_000  # rows per iteration; ~2 full passes of total work at the default n_iter
MINIBATCH_ITER = 100  # mini-batch needs many cheap passes where Lloyd needs few expensive ones

# Native k-means, built into the package as fastumap._accel: a matrixmultiply-GEMM + rayon Lloyd.
# Clusters 1M x 128 in ~10s (multi-core). Mini-batch (Sculley 2010) is its CPU-constrained
# sibling; see the batch_size docs on `kmeans`. A missing extension raises at import.
_accel = load_accel()
_accel_kmeans = _accel.kmeans
_accel_minibatch = _accel.minibatch_kmeans


def inertia(x: FloatArray, labels: IntArray, centroids: FloatArray) -> float:
    """k-means objective: sum of squared distances to the assigned centroid, one block at a time
    so the (n, d) difference is never materialised all at once."""
    total = 0.0
    for start in range(0, x.shape[0], BLOCK):
        block = x[start : start + BLOCK]
        diff = block - centroids[labels[start : start + BLOCK]]
        total += float(np.einsum("ij,ij->", diff, diff))
    return total


def dispersion_and_sizes(
    x: FloatArray, labels: IntArray, centroids: FloatArray, k: int
) -> tuple[FloatArray, IntArray]:
    """Per-cluster mean distance to the assigned centroid, and per-cluster point counts. Blocked
    like ``inertia`` (the (n, d) difference is never materialised at once). An empty cluster gets
    dispersion 0.0."""
    sizes = np.bincount(labels, minlength=k).astype(np.int64)
    dist_sums = np.zeros(k, dtype=np.float64)
    for start in range(0, x.shape[0], BLOCK):
        lbl = labels[start : start + BLOCK]
        diff = x[start : start + BLOCK] - centroids[lbl]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dist_sums += np.bincount(lbl, weights=dist, minlength=k)
    dispersion = np.divide(dist_sums, sizes, out=np.zeros(k, dtype=np.float64), where=sizes > 0)
    return dispersion, sizes


def assign(x: FloatArray, centroids: FloatArray) -> IntArray:
    """Nearest-centroid label per row, computed one block at a time (bounded memory)."""
    n = x.shape[0]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, BLOCK):
        block = x[start : start + BLOCK]
        # argmin over ||b - c||^2 = ||b||^2 - 2 b.c + ||c||^2; the ||b||^2 term is constant per row.
        d2 = c_sq[None, :] - 2.0 * (block @ centroids.T)
        labels[start : start + BLOCK] = np.argmin(d2, axis=1)
    return labels


def assign_with_distances(x: FloatArray, centroids: FloatArray) -> tuple[IntArray, FloatArray]:
    """Nearest-centroid label AND the distance to it, per row, one block at a time. Unlike
    ``assign`` this keeps the ``||b||^2`` term (it does not affect the argmin, but it is needed for
    the actual distance a caller can threshold on)."""
    n = x.shape[0]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    labels = np.empty(n, dtype=np.int64)
    nearest_d2 = np.empty(n, dtype=np.float64)
    for start in range(0, n, BLOCK):
        block = x[start : start + BLOCK]
        b_sq = np.einsum("ij,ij->i", block, block)
        d2 = b_sq[:, None] + c_sq[None, :] - 2.0 * (block @ centroids.T)
        np.maximum(d2, 0.0, out=d2)  # clip negative round-off before the sqrt
        idx = np.argmin(d2, axis=1)
        labels[start : start + BLOCK] = idx
        nearest_d2[start : start + BLOCK] = d2[np.arange(idx.shape[0]), idx]
    return labels, np.sqrt(nearest_d2)


def centroids_from_labels(x: FloatArray, labels: IntArray, k: int) -> FloatArray:
    """Per-cluster mean — the accel returns labels only, so recover centroids here."""
    from scipy.sparse import csr_matrix  # lazy: keeps import light

    n = x.shape[0]
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    # One sparse (k, n) membership matrix times x: one pass over the data, not one per dimension.
    member = csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
    centroids = np.asarray(member @ x, dtype=np.float64)
    nonempty = counts > 0
    centroids[nonempty] /= counts[nonempty, None]
    return centroids


def kmeans_core(
    x: FloatArray,
    n_clusters: int,
    *,
    n_iter: int = 50,
    random_state: int = 42,
    batch_size: int | None = None,
    n_init: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """k-means without the BLAS-cap decorator, so it can be called concurrently (e.g. from
    choose_k's parallel sweep) under a cap the caller has already set, with no nested and
    thread-unsafe re-entry into ``threadpool_limits``.

    ``n_init`` runs the fit that many times from different seeds and keeps the lowest-inertia
    result. ``None`` (default) is a single run: fast, and enough for full Lloyd and for mini-batch
    on the overlapping data real embeddings produce. Raise it for the mini-batch path on
    well-separated clusters, where its running-mean update can settle into a bad optimum a different
    seed escapes (n_init=5 recovers full-Lloyd parity on the adversarial blobs). Deterministic: the
    seeds are ``random_state + i`` and the best is chosen by inertia."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_init is not None and n_init < 1:
        raise ValueError("n_init must be >= 1")
    inits = 1 if n_init is None else n_init
    if inits == 1:
        return kmeans_single(
            x, n_clusters, n_iter=n_iter, random_state=random_state, batch_size=batch_size
        )
    best: tuple[IntArray, FloatArray] | None = None
    best_inertia = float("inf")
    for i in range(inits):
        labels, centroids = kmeans_single(
            x, n_clusters, n_iter=n_iter, random_state=random_state + i, batch_size=batch_size
        )
        val = inertia(x, labels, centroids)
        if val < best_inertia:
            best_inertia, best = val, (labels, centroids)
    assert best is not None
    return best


def kmeans_single(
    x: FloatArray,
    n_clusters: int,
    *,
    n_iter: int = 50,
    random_state: int = 42,
    batch_size: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """One k-means fit on the native kernel: mini-batch above ``MINIBATCH_MIN`` rows, else Lloyd."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > x.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} exceeds n={x.shape[0]}")
    if batch_size is not None and batch_size < 0:
        raise ValueError("batch_size must be >= 0 (0 forces full Lloyd, None picks by n)")
    n = x.shape[0]
    auto_minibatch = batch_size is None and n >= MINIBATCH_MIN
    if auto_minibatch or (batch_size or 0) > 0:
        batch = MINIBATCH_BATCH if batch_size in (None, 0) else batch_size
        # Mini-batch trades an expensive pass for many cheap ones, so it wants more iterations
        # than Lloyd. Only override a caller who left n_iter at its Lloyd default.
        iters = MINIBATCH_ITER if n_iter == 50 else n_iter
        data = np.ascontiguousarray(x, dtype=np.float64)
        labels = np.asarray(
            _accel_minibatch(data, n_clusters, iters, int(batch), int(random_state)),
            dtype=np.int64,
        )
        return nearest_centroid_fit(data, labels, n_clusters)
    # Always the native kernel, at any size. A size gate here cost callers seconds to a minute
    # whenever a subsample or a dimension count put them under it (choose_k on 5,000 x 768 rows
    # took 70 s), to save about 1 ms on tiny inputs (#29).
    data = np.ascontiguousarray(x, dtype=np.float64)
    labels = np.asarray(_accel_kmeans(data, n_clusters, n_iter, int(random_state)), dtype=np.int64)
    return nearest_centroid_fit(data, labels, n_clusters)


def nearest_centroid_fit(
    data: FloatArray, labels: IntArray, n_clusters: int
) -> tuple[IntArray, FloatArray]:
    """Return centroids as the label means, and labels as the nearest of those centroids.

    The kernel works in float32 and may stop before convergence, so a few of its labels are not
    the nearest float64 centroid. One more assignment pass fixes that: the returned labels are
    then exactly what ``assign_clusters`` gives for the returned centroids. The centroids are not
    recomputed after it.
    """
    centroids = centroids_from_labels(data, labels, n_clusters)
    return assign(data, centroids), centroids
