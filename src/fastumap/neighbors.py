"""k-nearest-neighbour graph by blocked brute force.

Exact kNN, computed one row-block at a time so the n-by-n distance matrix is never
materialised: at 5000 points that matrix would be 100MB and its argpartition index
another 200MB. Blocking bounds peak memory to block_size-by-n. Brute force is the right
call below ~10k points and needs no NN-descent dependency; above that, use the
approximate backend (roadmap #15).

Two metrics: ``euclidean`` (default) and ``cosine`` — the latter is what text/CLS
embeddings usually want, since euclidean on unnormalised encoder output is rarely the
structure you care about. ``knn_between`` finds neighbours of one set inside another,
which is what ``transform`` needs to place new points against a fitted training set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._types import FloatArray, IntArray


def _knn_from_tiles(
    n_query: int,
    n_ref: int,
    n_neighbors: int,
    block_size: int,
    tile: Callable[[int, int], FloatArray],
) -> tuple[IntArray, FloatArray]:
    """Assemble a kNN graph from per-block distance tiles.

    ``tile(start, stop)`` returns the non-negative ``(stop-start, n_ref)`` distances from
    query rows ``[start:stop]`` to every reference row. Returned distances are the metric's
    own (squared or not); callers transform them if needed. Ties break by index.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if n_neighbors > n_ref:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds reference size {n_ref}")

    indices = np.empty((n_query, n_neighbors), dtype=np.int64)
    distances = np.empty((n_query, n_neighbors), dtype=np.float64)
    for start in range(0, n_query, block_size):
        stop = min(start + block_size, n_query)
        block_dist = tile(start, stop)
        part = np.argpartition(block_dist, n_neighbors - 1, axis=1)[:, :n_neighbors]
        rows = np.arange(stop - start)[:, None]
        part_d = block_dist[rows, part]
        order = np.argsort(part_d, axis=1, kind="stable")
        indices[start:stop] = part[rows, order]
        distances[start:stop] = part_d[rows, order]
    return indices, distances


def _euclidean_tile(
    query: FloatArray, reference: FloatArray, q_sq: FloatArray, r_sq: FloatArray
) -> Callable[[int, int], FloatArray]:
    def tile(start: int, stop: int) -> FloatArray:
        block = query[start:stop]
        d2 = q_sq[start:stop, None] - 2.0 * (block @ reference.T) + r_sq[None, :]
        np.maximum(d2, 0.0, out=d2)  # kill negative round-off before the sqrt
        return d2

    return tile


def _unit_rows(matrix: FloatArray) -> FloatArray:
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    return matrix / np.where(norms > 0.0, norms, 1.0)[:, None]


def knn_euclidean(
    matrix: FloatArray, n_neighbors: int, *, block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Euclidean self-kNN. Column 0 is the point itself (distance 0)."""
    sq = np.einsum("ij,ij->i", matrix, matrix)
    n = matrix.shape[0]
    indices, d2_sorted = _knn_from_tiles(
        n, n, n_neighbors, block_size, _euclidean_tile(matrix, matrix, sq, sq)
    )
    return indices, np.sqrt(d2_sorted)


def knn_cosine(
    matrix: FloatArray, n_neighbors: int, *, block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Cosine self-kNN, distance = 1 - cosine similarity, in [0, 2]."""
    unit = _unit_rows(matrix)

    def tile(start: int, stop: int) -> FloatArray:
        dist = 1.0 - unit[start:stop] @ unit.T
        np.maximum(dist, 0.0, out=dist)
        return dist

    n = matrix.shape[0]
    return _knn_from_tiles(n, n, n_neighbors, block_size, tile)


def knn(
    matrix: FloatArray, n_neighbors: int, *, metric: str = "euclidean", block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Dispatch to the euclidean or cosine blocked-brute-force self-kNN."""
    if metric == "euclidean":
        return knn_euclidean(matrix, n_neighbors, block_size=block_size)
    if metric == "cosine":
        return knn_cosine(matrix, n_neighbors, block_size=block_size)
    raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")


def knn_between(
    query: FloatArray,
    reference: FloatArray,
    n_neighbors: int,
    *,
    metric: str = "euclidean",
    block_size: int = 512,
) -> tuple[IntArray, FloatArray]:
    """Neighbours of each ``query`` row inside ``reference`` (indices into reference).

    Used by ``transform``: query = new points, reference = fitted training set. No
    self-column, since query and reference are different sets.
    """
    n_q = query.shape[0]
    n_r = reference.shape[0]
    if metric == "euclidean":
        q_sq = np.einsum("ij,ij->i", query, query)
        r_sq = np.einsum("ij,ij->i", reference, reference)
        indices, d2_sorted = _knn_from_tiles(
            n_q, n_r, n_neighbors, block_size, _euclidean_tile(query, reference, q_sq, r_sq)
        )
        return indices, np.sqrt(d2_sorted)
    if metric == "cosine":
        q_unit = _unit_rows(query)
        r_unit = _unit_rows(reference)

        def tile(start: int, stop: int) -> FloatArray:
            dist = 1.0 - q_unit[start:stop] @ r_unit.T
            np.maximum(dist, 0.0, out=dist)
            return dist

        return _knn_from_tiles(n_q, n_r, n_neighbors, block_size, tile)
    raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
