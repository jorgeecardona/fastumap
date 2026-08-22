"""k-nearest-neighbour graph by blocked brute force.

Exact kNN, computed one row-block at a time so the n-by-n distance matrix is never
materialised: at 5000 points that matrix would be 100MB and its argpartition index
another 200MB. Blocking bounds peak memory to block_size-by-n. Brute force is the right
call below ~10k points and needs no NN-descent dependency; above that, use the
approximate backend (roadmap #15).

Two metrics: ``euclidean`` (default) and ``cosine`` — the latter is what text/CLS
embeddings usually want, since euclidean on unnormalised encoder output is rarely the
structure you care about.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._types import FloatArray, IntArray


def _knn_from_tiles(
    n: int,
    n_neighbors: int,
    block_size: int,
    tile: Callable[[int, int], FloatArray],
) -> tuple[IntArray, FloatArray]:
    """Assemble the kNN graph from per-block distance tiles.

    ``tile(start, stop)`` returns the non-negative ``(stop-start, n)`` distances for that
    row block. The returned distances are the metric's own (squared or not) — callers
    transform them if needed. Ties break by index for determinism.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if n_neighbors > n:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds n={n}")

    indices = np.empty((n, n_neighbors), dtype=np.int64)
    distances = np.empty((n, n_neighbors), dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block_dist = tile(start, stop)
        part = np.argpartition(block_dist, n_neighbors - 1, axis=1)[:, :n_neighbors]
        rows = np.arange(stop - start)[:, None]
        part_d = block_dist[rows, part]
        order = np.argsort(part_d, axis=1, kind="stable")
        indices[start:stop] = part[rows, order]
        distances[start:stop] = part_d[rows, order]
    return indices, distances


def knn_euclidean(
    matrix: FloatArray, n_neighbors: int, *, block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Euclidean ``(indices, distances)``, each ``(n, n_neighbors)``, sorted ascending.

    Column 0 is the point itself (distance 0); callers that want strict neighbours drop it.
    """
    sq_norms = np.einsum("ij,ij->i", matrix, matrix)

    def tile(start: int, stop: int) -> FloatArray:
        block = matrix[start:stop]
        # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2, held to a (block, n) tile.
        d2 = sq_norms[start:stop, None] - 2.0 * (block @ matrix.T) + sq_norms[None, :]
        np.maximum(d2, 0.0, out=d2)  # kill negative round-off before the sqrt
        return d2

    indices, d2_sorted = _knn_from_tiles(matrix.shape[0], n_neighbors, block_size, tile)
    return indices, np.sqrt(d2_sorted)


def knn_cosine(
    matrix: FloatArray, n_neighbors: int, *, block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Cosine ``(indices, distances)``, distance = 1 - cosine similarity, in [0, 2].

    Rows are L2-normalised once; a zero row is left at the origin (its cosine distance to
    everything is 1). The block dot product then gives cosine similarity directly.
    """
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    unit = matrix / np.where(norms > 0.0, norms, 1.0)[:, None]

    def tile(start: int, stop: int) -> FloatArray:
        sim = unit[start:stop] @ unit.T  # (block, n) cosine similarity in [-1, 1]
        dist = 1.0 - sim
        np.maximum(dist, 0.0, out=dist)  # kill negative round-off (self-sim slightly > 1)
        return dist

    return _knn_from_tiles(matrix.shape[0], n_neighbors, block_size, tile)


def knn(
    matrix: FloatArray, n_neighbors: int, *, metric: str = "euclidean", block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Dispatch to the euclidean or cosine blocked-brute-force kNN."""
    if metric == "euclidean":
        return knn_euclidean(matrix, n_neighbors, block_size=block_size)
    if metric == "cosine":
        return knn_cosine(matrix, n_neighbors, block_size=block_size)
    raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
