"""k-nearest-neighbour graph by blocked brute force.

Exact euclidean kNN, computed one row-block at a time so the n-by-n distance
matrix is never materialised: at 5000 points that matrix would be 100MB and its
argpartition index another 200MB. Blocking bounds peak memory to block_size-by-n.
Brute force is the right call below ~10k points and needs no NN-descent dependency;
above that, use the approximate backend (roadmap #15).
"""

from __future__ import annotations

import numpy as np

from ._types import FloatArray, IntArray


def knn_euclidean(
    matrix: FloatArray, n_neighbors: int, *, block_size: int = 512
) -> tuple[IntArray, FloatArray]:
    """Return ``(indices, distances)``, each ``(n, n_neighbors)``, sorted ascending.

    Column 0 is the point itself (distance 0); callers that want strict neighbours
    drop it. Ties break by index for determinism.
    """
    n = matrix.shape[0]
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if n_neighbors > n:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds n={n}")

    sq_norms = np.einsum("ij,ij->i", matrix, matrix)
    indices = np.empty((n, n_neighbors), dtype=np.int64)
    distances = np.empty((n, n_neighbors), dtype=np.float64)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = matrix[start:stop]
        # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2, held to a (block, n) tile.
        d2 = sq_norms[start:stop, None] - 2.0 * (block @ matrix.T) + sq_norms[None, :]
        np.maximum(d2, 0.0, out=d2)  # kill negative round-off before the sqrt
        part = np.argpartition(d2, n_neighbors - 1, axis=1)[:, :n_neighbors]
        rows = np.arange(stop - start)[:, None]
        part_d = d2[rows, part]
        order = np.argsort(part_d, axis=1, kind="stable")
        indices[start:stop] = part[rows, order]
        distances[start:stop] = np.sqrt(part_d[rows, order])

    return indices, distances
