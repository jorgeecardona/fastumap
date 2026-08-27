"""k-nearest-neighbour graph by blocked brute force.

Exact kNN, computed one row-block at a time so the n-by-n distance matrix is never
materialised. At 5000 points that matrix would be 100MB and its argpartition index another
200MB. Blocking bounds peak memory to block_size-by-n. Brute force is the right call below ~10k
points and needs no NN-descent dependency. Above that, use the approximate backend (roadmap #15).

Two metrics: ``euclidean`` (default) and ``cosine``. Cosine is what text/CLS embeddings usually
want, since euclidean on unnormalised encoder output is rarely the structure you care about.
``knn_between`` finds neighbours of one set inside another. That is what ``transform`` needs to
place new points against a fitted training set.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable

import numpy as np

from ._types import FloatArray, IntArray
from ._types import unit_rows as _unit_rows

# Below this many points the exact blocked kNN is still fast (and bit-identical), so `knn="auto"`
# keeps it. Above it the O(n^2) kNN dominates and approximate wins clearly (measured ~2.4x at
# 20k, ~2.8x at 30k), so auto switches to the fastumap[ann] backend when it is installed.
_ANN_MIN = 16384


def _knn_from_tiles(
    n_query: int,
    n_ref: int,
    n_neighbors: int,
    block_size: int,
    tile: Callable[[int, int], FloatArray],
) -> tuple[IntArray, FloatArray]:
    """Assemble a kNN graph from per-block distance tiles.

    ``tile(start, stop)`` returns the non-negative ``(stop-start, n_ref)`` distances from query
    rows ``[start:stop]`` to every reference row. Returned distances are the metric's own (squared
    or not). Callers transform them if needed. Ties break by index.
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


def ann_available() -> bool:
    """Whether the approximate-kNN backend (``pip install fastumap[ann]`` → faiss) is present."""
    return importlib.util.find_spec("faiss") is not None


def knn_approx(
    matrix: FloatArray,
    n_neighbors: int,
    *,
    metric: str = "euclidean",
    m: int = 16,
    ef_construction: int = 200,
) -> tuple[IntArray, FloatArray]:
    """Approximate self-kNN via faiss HNSW, for inputs where exact O(n^2) is too slow.

    Output matches the exact path's contract: euclidean distance for ``"euclidean"``,
    ``1 - cosine similarity`` for ``"cosine"``, self in column 0 (distance 0), nearest first.
    Deterministic run-to-run (single-threaded build). Requires ``pip install fastumap[ann]``. It
    raises :class:`ImportError` with that hint otherwise.
    """
    if metric == "cosine":
        data = _unit_rows(matrix)  # squared-L2 on unit rows = 2*(1 - cos), so cosine dist = d2/2
    elif metric == "euclidean":
        data = matrix
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")

    try:
        faiss = importlib.import_module("faiss")  # dynamic: no stubs, optional at runtime
    except ImportError as exc:  # pragma: no cover - exercised via the dispatcher's error path
        raise ImportError(
            "approximate kNN needs the faiss backend; install it with `pip install fastumap[ann]`"
        ) from exc

    faiss.omp_set_num_threads(1)  # single-threaded build + search -> deterministic
    contiguous = np.ascontiguousarray(data, dtype=np.float32)
    dim = contiguous.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)  # L2 space
    index.hnsw.efConstruction = ef_construction
    index.add(contiguous)
    index.hnsw.efSearch = max(ef_construction, n_neighbors + 1)
    d2, labels = index.search(contiguous, n_neighbors)  # squared L2, ascending; self at col 0

    indices: IntArray = labels.astype(np.int64)
    d2_pos = np.maximum(d2.astype(np.float64), 0.0)
    distances = np.sqrt(d2_pos) if metric == "euclidean" else d2_pos / 2.0
    return indices, distances


def knn_graph(
    matrix: FloatArray,
    n_neighbors: int,
    *,
    metric: str = "euclidean",
    mode: str = "auto",
    block_size: int = 512,
) -> tuple[IntArray, FloatArray]:
    """Self-kNN, exact or approximate per ``mode``.

    - ``"exact"`` always uses the blocked brute force (bit-identical, the default).
    - ``"approx"`` always uses the faiss HNSW backend (errors if ``fastumap[ann]`` is absent).
    - ``"auto"`` uses approximate only when the backend is installed *and* ``n >= 16384``.
      Otherwise exact, so small inputs stay bit-identical and nothing changes unless you opted
      into the extra.
    """
    if mode == "exact":
        return knn(matrix, n_neighbors, metric=metric, block_size=block_size)
    if mode == "approx":
        return knn_approx(matrix, n_neighbors, metric=metric)
    if mode == "auto":
        if matrix.shape[0] >= _ANN_MIN and ann_available():
            return knn_approx(matrix, n_neighbors, metric=metric)
        return knn(matrix, n_neighbors, metric=metric, block_size=block_size)
    raise ValueError(f"unknown knn mode {mode!r}; use 'auto', 'exact', or 'approx'")
