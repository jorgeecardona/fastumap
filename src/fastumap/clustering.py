"""Clustering, implemented ourselves. numpy + scipy plus a built-in native k-means kernel. No
third-party clustering library.

fastumap reimplements UMAP from scratch. Clustering is the same. ``cluster()`` does NOT compute
a 2-D/3-D layout first. That layout is an iterative SGD and takes minutes at a million points.
Clustering does not need it.

Four methods, all ours:

- ``method="kmeans"`` — k-means (k-means++ seeding, Lloyd iterations, blocked assignment so the
  n-by-k distance matrix is never materialised). Fastest. You pick ``n_clusters``. Above ~50k
  points it routes through the built-in native kernel (``fastumap._accel``): ~10 s at 1M x 128.
  Below that, the deterministic numpy path here.
- ``method="spectral"`` — spectral clustering. Build the fuzzy kNN graph. Take the leading
  normalised-adjacency eigenvectors (fastumap's own spectral machinery). k-means those. Finds
  non-convex clusters k-means alone misses. Needs ``fastumap[ann]`` only above ~16k points, where
  the kNN switches to the approximate backend.
- ``method="dbscan"`` — density clustering. Finds clusters of any shape, labels outliers ``-1``,
  and picks the number of clusters itself. You give it ``eps`` (or let it auto-pick).
- ``method="hdbscan"`` — the hierarchical version: no ``eps`` at all, so clusters of different
  densities come out together. You give it ``min_cluster_size``. The usual choice after a UMAP
  reduction when you don't know k.

Memory: the native kernel runs in float32, but the numpy entry points upcast the input to float64
(``np.ascontiguousarray(..., dtype=np.float64)``) for a deterministic-to-the-byte result and for
the inertia. So on any path the float64 work array is the memory driver at large n: a
1,000,000 x 1024 float32 input becomes an ~8 GB float64 array. The assignment step is blocked, so
the n-by-k distance matrix on top of that stays bounded by the block size, not by n. At 1M the
native kernel meets the ~10 s budget; without it (an unbuilt checkout) the numpy fallback is
minutes, since a numpy inner loop is not the threaded float32 GEMM that budget assumes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from ._threads import with_blas_limit
from ._types import FloatArray, IntArray
from ._types import unit_rows as _unit_rows

_BLOCK = 4096  # rows per block in the assignment step; bounds peak memory to block-by-k
_ACCEL_MIN = 50_000  # below this the numpy path is fast and deterministic; above, use the accel
# Above this, full Lloyd's per-iteration O(n*k*d) pass is the wall on a small container, so the
# default switches to mini-batch. Chosen by n ALONE, never by core count: gating on cores would
# make the same input and seed give different labels on different machines, which is exactly the
# guarantee this library sells. batch_size=0 forces full Lloyd back on at any size.
_MINIBATCH_MIN = 200_000
_MINIBATCH_BATCH = 10_000  # rows per iteration; ~2 full passes of total work at the default n_iter
_MINIBATCH_ITER = 100  # mini-batch needs many cheap passes where Lloyd needs few expensive ones

try:
    # Native k-means, built into the package as fastumap._accel: a matrixmultiply-GEMM + rayon
    # Lloyd. Clusters 1M x 128 in ~10s (multi-core) vs the numpy path's minutes. Same API,
    # different (f32) results. Falls back to numpy only if the extension is absent (unbuilt checkout).
    _accel_kmeans = importlib.import_module("fastumap._accel").kmeans
except (ImportError, AttributeError):
    _accel_kmeans = None

try:
    # Mini-batch k-means (Sculley 2010), the same kernel's CPU-constrained sibling. See the
    # batch_size docs on `kmeans` for when it is used.
    _accel_minibatch = importlib.import_module("fastumap._accel").minibatch_kmeans
except (ImportError, AttributeError):
    _accel_minibatch = None


@dataclass(frozen=True)
class ClusterResult:
    """What ``cluster(..., return_centroids=True)`` returns.

    ``centroids``, ``inertia`` and ``dispersion`` are in the space the clustering ran in: the input
    space, or unit-normalised rows for ``metric="cosine"``. ``inertia`` is the k-means objective,
    the sum of squared distances from each point to its assigned centroid. Lower is tighter. It is
    the number to compare across ``n_clusters`` when picking k.

    ``dispersion[c]`` is the mean distance of cluster ``c``'s points to centroid ``c``, and
    ``sizes[c]`` is how many points it has (both length ``n_clusters``, indexed by label). Where
    ``inertia`` is one number for the whole fit, ``dispersion`` describes each cluster on its own:
    a diffuse cluster and a tight one of the same size look identical by size and total inertia, but
    differ in dispersion, which is what tells a caller which clusters hold real structure worth
    splitting. Computed in the fit's own metric, so a cosine fit is not silently measured in
    euclidean.
    """

    labels: IntArray
    centroids: FloatArray
    inertia: float
    dispersion: FloatArray
    sizes: IntArray


@dataclass(frozen=True)
class ChooseKResult:
    """What ``choose_k(..., return_evidence=True)`` returns: the chosen ``k`` and the evidence for
    it, so a caller can publish why a k was picked rather than just the integer.

    - ``candidates`` / ``inertias`` — the k-means objective at each candidate k (parallel lists).
      The chosen k is the elbow of this curve; a sharp knee is a stronger claim than a flat one.
    - ``criterion`` — the selection rule, ``"inertia_elbow"`` (the kneedle knee). This is separate
      from the distance ``metric``.
    - ``n_samples`` — how many points the sweep actually ran on (``choose_k`` subsamples above
      ``max_samples``); with ``n_iter`` and ``random_state`` it makes the choice reproducible from
      the result alone.
    """

    k: int
    candidates: list[int]
    inertias: list[float]
    n_samples: int
    n_iter: int
    random_state: int
    criterion: str = "inertia_elbow"


def _inertia(x: FloatArray, labels: IntArray, centroids: FloatArray) -> float:
    """k-means objective: sum of squared distances to the assigned centroid, one block at a time
    so the (n, d) difference is never materialised all at once."""
    total = 0.0
    for start in range(0, x.shape[0], _BLOCK):
        block = x[start : start + _BLOCK]
        diff = block - centroids[labels[start : start + _BLOCK]]
        total += float(np.einsum("ij,ij->", diff, diff))
    return total


def _dispersion_and_sizes(
    x: FloatArray, labels: IntArray, centroids: FloatArray, k: int
) -> tuple[FloatArray, IntArray]:
    """Per-cluster mean distance to the assigned centroid, and per-cluster point counts. Blocked
    like ``_inertia`` (the (n, d) difference is never materialised at once). An empty cluster gets
    dispersion 0.0."""
    sizes = np.bincount(labels, minlength=k).astype(np.int64)
    dist_sums = np.zeros(k, dtype=np.float64)
    for start in range(0, x.shape[0], _BLOCK):
        lbl = labels[start : start + _BLOCK]
        diff = x[start : start + _BLOCK] - centroids[lbl]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dist_sums += np.bincount(lbl, weights=dist, minlength=k)
    dispersion = np.divide(dist_sums, sizes, out=np.zeros(k, dtype=np.float64), where=sizes > 0)
    return dispersion, sizes


@overload
def cluster(
    embeddings: FloatArray,
    *,
    method: str = ...,
    n_clusters: int | None = ...,
    n_neighbors: int | None = ...,
    metric: str = ...,
    n_iter: int = ...,
    n_init: int | None = ...,
    random_state: int = ...,
    eps: float | None = ...,
    min_samples: int | None = ...,
    min_cluster_size: int = ...,
    return_centroids: Literal[False] = ...,
) -> IntArray: ...
@overload
def cluster(
    embeddings: FloatArray,
    *,
    method: str = ...,
    n_clusters: int | None = ...,
    n_neighbors: int | None = ...,
    metric: str = ...,
    n_iter: int = ...,
    n_init: int | None = ...,
    random_state: int = ...,
    eps: float | None = ...,
    min_samples: int | None = ...,
    min_cluster_size: int = ...,
    return_centroids: Literal[True],
) -> ClusterResult: ...
@with_blas_limit
def cluster(
    embeddings: FloatArray,
    *,
    method: str = "kmeans",
    n_clusters: int | None = None,
    n_neighbors: int | None = None,
    metric: str = "euclidean",
    n_iter: int = 50,
    n_init: int | None = None,
    random_state: int = 42,
    eps: float | None = None,
    min_samples: int | None = None,
    min_cluster_size: int = 5,
    return_centroids: bool = False,
) -> IntArray | ClusterResult:
    """Cluster ``embeddings`` ``(n, d)`` into integer labels ``(n,)``.

    Four methods:

    - ``"kmeans"`` and ``"spectral"`` take ``n_clusters`` (you pick k).
    - ``"dbscan"`` picks the number of clusters itself and labels outliers ``-1``. It takes
      ``eps`` (``None`` = auto) and ``min_samples`` instead of ``n_clusters``. Good when you don't
      know k or want noise handling; see :func:`dbscan`.
    - ``"hdbscan"`` does the same but with no ``eps`` at all, so clusters of different densities
      are found together. It takes ``min_cluster_size`` (and ``min_samples``, which defaults to
      ``min_cluster_size`` here, not to dbscan's 5); see :func:`hdbscan`.

    ``metric`` is ``"euclidean"`` or ``"cosine"`` (cosine unit-normalises the rows). Deterministic
    given ``random_state``. Does NOT compute a layout. For a picture, call ``umap_project``.

    ``n_init`` (``"kmeans"`` only) restarts the fit from that many seeds and keeps the lowest
    inertia. The default single run is fine on the overlapping data real embeddings produce; raise
    it (``n_init=5``) if your data is well-separated, where mini-batch can otherwise settle into a
    worse optimum.

    With ``return_centroids=True`` the return is a :class:`ClusterResult` (labels, centroids,
    inertia) instead of just labels (``"kmeans"`` / ``"spectral"`` only).
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    work = _unit_rows(x) if metric == "cosine" else x
    if method in ("dbscan", "hdbscan"):
        if return_centroids:
            raise ValueError(
                f"return_centroids is not supported for method={method!r} (noise has no centroid)"
            )
        # work is already metric-normalised, so run the density methods in euclidean space on it.
        # n_neighbors=None means "use the method's own default" (30 for the density methods), so
        # an unset call is unchanged; an explicit value is honoured (previously it was dropped).
        density_neighbors = 30 if n_neighbors is None else n_neighbors
        if method == "dbscan":
            # dbscan's own default; hdbscan's is min_cluster_size, so it takes None straight through.
            return dbscan(
                work,
                eps=eps,
                min_samples=5 if min_samples is None else min_samples,
                n_neighbors=density_neighbors,
                metric="euclidean",
            )
        return hdbscan(
            work,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_neighbors=density_neighbors,
            metric="euclidean",
        )
    if n_clusters is None:
        # kmeans/spectral need a k. The documented default (None) means "pick one" via choose_k,
        # not "refuse the call". work is already metric-normalised, so choose_k runs in euclidean
        # space on it; it passes an explicit k back into cluster, so this does not recurse forever.
        n_clusters = choose_k(work, metric="euclidean", n_iter=n_iter, random_state=random_state)
    if method == "kmeans":
        labels, centroids = kmeans(
            work, n_clusters, n_iter=n_iter, n_init=n_init, random_state=random_state
        )
    elif method == "spectral":
        # spectral's own default kNN size is 15; None means "use it".
        labels = _spectral(
            work,
            n_clusters,
            15 if n_neighbors is None else n_neighbors,
            metric,
            n_iter,
            random_state,
        )
        # spectral returns labels only; recover input-space centroids from them when asked.
        centroids = _centroids_from_labels(work, labels, n_clusters) if return_centroids else None
    else:
        raise ValueError(
            f"unknown method {method!r}; use 'kmeans', 'spectral', 'dbscan', or 'hdbscan'"
        )
    if return_centroids:
        assert centroids is not None
        dispersion, sizes = _dispersion_and_sizes(work, labels, centroids, n_clusters)
        return ClusterResult(
            labels, centroids, _inertia(work, labels, centroids), dispersion, sizes
        )
    return labels


@overload
def assign_clusters(
    embeddings: FloatArray,
    centroids: FloatArray,
    *,
    metric: str = ...,
    max_distance: float | None = ...,
    return_distances: Literal[False] = ...,
) -> IntArray: ...
@overload
def assign_clusters(
    embeddings: FloatArray,
    centroids: FloatArray,
    *,
    metric: str = ...,
    max_distance: float | None = ...,
    return_distances: Literal[True],
) -> tuple[IntArray, FloatArray]: ...
@with_blas_limit
def assign_clusters(
    embeddings: FloatArray,
    centroids: FloatArray,
    *,
    metric: str = "euclidean",
    max_distance: float | None = None,
    return_distances: bool = False,
) -> IntArray | tuple[IntArray, FloatArray]:
    """Assign each row of ``embeddings`` to its nearest of ``centroids``. This is the *predict*
    half of a fit/predict split. New points get labels without re-clustering.

    ``centroids`` come from a prior ``cluster(..., return_centroids=True)`` or ``kmeans`` run.
    ``metric`` must match the one they were fitted with (``"cosine"`` unit-normalises the rows, as
    fitting did). Uses the same blocked nearest-centroid step as ``cluster``, so peak memory is
    bounded by the block size, no matter how many rows you pass.

    By default a point is always given its nearest centroid, however far that is. To stop far
    points being absorbed into whatever cluster is least far:

    - ``max_distance`` labels any point whose nearest centroid is beyond it ``-1`` (noise), the
      same outlier label the density methods use. The distance is in the space the fit ran in
      (unit-normalised rows for ``"cosine"``), so the threshold is comparable to the ``inertia``
      per point.
    - ``return_distances=True`` also returns the distance to the assigned centroid for every row
      (the raw distance, even for ``-1`` rows), so a caller can pick a threshold from the data.
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    c = np.ascontiguousarray(centroids, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("centroids must be 2-D (k, d)")
    if x.shape[1] != c.shape[1]:
        raise ValueError(f"embeddings dim {x.shape[1]} != centroids dim {c.shape[1]}")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    if max_distance is not None and max_distance < 0.0:
        raise ValueError("max_distance must be >= 0")
    work = _unit_rows(x) if metric == "cosine" else x
    if not return_distances and max_distance is None:
        return _assign(work, c)  # fast path: argmin only, no per-row distance needed
    labels, distances = _assign_with_distances(work, c)
    if max_distance is not None:
        labels = labels.copy()
        labels[distances > max_distance] = -1
    return (labels, distances) if return_distances else labels


@with_blas_limit
def dbscan(
    embeddings: FloatArray,
    *,
    eps: float | None = None,
    min_samples: int = 5,
    metric: str = "euclidean",
    n_neighbors: int = 30,
) -> IntArray:
    """Density clustering (DBSCAN). Finds clusters of any shape and labels outliers ``-1``, with
    no ``n_clusters`` to choose.

    A point is a *core* point if at least ``min_samples`` points (itself included) lie within
    ``eps`` of it. Core points within ``eps`` of each other join one cluster. A non-core point
    within ``eps`` of a core joins as a *border*. Everything else is noise (``-1``).

    ``eps=None`` picks ``eps`` as the median distance to each point's ``min_samples``-th neighbour
    (the usual k-distance heuristic). Neighbours come from fastumap's kNN (blocked exact, or the
    faiss approx backend above ~16k points), so this scales past a naive O(n^2) DBSCAN. Only the
    ``n_neighbors`` nearest are considered, so a very dense cluster may need a larger
    ``n_neighbors`` (which must exceed ``min_samples``). Deterministic given the kNN.
    """
    from .neighbors import knn_graph  # lazy: keeps import light

    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    work = _unit_rows(x) if metric == "cosine" else x
    n = work.shape[0]
    k = min(max(n_neighbors, min_samples + 1), n)
    indices, dists = knn_graph(work, k, metric=metric, mode="auto")
    if eps is None:
        col = min(min_samples, k - 1)  # distance to the min_samples-th neighbour (column 0 = self)
        eps = float(np.median(dists[:, col]))
    return _dbscan(indices, dists, float(eps), min_samples)


def _uf_find(parent: list[int], a: int) -> int:
    """Union-find root of ``a`` with path halving; ``parent`` is mutated in place. Shared by dbscan,
    the mutual-reachability MST, and single linkage, which otherwise each reimplemented it."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _dbscan(indices: IntArray, dists: FloatArray, eps: float, min_samples: int) -> IntArray:  # noqa: C901
    """DBSCAN over a precomputed kNN. Core points are unioned to their eps-neighbours; borders take
    a neighbouring core's label; the rest are noise (-1). Complexity 13: the union pass, the
    core-relabel pass and the border pass are one algorithm; splitting them would not read better."""
    n = indices.shape[0]
    within = dists <= eps  # (n, k); column 0 is self at distance 0, so always within
    core = within.sum(axis=1) >= min_samples  # (n,) bool

    parent = list(range(n))

    def union(a: int, b: int) -> None:
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    core_points = np.where(core)[0]
    for i in core_points:
        ii = int(i)
        row_idx = indices[ii]
        row_within = within[ii]
        for pos in range(row_idx.shape[0]):
            if row_within[pos]:
                j = int(row_idx[pos])
                if j != ii and core[j]:
                    union(ii, j)

    labels = np.full(n, -1, dtype=np.int64)
    remap: dict[int, int] = {}
    for i in core_points:
        root = _uf_find(parent, int(i))
        if root not in remap:
            remap[root] = len(remap)
        labels[int(i)] = remap[root]
    # border points: a non-core point within eps of a core takes that core's cluster label.
    for i in range(n):
        if core[i] or labels[i] != -1:
            continue
        row_idx = indices[i]
        row_within = within[i]
        for pos in range(row_idx.shape[0]):
            if row_within[pos] and core[int(row_idx[pos])]:
                labels[i] = labels[int(row_idx[pos])]
                break
    return labels


@with_blas_limit
def hdbscan(
    embeddings: FloatArray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    n_neighbors: int = 30,
) -> IntArray:
    """Hierarchical density clustering (HDBSCAN). Like :func:`dbscan` but with no ``eps`` to pick,
    so clusters of *different* densities are all found at once. Outliers are ``-1``.

    DBSCAN cuts the density hierarchy at one global ``eps``: pick it loose and two nearby clusters
    merge, pick it tight and the sparse cluster becomes noise. HDBSCAN builds the whole hierarchy
    and then keeps the branches that persist longest over it, so each cluster is effectively cut at
    its own density.

    Four steps, all over fastumap's kNN: the *core distance* of a point (its ``min_samples``-th
    neighbour distance) inflates every edge into a *mutual reachability* distance
    ``max(core_i, core_j, d_ij)``; an MST over those edges gives the single-linkage hierarchy;
    branches smaller than ``min_cluster_size`` are condensed away as noise; what is left is scored
    by stability (excess of mass) and the best non-overlapping set wins.

    ``min_cluster_size`` is the real knob: the smallest group you are willing to call a cluster.
    ``min_samples`` (default: ``min_cluster_size``) sets how conservative the density estimate is —
    raise it to push more points into noise.

    One consequence to know before you call it: the root of the hierarchy is never a cluster, so
    data with only *one* dense group in it comes back all ``-1``. HDBSCAN answers "which groups
    stand out from each other", not "is there a group here". This matches scikit-learn's
    ``allow_single_cluster=False`` default.

    This is *approximate* HDBSCAN: the MST is built over the ``n_neighbors`` kNN edges, not the
    full mutual-reachability graph (the same tradeoff RAPIDS makes), so a cluster whose points are
    all further apart than their ``n_neighbors``-th neighbour can split. The kNN graph may also be
    disconnected; the pieces are joined at infinite distance, which is the honest answer for
    points that share no neighbourhood. Deterministic given the kNN.
    """
    from .neighbors import knn_graph  # lazy: keeps import light

    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")
    samples = min_cluster_size if min_samples is None else min_samples
    if samples < 1:
        raise ValueError("min_samples must be >= 1")
    work = _unit_rows(x) if metric == "cosine" else x
    n = work.shape[0]
    if n < 2:
        return np.full(n, -1, dtype=np.int64)
    k = min(max(n_neighbors, samples + 1), n)
    indices, dists = knn_graph(work, k, metric=metric, mode="auto")
    core = dists[:, min(samples, k - 1)]  # min_samples-th neighbour; column 0 is self
    return _hdbscan(indices, dists, core, min_cluster_size)


def _hdbscan(
    indices: IntArray, dists: FloatArray, core: FloatArray, min_cluster_size: int
) -> IntArray:
    """HDBSCAN over a precomputed kNN and its core distances: MST -> single-linkage tree ->
    condensed tree -> stability selection -> labels."""
    n = indices.shape[0]
    edges = _mutual_reachability_mst(indices, dists, core)
    left, right, height, size = _single_linkage(edges, n)
    rows = _condense(left, right, height, size, n, min_cluster_size)
    return _labels_from_condensed(rows, n)


def _mutual_reachability_mst(
    indices: IntArray, dists: FloatArray, core: FloatArray
) -> list[tuple[float, int, int]]:
    """Kruskal over the kNN edges, weighted by mutual reachability. Returns exactly ``n - 1``
    edges ``(weight, a, b)`` ascending. The kNN graph can be disconnected, so whatever components
    survive are joined at ``inf`` — a merge that costs infinite density, i.e. lambda 0."""
    n, k = indices.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k - 1)
    dst = np.ascontiguousarray(indices[:, 1:]).ravel()
    d = np.ascontiguousarray(dists[:, 1:]).ravel()
    weight = np.maximum(np.maximum(core[src], core[dst]), d)
    order = np.argsort(weight, kind="stable")
    # .tolist() once: iterating numpy scalars in the Kruskal loop below is several times slower.
    ws = weight[order].tolist()
    as_ = src[order].tolist()
    bs = dst[order].tolist()

    parent = list(range(n))

    mst: list[tuple[float, int, int]] = []
    for pos in range(len(ws)):
        ra, rb = _uf_find(parent, as_[pos]), _uf_find(parent, bs[pos])
        if ra == rb:
            continue
        parent[rb] = ra
        mst.append((ws[pos], as_[pos], bs[pos]))
        if len(mst) == n - 1:
            break
    if len(mst) < n - 1:  # disconnected kNN graph: bridge the components at infinite distance
        roots = [i for i in range(n) if _uf_find(parent, i) == i]
        for other in roots[1:]:
            parent[_uf_find(parent, other)] = _uf_find(parent, roots[0])
            mst.append((float("inf"), roots[0], other))
    return mst


def _single_linkage(
    mst: list[tuple[float, int, int]], n: int
) -> tuple[list[int], list[int], list[float], list[int]]:
    """MST edges (ascending) -> single-linkage tree. Node ``i < n`` is a point; node ``n + j`` is
    the merge made by edge ``j``, joining ``left[j]`` and ``right[j]`` at ``height[j]``."""
    parent = list(range(n))
    node_of = list(range(n))  # union-find root -> the tree node that root currently stands for
    size = [1] * (2 * n - 1)
    left = [0] * (n - 1)
    right = [0] * (n - 1)
    height = [0.0] * (n - 1)

    for j, (w, a, b) in enumerate(mst):
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        na, nb = node_of[ra], node_of[rb]
        node = n + j
        left[j], right[j], height[j] = na, nb, w
        size[node] = size[na] + size[nb]
        parent[rb] = ra
        node_of[ra] = node
    return left, right, height, size


def _condense(
    left: list[int],
    right: list[int],
    height: list[float],
    size: list[int],
    n: int,
    min_cluster_size: int,
) -> list[tuple[int, int, float, int]]:
    """Walk the single-linkage tree down from the root, dropping every branch smaller than
    ``min_cluster_size``. Returns rows ``(cluster, child, lambda, child_size)``: ``child < n`` is a
    point leaving ``cluster`` at that lambda, ``child >= n`` is a sub-cluster born there. Cluster
    ids start at ``n`` (the root) and always exceed their parent's."""

    def subtree_points(node: int) -> list[int]:
        points: list[int] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur < n:
                points.append(cur)
            else:
                stack.append(left[cur - n])
                stack.append(right[cur - n])
        return points

    rows: list[tuple[int, int, float, int]] = []
    next_label = n + 1
    stack = [(2 * n - 2, n)]  # (tree node, the cluster id it belongs to)
    while stack:
        node, label = stack.pop()
        if node < n:
            continue
        j = node - n
        lam = 1.0 / height[j] if height[j] > 0.0 else float("inf")
        lc, rc = left[j], right[j]
        big_l, big_r = size[lc] >= min_cluster_size, size[rc] >= min_cluster_size
        if big_l and big_r:  # a real split: two new clusters born at lam
            for child in (lc, rc):
                rows.append((label, next_label, lam, size[child]))
                stack.append((child, next_label))
                next_label += 1
        elif not big_l and not big_r:  # the whole branch falls out; nothing below is a cluster
            for point in subtree_points(node):
                rows.append((label, point, lam, 1))
        else:  # one side is too small: it falls out, the other stays this same cluster
            small, big = (lc, rc) if not big_l else (rc, lc)
            for point in subtree_points(small):
                rows.append((label, point, lam, 1))
            stack.append((big, label))
    return rows


def _labels_from_condensed(rows: list[tuple[int, int, float, int]], n: int) -> IntArray:
    """Score every cluster in the condensed tree by stability and keep the best non-overlapping
    set (excess of mass), then label each point by the deepest kept cluster above it.

    Stability of a cluster is ``sum over its children of (lambda_leave - lambda_birth) * size`` —
    how much density range it survives, weighted by how many points survive it. A cluster is kept
    only if it beats the total of its descendants; otherwise the descendants win. The root is
    never a cluster (it holds everything, so "one big cluster" is never an answer here)."""
    labels = np.full(n, -1, dtype=np.int64)
    if not rows:
        return labels
    root = n
    birth: dict[int, float] = {root: 0.0}
    children: dict[int, list[int]] = {}
    for cluster, child, lam, _ in rows:
        if child >= n:
            birth[child] = lam
            children.setdefault(cluster, []).append(child)
    stability: dict[int, float] = dict.fromkeys(birth, 0.0)
    for cluster, _, lam, count in rows:
        stability[cluster] += (lam - birth[cluster]) * count

    # Bottom-up: a child always has a larger id than its parent, so descending order is leaves-up.
    selected = {c for c in stability if c != root}
    for cluster in sorted(selected, reverse=True):
        below = sum(stability[c] for c in children.get(cluster, ()))
        if below > stability[cluster]:
            selected.discard(cluster)
            stability[cluster] = below  # the descendants' score propagates up in its place
        else:
            stack = list(children.get(cluster, ()))
            while stack:  # this cluster wins, so nothing under it can also be a cluster
                sub = stack.pop()
                selected.discard(sub)
                stack.extend(children.get(sub, ()))
    if not selected:
        return labels

    label_of = {c: i for i, c in enumerate(sorted(selected))}
    up = {child: cluster for cluster, child, _, _ in rows if child not in selected}
    for point in range(n):
        node = point
        while node in up:
            node = up[node]
        if node in label_of:
            labels[point] = label_of[node]
    return labels


@overload
def choose_k(
    embeddings: FloatArray,
    *,
    k_min: int = ...,
    k_max: int = ...,
    metric: str = ...,
    n_iter: int = ...,
    random_state: int = ...,
    max_samples: int = ...,
    n_jobs: int = ...,
    return_evidence: Literal[False] = ...,
) -> int: ...
@overload
def choose_k(
    embeddings: FloatArray,
    *,
    k_min: int = ...,
    k_max: int = ...,
    metric: str = ...,
    n_iter: int = ...,
    random_state: int = ...,
    max_samples: int = ...,
    n_jobs: int = ...,
    return_evidence: Literal[True],
) -> ChooseKResult: ...
@with_blas_limit
def choose_k(
    embeddings: FloatArray,
    *,
    k_min: int = 2,
    k_max: int = 10,
    metric: str = "euclidean",
    n_iter: int = 50,
    random_state: int = 42,
    max_samples: int = 20_000,
    n_jobs: int = 1,
    return_evidence: bool = False,
) -> int | ChooseKResult:
    """Pick ``n_clusters`` automatically by the inertia elbow, the selection criterion here. It is
    the "kneedle" knee of the inertia-vs-k curve: the k past which adding clusters stops buying
    much tightness. This is separate from ``metric``, which is the distance (euclidean or cosine).

    Runs k-means for each ``k`` in ``[k_min, k_max]`` and returns the elbow. On more than
    ``max_samples`` rows it searches on a deterministic random subsample to bound the cost (the
    returned k still applies to the full data).

    ``n_jobs`` runs the independent per-k fits in parallel. The fits are pinned to one BLAS thread
    each for the duration, so ``n_jobs`` of them run concurrently without oversubscribing; the
    result is identical to ``n_jobs=1``. ``return_evidence=True`` returns a :class:`ChooseKResult`
    (the chosen k plus the per-candidate inertias and the sample size) instead of the bare int, so
    a caller can publish the evidence behind the choice.

    It is a heuristic. For a clear cluster structure it lands on the true k give or take one. For
    gradual structure, treat it as a starting point.
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    n = x.shape[0]
    hi = min(k_max, n - 1)
    if k_min < 2 or hi < k_min:
        raise ValueError(f"need 2 <= k_min <= k_max < n; got k_min={k_min}, k_max={k_max}, n={n}")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        x = x[rng.choice(n, max_samples, replace=False)]
    # Normalise once for cosine (squared-L2 on unit rows), then every fit runs in euclidean space
    # on the same array — matching cluster(metric=...) but without re-normalising per candidate.
    work = _unit_rows(x) if metric == "cosine" else x
    ks = list(range(k_min, hi + 1))

    def _fit_inertia(k: int) -> float:
        labels, centroids = _kmeans_core(work, k, n_iter=n_iter, random_state=random_state)
        return _inertia(work, labels, centroids)

    if n_jobs == 1 or len(ks) == 1:
        inertias = [_fit_inertia(k) for k in ks]
    else:
        # The fits are embarrassingly parallel. Pin BLAS to one thread for the sweep so each fit is
        # single-threaded and n_jobs of them run concurrently (the GEMM in _assign releases the
        # GIL); this avoids the oversubscription that per-fit multi-threaded BLAS would cause.
        # _kmeans_core carries no BLAS-cap decorator, so the threads never race on threadpool_limits.
        from concurrent.futures import ThreadPoolExecutor

        import threadpoolctl

        with (
            threadpoolctl.threadpool_limits(1, "blas"),
            ThreadPoolExecutor(max_workers=min(n_jobs, len(ks))) as pool,
        ):
            inertias = list(pool.map(_fit_inertia, ks))

    best = ks[0] if len(ks) == 1 else _elbow(ks, inertias)
    if return_evidence:
        return ChooseKResult(
            k=best,
            candidates=ks,
            inertias=inertias,
            n_samples=work.shape[0],
            n_iter=n_iter,
            random_state=random_state,
        )
    return best


def _elbow(ks: list[int], inertias: list[float]) -> int:
    """Kneedle: normalise the (k, inertia) curve to the unit square and return the k whose point
    sits farthest above the straight chord from first to last — the sharpest bend."""
    xs = np.asarray(ks, dtype=np.float64)
    ys = np.asarray(inertias, dtype=np.float64)
    xs = (xs - xs[0]) / (xs[-1] - xs[0])
    span = float(ys[0] - ys[-1])
    ys = (ys[0] - ys) / (span if span > 0.0 else 1.0)  # decreasing inertia -> increasing 0..1
    return ks[int(np.argmax(ys - xs))]  # farthest above the y=x chord


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


def _assign_with_distances(x: FloatArray, centroids: FloatArray) -> tuple[IntArray, FloatArray]:
    """Nearest-centroid label AND the distance to it, per row, one block at a time. Unlike
    ``_assign`` this keeps the ``||b||^2`` term (it does not affect the argmin, but it is needed for
    the actual distance a caller can threshold on)."""
    n = x.shape[0]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    labels = np.empty(n, dtype=np.int64)
    nearest_d2 = np.empty(n, dtype=np.float64)
    for start in range(0, n, _BLOCK):
        block = x[start : start + _BLOCK]
        b_sq = np.einsum("ij,ij->i", block, block)
        d2 = b_sq[:, None] + c_sq[None, :] - 2.0 * (block @ centroids.T)
        np.maximum(d2, 0.0, out=d2)  # clip negative round-off before the sqrt
        idx = np.argmin(d2, axis=1)
        labels[start : start + _BLOCK] = idx
        nearest_d2[start : start + _BLOCK] = d2[np.arange(idx.shape[0]), idx]
    return labels, np.sqrt(nearest_d2)


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


@with_blas_limit
def kmeans(
    x: FloatArray,
    n_clusters: int,
    *,
    n_iter: int = 50,
    random_state: int = 42,
    batch_size: int | None = None,
    n_init: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """Lloyd's k-means with k-means++ init. Returns ``(labels, centroids)``.

    Uses the built-in native kernel (fastumap._accel) above ``_ACCEL_MIN`` points when present
    (~10s at 1M x 128 vs minutes for numpy). Otherwise the pure-numpy path here, which is
    deterministic to the byte. Both converge to the same objective within the usual tolerance.

    ``batch_size`` picks between full Lloyd and **mini-batch** k-means (Sculley 2010). Full Lloyd
    touches all ``n`` rows every iteration, which is fine with cores to spare and slow on a
    0.5-2 vCPU container. Mini-batch samples ``batch_size`` rows per iteration and moves only the
    centroids they land on, so per-iteration work drops from ``O(n*k*d)`` to ``O(batch*k*d)`` at a
    small cost in inertia.

    - ``None`` (default) — mini-batch above ``_MINIBATCH_MIN`` rows, full Lloyd below. The switch
      is on ``n`` only, so the same input and seed give the same labels on any machine.
    - ``0`` — force full Lloyd at any size.
    - a positive int — force mini-batch with that batch size.

    Mini-batch needs the native kernel; without it the numpy path runs full Lloyd regardless.

    ``n_init`` restarts the fit from that many seeds and keeps the lowest-inertia result. ``None``
    (default) is a single run. Mini-batch can settle into a bad optimum on well-separated clusters
    (4.8x worse inertia than full Lloyd on adversarial blobs); it is fine on the overlapping data
    real embeddings produce (~1.01x), so the default stays a single fast run. If your data is
    well-separated, pass ``n_init=5`` to recover full-Lloyd parity. Runtime scales with ``n_init``.
    """
    return _kmeans_core(
        x,
        n_clusters,
        n_iter=n_iter,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )


def _kmeans_core(
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
        return _kmeans_single(
            x, n_clusters, n_iter=n_iter, random_state=random_state, batch_size=batch_size
        )
    best: tuple[IntArray, FloatArray] | None = None
    best_inertia = float("inf")
    for i in range(inits):
        labels, centroids = _kmeans_single(
            x, n_clusters, n_iter=n_iter, random_state=random_state + i, batch_size=batch_size
        )
        val = _inertia(x, labels, centroids)
        if val < best_inertia:
            best_inertia, best = val, (labels, centroids)
    assert best is not None
    return best


def _kmeans_single(
    x: FloatArray,
    n_clusters: int,
    *,
    n_iter: int = 50,
    random_state: int = 42,
    batch_size: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """One k-means fit (mini-batch, native full Lloyd, or numpy full Lloyd by size and build)."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > x.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} exceeds n={x.shape[0]}")
    if batch_size is not None and batch_size < 0:
        raise ValueError("batch_size must be >= 0 (0 forces full Lloyd, None picks by n)")
    n = x.shape[0]
    auto_minibatch = batch_size is None and n >= _MINIBATCH_MIN
    if _accel_minibatch is not None and (auto_minibatch or (batch_size or 0) > 0):
        batch = _MINIBATCH_BATCH if batch_size in (None, 0) else batch_size
        # Mini-batch trades an expensive pass for many cheap ones, so it wants more iterations
        # than Lloyd. Only override a caller who left n_iter at its Lloyd default.
        iters = _MINIBATCH_ITER if n_iter == 50 else n_iter
        data = np.ascontiguousarray(x, dtype=np.float64)
        labels = np.asarray(
            _accel_minibatch(data, n_clusters, iters, int(batch), int(random_state)),
            dtype=np.int64,
        )
        return labels, _centroids_from_labels(data, labels, n_clusters)
    if _accel_kmeans is not None and n >= _ACCEL_MIN:
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
