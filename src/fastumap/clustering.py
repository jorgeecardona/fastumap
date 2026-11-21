"""Clustering, implemented ourselves. numpy + scipy plus a built-in native k-means kernel. No
third-party clustering library.

fastumap reimplements UMAP from scratch. Clustering is the same. ``cluster()`` does NOT compute
a 2-D/3-D layout first. That layout is an iterative SGD and takes minutes at a million points.
Clustering does not need it.

Four methods, all ours:

- ``method="kmeans"``: k-means (k-means++ seeding, Lloyd iterations, blocked assignment so the
  n-by-k distance matrix is never materialised). Fastest. You pick ``n_clusters``. Runs on the
  built-in native kernel (``fastumap._accel``) at every size: ~10 s at 1M x 128.
- ``method="spectral"``: spectral clustering. Build the fuzzy kNN graph. Take the leading
  normalised-adjacency eigenvectors (fastumap's own spectral machinery). k-means those. Finds
  non-convex clusters k-means alone misses. Needs ``fastumap[ann]`` only above ~16k points, where
  the kNN switches to the approximate backend.
- ``method="dbscan"``: density clustering. Finds clusters of any shape, labels outliers ``-1``,
  and picks the number of clusters itself. You give it ``eps`` (or let it auto-pick).
- ``method="hdbscan"``: the hierarchical version: no ``eps`` at all, so clusters of different
  densities come out together. You give it ``min_cluster_size``. The usual choice after a UMAP
  reduction when you don't know k.

Memory: the native kernel runs in float32, but the entry points upcast the input to float64
for the inertia. So the float64 work array is the memory driver at large n: a 1,000,000 x 1024
float32 input becomes an ~8 GB float64 array. The assignment step is blocked, so
the n-by-k distance matrix on top of that stays bounded by the block size, not by n.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, overload

import numpy as np

from . import _density, _kmeans
from ._threads import with_blas_limit
from ._types import FloatArray, IntArray
from ._types import unit_rows as _unit_rows


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

    - ``candidates`` / ``inertias``: the k-means objective at each candidate k (parallel lists).
      The chosen k is the elbow of this curve; a sharp knee is a stronger claim than a flat one.
    - ``criterion``: the selection rule, ``"log_inertia_elbow"`` (the kneedle knee on log inertia). This is separate
      from the distance ``metric``.
    - ``n_samples``: how many points the sweep actually ran on (``choose_k`` subsamples above
      ``max_samples``); with ``n_iter`` and ``random_state`` it makes the choice reproducible from
      the result alone.
    - ``n_init``: k-means starts per candidate; the lowest inertia is kept.
    - ``inertia_rises``: how many candidates have a higher inertia than the candidate before. At
      the optimum inertia never rises with k, so a value above 0 means a fit failed and the elbow
      is less reliable. Raise ``n_init`` (#30).
    """

    k: int
    candidates: list[int]
    inertias: list[float]
    n_samples: int
    n_iter: int
    random_state: int
    criterion: str = "log_inertia_elbow"
    n_init: int = 1
    inertia_rises: int = 0


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
    pca_dim: int | None = ...,
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
    pca_dim: int | None = ...,
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
    pca_dim: int | None = None,
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

    ``pca_dim`` (default ``None`` = off) reduces the input to that many principal components
    to find the clusters. The kNN and k-means cost grow with the dimension, so 768-d embeddings
    reduced to 16-32 cluster about 3x faster at the same inertia (MNIST 24k). For ``"kmeans"``
    and ``"spectral"``, centroids are means in the input space and the returned labels are the
    nearest of those centroids, so ``assign_clusters`` on the same points gives the same labels. A ``pca_dim`` at or above the input
    dimension does nothing.
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D (n, d)")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    work = _unit_rows(x) if metric == "cosine" else x
    if pca_dim is not None and pca_dim < 1:
        raise ValueError("pca_dim must be >= 1")
    if pca_dim is not None and pca_dim < work.shape[1]:
        return _cluster_reduced(
            work,
            pca_dim,
            method=method,
            n_clusters=n_clusters,
            n_neighbors=n_neighbors,
            n_iter=n_iter,
            n_init=n_init,
            random_state=random_state,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            return_centroids=return_centroids,
        )
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
        centroids = (
            _kmeans.centroids_from_labels(work, labels, n_clusters) if return_centroids else None
        )
    else:
        raise ValueError(
            f"unknown method {method!r}; use 'kmeans', 'spectral', 'dbscan', or 'hdbscan'"
        )
    if return_centroids:
        assert centroids is not None
        dispersion, sizes = _kmeans.dispersion_and_sizes(work, labels, centroids, n_clusters)
        return ClusterResult(
            labels, centroids, _kmeans.inertia(work, labels, centroids), dispersion, sizes
        )
    return labels


def _cluster_reduced(
    work: FloatArray,
    pca_dim: int,
    *,
    method: str,
    n_clusters: int | None,
    n_neighbors: int | None,
    n_iter: int,
    n_init: int | None,
    random_state: int,
    eps: float | None,
    min_samples: int | None,
    min_cluster_size: int,
    return_centroids: bool,
) -> IntArray | ClusterResult:
    """``cluster`` on the top ``pca_dim`` principal components of ``work``.

    Clusters are found in the reduced space. For ``"kmeans"`` and ``"spectral"`` each centroid is
    then the mean of its cluster in the space of ``work``, and every point is re-assigned to its
    nearest centroid there. So the returned labels equal ``assign_clusters`` on the same points.
    Density methods return the reduced-space labels.
    """
    if return_centroids and method in ("dbscan", "hdbscan"):
        raise ValueError(
            f"return_centroids is not supported for method={method!r} (noise has no centroid)"
        )
    from .projection import pca_reduce  # lazy: keeps import light

    reduced = pca_reduce(work, pca_dim)[0]
    # Label in the reduced space, then report centroids in the input space.
    labels = cluster(
        reduced,
        method=method,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        n_iter=n_iter,
        n_init=n_init,
        random_state=random_state,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )
    if method in ("dbscan", "hdbscan"):
        return labels  # density labels have noise (-1) and no centroids to re-assign against
    k = n_clusters if n_clusters is not None else int(labels.max()) + 1
    centroids = _kmeans.centroids_from_labels(work, labels, k)
    # Re-assign every point to its nearest full-space centroid, so the returned labels follow the
    # same rule assign_clusters applies to future points. Below ~32 dims up to 9-60% of points
    # would otherwise disagree (MNIST 24k, k=50). The centroids are not recomputed after this.
    labels = _kmeans.assign(work, centroids)
    if not return_centroids:
        return labels
    dispersion, sizes = _kmeans.dispersion_and_sizes(work, labels, centroids, k)
    return ClusterResult(
        labels, centroids, _kmeans.inertia(work, labels, centroids), dispersion, sizes
    )


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
        return _kmeans.assign(work, c)  # fast path: argmin only, no per-row distance needed
    labels, distances = _kmeans.assign_with_distances(work, c)
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
    return _density.dbscan_from_knn(indices, dists, float(eps), min_samples)


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
    ``min_samples`` (default: ``min_cluster_size``) sets how conservative the density estimate is:
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
    return _density.hdbscan_from_knn(indices, dists, core, min_cluster_size)


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
    n_init: int | None = ...,
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
    n_init: int | None = ...,
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
    n_init: int | None = None,
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

    ``n_init`` runs that many k-means starts per candidate and keeps the lowest inertia, as in
    ``kmeans``. ``None`` (default) is one start. The cost grows linearly with it.

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
    # on the same array, matching cluster(metric=...) but without re-normalising per candidate.
    work = _unit_rows(x) if metric == "cosine" else x
    ks = list(range(k_min, hi + 1))

    def _fit_inertia(k: int) -> float:
        labels, centroids = _kmeans.kmeans_core(
            work, k, n_iter=n_iter, random_state=random_state, n_init=n_init
        )
        return _kmeans.inertia(work, labels, centroids)

    if n_jobs == 1 or len(ks) == 1:
        inertias = [_fit_inertia(k) for k in ks]
    else:
        # The fits are embarrassingly parallel. Pin BLAS to one thread for the sweep so each fit is
        # single-threaded and n_jobs of them run concurrently (the GEMM in _kmeans.assign releases
        # the GIL); this avoids the oversubscription that per-fit multi-threaded BLAS would cause.
        # _kmeans.kmeans_core carries no BLAS-cap decorator, so the threads never race on
        # threadpool_limits.
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
            n_init=1 if n_init is None else n_init,
            inertia_rises=sum(b > a for a, b in pairwise(inertias)),
        )
    return best


def _elbow(ks: list[int], inertias: list[float]) -> int:
    """Kneedle on log inertia: return the k farthest above the chord of the normalised curve.

    The log scale measures each drop relative to the current inertia. On raw inertia a run of
    large early drops looks linear and hides the real knee (7 planted blobs gave k=5) (#30).
    """
    xs = np.asarray(ks, dtype=np.float64)
    ys = np.log(np.maximum(np.asarray(inertias, dtype=np.float64), np.finfo(np.float64).tiny))
    xs = (xs - xs[0]) / (xs[-1] - xs[0])
    span = float(ys[0] - ys[-1])
    ys = (ys[0] - ys) / (span if span > 0.0 else 1.0)  # decreasing inertia -> increasing 0..1
    return ks[int(np.argmax(ys - xs))]  # farthest above the y=x chord


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

    Runs on the built-in native kernel (fastumap._accel) at every size: ~10 s at 1M x 128.
    Labels are identical for the same input and seed at any thread count.

    ``batch_size`` picks between full Lloyd and **mini-batch** k-means (Sculley 2010). Full Lloyd
    touches all ``n`` rows every iteration, which is fine with cores to spare and slow on a
    0.5-2 vCPU container. Mini-batch samples ``batch_size`` rows per iteration and moves only the
    centroids they land on, so per-iteration work drops from ``O(n*k*d)`` to ``O(batch*k*d)`` at a
    small cost in inertia.

    - ``None`` (default): mini-batch above 200,000 rows, full Lloyd below. The switch
      is on ``n`` only, so the same input and seed give the same labels on any machine.
    - ``0``: force full Lloyd at any size.
    - a positive int: force mini-batch with that batch size.

    ``n_init`` restarts the fit from that many seeds and keeps the lowest-inertia result. ``None``
    (default) is a single run. Pass ``n_init=5`` on well-separated clusters for extra assurance.
    Runtime scales with ``n_init``.
    """
    return _kmeans.kmeans_core(
        x,
        n_clusters,
        n_iter=n_iter,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )


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
    # Leading (k) eigenvectors of the normalised adjacency, the same routine spectral_layout
    # uses, asking for k components instead of the embedding dimension.
    _, vecs = _scipy.normalized_adjacency_eigenvectors(
        n, graph.head, graph.tail, graph.weight, k, v0
    )
    embedding = _unit_rows(
        np.ascontiguousarray(vecs, dtype=np.float64)
    )  # row-normalise (Ng-Jordan-Weiss)
    return kmeans(embedding, k, n_iter=n_iter, random_state=random_state)[0]
