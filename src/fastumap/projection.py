"""Public projection entry points: ``umap_project``, ``spectral_project``, and the
``fit`` / ``transform`` pair for placing new points into an existing layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO, Literal, overload

import numpy as np

from ._threads import with_blas_limit
from ._types import Float32Array, FloatArray, IntArray
from .graph import (
    FuzzyGraph,
    categorical_intersection,
    compute_membership,
    find_ab_params,
    fuzzy_simplicial_set,
    smooth_knn_dist,
)
from .layout import optimize_layout, optimize_transform, spectral_layout
from .neighbors import knn_between, knn_graph


def _default_epochs(n: int) -> int:
    # umap-learn's rule: more epochs when the graph is small enough to afford them.
    return 500 if n <= 10_000 else 200


def _validate(matrix: FloatArray, dimensions: int, *, min_points: int = 2) -> FloatArray:
    # Sparse input is out of scope (see README "Sparse input"). The blocked kNN relies on
    # dense BLAS matmul, and the target workload is dense encoder/CLS embeddings. Detect a
    # sparse matrix by its `toarray` method. Fail with a clear pointer, rather than the
    # cryptic error np.ascontiguousarray would raise on it.
    if hasattr(matrix, "toarray"):  # scipy.sparse matrices have it; numpy arrays do not
        raise TypeError(
            "fastumap takes a dense numpy array, not a sparse matrix; "
            "densify first with matrix.toarray() (sparse input is not supported — see README)"
        )
    x = np.ascontiguousarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D (n_samples, n_features)")
    if dimensions < 1:
        raise ValueError("dimensions must be >= 1")
    if dimensions > x.shape[1]:
        raise ValueError(
            f"dimensions ({dimensions}) cannot exceed n_features ({x.shape[1]}) — not a reduction"
        )
    if x.shape[0] < min_points:
        raise ValueError(f"need at least {min_points} point(s) to project")
    return x


def _validate_precomputed_knn(
    precomputed_knn: tuple[IntArray, FloatArray], n: int
) -> tuple[IntArray, FloatArray]:
    """Check a caller-supplied kNN and cast it to the internal dtypes.

    ``precomputed_knn`` is ``(indices, distances)``, each ``(n, k)``. Column 0 must be the point
    itself (distance 0). That is what a self-inclusive exact kNN returns, and what a faiss query
    with the same vectors returns. It is the same convention ``knn_graph`` produces.
    """
    indices, distances = precomputed_knn
    idx = np.ascontiguousarray(indices, dtype=np.int64)
    dst = np.ascontiguousarray(distances, dtype=np.float64)
    if idx.ndim != 2 or idx.shape != dst.shape:
        raise ValueError(
            f"precomputed_knn must be two (n, k) arrays of the same shape; "
            f"got indices {idx.shape} and distances {dst.shape}"
        )
    if idx.shape[0] != n:
        raise ValueError(f"precomputed_knn has {idx.shape[0]} rows, but the input has {n} points")
    if idx.shape[1] < 2:
        raise ValueError("precomputed_knn needs at least 2 neighbours per point (self + 1)")
    return idx, dst


def _original_radius(work: FloatArray, graph: FuzzyGraph) -> FloatArray:
    """densMAP's original-space local radius per vertex, z-scored.

    It is the membership-weighted mean of squared edge distances (in the kNN space), logged and
    then standardised. A faithful port of umap-learn's density setup. The z-score makes it robust
    to whether the graph stores one or both directions of each edge (both scale numerator and
    denominator equally).
    """
    diff = work[graph.head] - work[graph.tail]
    edge_d2: FloatArray = np.einsum("ij,ij->i", diff, diff)
    mu = graph.weight
    n = work.shape[0]
    ro_num = np.bincount(graph.head, weights=mu * edge_d2, minlength=n) + np.bincount(
        graph.tail, weights=mu * edge_d2, minlength=n
    )
    mu_sum = np.bincount(graph.head, weights=mu, minlength=n) + np.bincount(
        graph.tail, weights=mu, minlength=n
    )
    ro = np.log(1e-8 + ro_num / mu_sum)
    sd = ro.std()
    # all-equal radii (e.g. all-duplicate rows, a perfectly regular graph) give std 0; the
    # z-score is then 0/0 = NaN and poisons the whole embedding. A flat radius field is 0.
    return (ro - ro.mean()) / sd if sd > 0.0 else np.zeros_like(ro)


def _pca_reduce(x: FloatArray, pca_dim: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reduce ``x`` to ``pca_dim`` principal components (deterministic, exact SVD).

    The kNN cost is O(n^2 * d). At 1024-d that d dominates. Projecting onto the top ~100
    components first shrinks it ~10x with little quality loss. This works because encoder/CLS
    embeddings live on a low-rank subspace. Returns ``(reduced, mean, components)`` so
    ``transform`` can project new points the same way.
    """
    mean = np.ascontiguousarray(x.mean(axis=0))
    centered = x - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.ascontiguousarray(vt[:pca_dim])  # (pca_dim, n_features)
    reduced = np.ascontiguousarray(centered @ components.T, dtype=np.float64)
    return reduced, mean, components


_MODEL_FORMAT = 1  # bump when the serialised fields change; from_npz checks it and errors clearly


@dataclass(frozen=True)
class UMAPModel:
    """A fitted layout plus everything ``transform`` needs to place new points.

    ``train`` is kept because placing a new point needs its nearest neighbours in the
    training data. Those neighbours must be found in the same space the fit used: the
    PCA-reduced space when ``pca_dim`` was set. ``embedding`` is the frozen training layout
    that new points are positioned against.
    """

    train: Float32Array
    embedding: FloatArray
    a: float
    b: float
    n_neighbors: int
    dimensions: int
    metric: str
    random_state: int
    negative_sample_rate: int
    n_features: int
    pca_mean: FloatArray | None = None
    pca_components: FloatArray | None = None

    def to_npz(self, file: str | os.PathLike[str] | BinaryIO) -> None:
        """Save to a versioned ``.npz``. Portable across releases, unlike ``pickle``, which
        welds the model to the exact dataclass layout. Load it back with :meth:`from_npz`."""
        np.savez(
            file,
            format_version=np.int64(_MODEL_FORMAT),
            train=self.train,
            embedding=self.embedding,
            scalars=np.array([self.a, self.b], dtype=np.float64),
            ints=np.array(
                [
                    self.n_neighbors,
                    self.dimensions,
                    self.random_state,
                    self.negative_sample_rate,
                    self.n_features,
                ],
                dtype=np.int64,
            ),
            metric=np.array(self.metric),
            has_pca=np.array(self.pca_mean is not None),
            pca_mean=self.pca_mean if self.pca_mean is not None else np.zeros(0),
            pca_components=self.pca_components if self.pca_components is not None else np.zeros(0),
        )

    @classmethod
    def from_npz(cls, file: str | os.PathLike[str] | BinaryIO) -> UMAPModel:
        """Load a model written by :meth:`to_npz`. If the file was written by an incompatible
        release the error names both format versions rather than failing obscurely."""
        with np.load(file, allow_pickle=False) as data:
            written = int(data["format_version"])
            if written != _MODEL_FORMAT:
                raise ValueError(
                    f"UMAPModel .npz is format {written}; this fastumap reads format "
                    f"{_MODEL_FORMAT} — re-save the model with a matching version."
                )
            scalars = data["scalars"]
            ints = data["ints"]
            has_pca = bool(data["has_pca"])
            return cls(
                train=np.ascontiguousarray(data["train"], dtype=np.float32),
                embedding=np.ascontiguousarray(data["embedding"], dtype=np.float64),
                a=float(scalars[0]),
                b=float(scalars[1]),
                n_neighbors=int(ints[0]),
                dimensions=int(ints[1]),
                metric=str(data["metric"]),
                random_state=int(ints[2]),
                negative_sample_rate=int(ints[3]),
                n_features=int(ints[4]),
                pca_mean=np.ascontiguousarray(data["pca_mean"], dtype=np.float64)
                if has_pca
                else None,
                pca_components=(
                    np.ascontiguousarray(data["pca_components"], dtype=np.float64)
                    if has_pca
                    else None
                ),
            )


@with_blas_limit
def fit(
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
    chunk_count: int = 1,
    pca_dim: int | None = None,
    knn: str = "auto",
    precomputed_knn: tuple[IntArray, FloatArray] | None = None,
    init: FloatArray | None = None,
    y: IntArray | None = None,
    target_weight: float = 0.5,
    densmap: bool = False,
    dens_lambda: float = 2.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
) -> UMAPModel:
    """Fit a UMAP layout and return a :class:`UMAPModel` (use ``.embedding`` for the coords).

    Deterministic given ``random_state``. See :func:`umap_project` for the parameters.

    ``pca_dim`` (default ``None`` = off) pre-reduces the input to that many principal
    components before the kNN. This speeds up high-dimensional inputs. ``transform``
    projects new points through the same basis.

    ``precomputed_knn`` skips the built-in neighbour search. Pass ``(indices, distances)``, each
    ``(n, k)`` with column 0 the point itself, e.g. a faiss query you already ran against your own
    index. Then ``n_neighbors``, ``metric`` and ``knn`` no longer affect the graph, and it cannot
    be combined with ``pca_dim``. ``transform`` still uses ``matrix``'s own kNN, so pass the kNN of
    ``matrix`` (in ``matrix``'s space) if you also want ``transform``.

    ``init`` overrides the spectral start with a given ``(n, dimensions)`` embedding. Pass
    the previous view's coordinates so a refreshed layout stays comparable, instead of
    rotating or reflecting. This is a warm start: the SGD still refines, so lower
    ``n_epochs`` for a lighter touch. Place any new rows yourself.

    ``y`` (categorical class labels, one per point, ``-1`` for unlabelled) turns on
    supervised projection: same-label points attract, weighted by ``target_weight`` in
    ``[0, 1]`` (default 0.5; 1.0 severs every inter-class edge). Default ``y=None`` is
    unchanged unsupervised UMAP.
    """
    x = _validate(matrix, dimensions)
    n, n_features = x.shape
    pca_mean: FloatArray | None = None
    pca_components: FloatArray | None = None
    if precomputed_knn is not None and pca_dim is not None:
        raise ValueError("pass either pca_dim or precomputed_knn, not both (pca moves the space)")
    work = x
    if pca_dim is not None and n_features > pca_dim:
        work, pca_mean, pca_components = _pca_reduce(x, pca_dim)
    k = min(n_neighbors, n)

    if precomputed_knn is not None:
        knn_indices, knn_dists = _validate_precomputed_knn(precomputed_knn, n)
    else:
        knn_indices, knn_dists = knn_graph(work, k, metric=metric, mode=knn)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    if y is not None:
        graph = categorical_intersection(
            graph, np.asarray(y, dtype=np.int64), target_weight=target_weight
        )
    dens_R = _original_radius(work, graph) if densmap else None
    rng = np.random.default_rng(random_state)
    if init is not None:
        if init.shape != (n, dimensions):
            raise ValueError(f"init must have shape ({n}, {dimensions}), got {tuple(init.shape)}")
        start = np.ascontiguousarray(init, dtype=np.float64)
    else:
        start = spectral_layout(graph, dimensions, rng=rng)
    a, b = find_ab_params(spread, min_dist)
    epochs = _default_epochs(n) if n_epochs is None else n_epochs
    if densmap:
        epochs += 200  # umap-learn adds a density-off warmup budget on top
    embedding = optimize_layout(
        start,
        graph,
        epochs,
        a=a,
        b=b,
        rng=rng,
        negative_sample_rate=negative_sample_rate,
        chunk_count=chunk_count,
        dens_lambda=dens_lambda if densmap else 0.0,
        dens_frac=dens_frac,
        dens_var_shift=dens_var_shift,
        dens_R=dens_R,
    )
    return UMAPModel(
        # Store the training set (only used by transform's kNN) as float32. This halves the
        # cached-model footprint on memory-constrained hosts (Lambda/ECS). transform upcasts
        # to float64 for the distance compute, so quality is unaffected.
        train=np.ascontiguousarray(work, dtype=np.float32),
        embedding=embedding,
        a=a,
        b=b,
        n_neighbors=n_neighbors,
        dimensions=dimensions,
        metric=metric,
        random_state=random_state,
        negative_sample_rate=negative_sample_rate,
        n_features=n_features,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )


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
    chunk_count: int = 1,
    pca_dim: int | None = None,
    knn: str = "auto",
    precomputed_knn: tuple[IntArray, FloatArray] | None = None,
    init: FloatArray | None = None,
    y: IntArray | None = None,
    target_weight: float = 0.5,
    densmap: bool = False,
    dens_lambda: float = 2.0,
) -> FloatArray:
    """Project ``matrix`` (n_samples, n_features) into ``dimensions`` output dims.

    Use 2 or 3 for a picture. Use a mid dimension (~10) when the goal is to *cluster* the
    result: UMAP's own guidance is to reduce to ~10 dims, not 2, before clustering, so the
    layout keeps more structure. ``dimensions`` must be at least 1 and at most ``n_features``.

    Deterministic given ``random_state``: the spectral start vector is pinned and the
    negative sampler is seeded, so the same input and seed give bit-identical output.

    ``metric`` is ``"euclidean"`` or ``"cosine"``. ``chunk_count > 1`` enables partial
    in-epoch feedback (higher local overlap, slower). ``pca_dim`` pre-reduces the input to
    that many principal components before the kNN, to speed up high-dimensional inputs.

    ``densmap=True`` turns on densMAP: it keeps dense regions dense and sparse regions sparse,
    instead of UMAP's usual flattening of local density. ``dens_lambda`` (default 2.0) is the
    strength. It runs on the numpy path (the native kernel does plain UMAP), so it is slower.
    See the README.
    """
    return fit(
        matrix,
        dimensions,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        negative_sample_rate=negative_sample_rate,
        random_state=random_state,
        metric=metric,
        chunk_count=chunk_count,
        pca_dim=pca_dim,
        knn=knn,
        precomputed_knn=precomputed_knn,
        init=init,
        y=y,
        target_weight=target_weight,
        densmap=densmap,
        dens_lambda=dens_lambda,
    ).embedding


@overload
def transform(
    model: UMAPModel,
    matrix: FloatArray,
    *,
    n_epochs: int | None = ...,
    return_distances: Literal[False] = ...,
) -> FloatArray: ...
@overload
def transform(
    model: UMAPModel,
    matrix: FloatArray,
    *,
    n_epochs: int | None = ...,
    return_distances: Literal[True],
) -> tuple[FloatArray, FloatArray]: ...
@with_blas_limit
def transform(
    model: UMAPModel,
    matrix: FloatArray,
    *,
    n_epochs: int | None = None,
    return_distances: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Place new points into ``model``'s existing layout without refitting it.

    New points are attracted to their nearest neighbours in the training set and repelled
    from random training points. The training layout stays fixed, so the picture is stable
    across calls. Deterministic given the model and input. Returns ``(n_new, dimensions)``.

    With ``return_distances=True`` it also returns, per new point, the distance to its
    nearest training neighbour. This is a per-point measure of fit. Watch it against the
    training set's own mean nearest-neighbour distance. Points landing 2-3x further out are
    extrapolations. A rising batch mean is the signal to refit (see the README).
    """
    # placing points into a fixed layout is well-defined for a single point (one query per
    # request is a primary serving path); only fit/spectral_project need the n>=2 floor.
    new_x = _validate(matrix, model.dimensions, min_points=1)
    if new_x.shape[1] != model.n_features:
        raise ValueError(
            f"new points have {new_x.shape[1]} features, model was fit on {model.n_features}"
        )
    n_new = new_x.shape[0]
    n_train = model.train.shape[0]
    k = min(model.n_neighbors, n_train)

    # Project new points through the same PCA basis the fit used, so kNN happens in the
    # same reduced space as model.train.
    if model.pca_components is not None and model.pca_mean is not None:
        new_work = np.ascontiguousarray((new_x - model.pca_mean) @ model.pca_components.T)
    else:
        new_work = new_x

    # train is stored float32 to halve the cached footprint; upcast for full-precision kNN.
    train = np.ascontiguousarray(model.train, dtype=np.float64)
    knn_indices, knn_dists = knn_between(new_work, train, k, metric=model.metric)
    sigma, rho = smooth_knn_dist(knn_dists, k)
    # exclude_self=False: query and reference are different sets, so index equality between
    # a new point and a training neighbour is a coincidence, not a self-edge to drop.
    rows, cols, data = compute_membership(knn_indices, knn_dists, sigma, rho, exclude_self=False)

    init = _weighted_average_init(rows, cols, data, model.embedding, n_new, model.dimensions)
    epochs = max(30, _default_epochs(n_train) // 3) if n_epochs is None else n_epochs
    rng = np.random.default_rng(model.random_state)
    coords = optimize_transform(
        init,
        model.embedding,
        rows,
        cols,
        data,
        epochs,
        a=model.a,
        b=model.b,
        rng=rng,
        negative_sample_rate=model.negative_sample_rate,
    )
    if return_distances:
        # knn_dists is sorted ascending, so column 0 is the nearest training neighbour.
        return coords, np.ascontiguousarray(knn_dists[:, 0])
    return coords


def _weighted_average_init(
    rows: IntArray,
    cols: IntArray,
    data: FloatArray,
    train_embedding: FloatArray,
    n_new: int,
    dim: int,
) -> FloatArray:
    """Seed each new point at the membership-weighted mean of its neighbours' positions."""
    init = np.zeros((n_new, dim), dtype=np.float64)
    for d in range(dim):
        init[:, d] = np.bincount(rows, weights=data * train_embedding[cols, d], minlength=n_new)
    weight_sum = np.bincount(rows, weights=data, minlength=n_new)
    nonzero = weight_sum > 0.0
    init[nonzero] /= weight_sum[nonzero, None]
    return init


@with_blas_limit
def spectral_project(
    matrix: FloatArray,
    dimensions: int = 2,
    *,
    n_neighbors: int = 15,
    random_state: int = 42,
    metric: str = "euclidean",
    pca_dim: int | None = None,
    knn: str = "auto",
) -> FloatArray:
    """Just the spectral initialisation: kNN graph, fuzzy set, spectral embedding.

    Useful on its own as a fast, deterministic layout, and as the seed the full
    optimisation refines. ``pca_dim`` pre-reduces the input before the kNN.
    """
    x = _validate(matrix, dimensions)
    n, n_features = x.shape
    work = x
    if pca_dim is not None and n_features > pca_dim:
        work, _mean, _components = _pca_reduce(x, pca_dim)
    k = min(n_neighbors, n)
    knn_indices, knn_dists = knn_graph(work, k, metric=metric, mode=knn)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    rng = np.random.default_rng(random_state)
    return spectral_layout(graph, dimensions, rng=rng)
