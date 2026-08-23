"""Public projection entry points: ``umap_project``, ``spectral_project``, and the
``fit`` / ``transform`` pair for placing new points into an existing layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import FloatArray, IntArray
from .graph import (
    compute_membership,
    find_ab_params,
    fuzzy_simplicial_set,
    smooth_knn_dist,
)
from .layout import optimize_layout, optimize_transform, spectral_layout
from .neighbors import knn, knn_between


def _default_epochs(n: int) -> int:
    # umap-learn's rule: more epochs when the graph is small enough to afford them.
    return 500 if n <= 10_000 else 200


def _validate(matrix: FloatArray, dimensions: int) -> FloatArray:
    # Sparse input is out of scope (see README "Sparse input"): the blocked kNN relies on
    # dense BLAS matmul, and the target workload is dense encoder/CLS embeddings. Detect a
    # sparse matrix by its `toarray` method and fail with a clear pointer rather than the
    # cryptic error np.ascontiguousarray would raise on it.
    if hasattr(matrix, "toarray"):  # scipy.sparse matrices have it; numpy arrays do not
        raise TypeError(
            "fastumap takes a dense numpy array, not a sparse matrix; "
            "densify first with matrix.toarray() (sparse input is not supported — see README)"
        )
    x = np.ascontiguousarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D (n_samples, n_features)")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 points to project")
    return x


def _pca_reduce(x: FloatArray, pca_dim: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reduce ``x`` to ``pca_dim`` principal components (deterministic, exact SVD).

    The kNN cost is O(n^2 * d); at 1024-d that d dominates. Projecting onto the top
    ~100 components first shrinks it ~10x with little quality loss, because encoder/CLS
    embeddings live on a low-rank subspace. Returns ``(reduced, mean, components)`` so
    ``transform`` can project new points the same way.
    """
    mean = np.ascontiguousarray(x.mean(axis=0))
    centered = x - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.ascontiguousarray(vt[:pca_dim])  # (pca_dim, n_features)
    reduced = np.ascontiguousarray(centered @ components.T, dtype=np.float64)
    return reduced, mean, components


@dataclass(frozen=True)
class UMAPModel:
    """A fitted layout plus everything ``transform`` needs to place new points.

    ``train`` is kept because placing a new point requires its nearest neighbours in the
    training data (in the same space the fit used — the PCA-reduced space when ``pca_dim``
    was set). ``embedding`` is the frozen training layout new points are positioned against.
    """

    train: FloatArray
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
) -> UMAPModel:
    """Fit a UMAP layout and return a :class:`UMAPModel` (use ``.embedding`` for the coords).

    Deterministic given ``random_state``. See :func:`umap_project` for the parameters.
    ``pca_dim`` (default ``None`` = off) pre-reduces the input to that many principal
    components before the kNN, which speeds up high-dimensional inputs; ``transform``
    projects new points through the same basis.
    """
    x = _validate(matrix, dimensions)
    n, n_features = x.shape
    pca_mean: FloatArray | None = None
    pca_components: FloatArray | None = None
    work = x
    if pca_dim is not None and n_features > pca_dim:
        work, pca_mean, pca_components = _pca_reduce(x, pca_dim)
    k = min(n_neighbors, n)

    knn_indices, knn_dists = knn(work, k, metric=metric)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    rng = np.random.default_rng(random_state)
    init = spectral_layout(graph, dimensions, rng=rng)
    a, b = find_ab_params(spread, min_dist)
    epochs = _default_epochs(n) if n_epochs is None else n_epochs
    embedding = optimize_layout(
        init,
        graph,
        epochs,
        a=a,
        b=b,
        rng=rng,
        negative_sample_rate=negative_sample_rate,
        chunk_count=chunk_count,
    )
    return UMAPModel(
        train=work,
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
) -> FloatArray:
    """Project ``matrix`` (n_samples, n_features) into ``dimensions`` (2 or 3).

    Deterministic given ``random_state``: the spectral start vector is pinned and the
    negative sampler is seeded, so the same input and seed give bit-identical output.
    ``metric`` is ``"euclidean"`` or ``"cosine"``. ``chunk_count > 1`` enables partial
    in-epoch feedback (higher local overlap, slower). ``pca_dim`` pre-reduces the input to
    that many principal components before the kNN, to speed up high-dimensional inputs.
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
    ).embedding


def transform(model: UMAPModel, matrix: FloatArray, *, n_epochs: int | None = None) -> FloatArray:
    """Place new points into ``model``'s existing layout without refitting it.

    New points are attracted to their nearest neighbours in the training set and repelled
    from random training points; the training layout stays fixed, so the picture is stable
    across calls. Deterministic given the model and input. Returns ``(n_new, dimensions)``.
    """
    new_x = _validate(matrix, model.dimensions)
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

    knn_indices, knn_dists = knn_between(new_work, model.train, k, metric=model.metric)
    sigma, rho = smooth_knn_dist(knn_dists, k)
    # exclude_self=False: query and reference are different sets, so index equality between
    # a new point and a training neighbour is a coincidence, not a self-edge to drop.
    rows, cols, data = compute_membership(knn_indices, knn_dists, sigma, rho, exclude_self=False)

    init = _weighted_average_init(rows, cols, data, model.embedding, n_new, model.dimensions)
    epochs = max(30, _default_epochs(n_train) // 3) if n_epochs is None else n_epochs
    rng = np.random.default_rng(model.random_state)
    return optimize_transform(
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


def spectral_project(
    matrix: FloatArray,
    dimensions: int = 2,
    *,
    n_neighbors: int = 15,
    random_state: int = 42,
    metric: str = "euclidean",
    pca_dim: int | None = None,
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
    knn_indices, knn_dists = knn(work, k, metric=metric)
    graph = fuzzy_simplicial_set(knn_indices, knn_dists)
    rng = np.random.default_rng(random_state)
    return spectral_layout(graph, dimensions, rng=rng)
