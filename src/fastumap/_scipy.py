# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Typed shims over the untyped scipy functions fastumap uses.

scipy ships no type stubs, so every call into it is `Unknown` under pyright strict.
Rather than scatter `# pyright: ignore` across the algorithm modules, every scipy
call lives here behind a precisely-typed signature; the file-level pragma above
confines the resulting "unknown type" noise to this one shim. Callers see only
numpy arrays and plain scalars.

scipy submodules are imported lazily inside each function so that `import fastumap`
(and even `import fastumap.projection`) never pays scipy's import cost until a
projection actually runs.
"""

from __future__ import annotations

import numpy as np

from ._types import FloatArray, IntArray


def fuzzy_union(
    rows: IntArray,
    cols: IntArray,
    data: FloatArray,
    n: int,
    set_op_mix_ratio: float,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Symmetrise a directed membership edge list by the fuzzy set union.

    Computes ``W + Wt - W∘Wt`` (for ``set_op_mix_ratio == 1``), the probabilistic
    t-conorm UMAP uses — not an elementwise maximum. Returns the resulting
    symmetric graph as COO arrays with structural zeros removed.
    """
    from scipy import sparse

    w = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    w.eliminate_zeros()
    wt = w.transpose().tocsr()
    prod = w.multiply(wt)
    # sparse on the left of the scalar so the result stays a sparse matrix, not float.
    combined = ((w + wt - prod) * set_op_mix_ratio + prod * (1.0 - set_op_mix_ratio)).tocsr()
    combined.eliminate_zeros()
    coo = combined.tocoo()
    return (
        np.asarray(coo.row, dtype=np.int64),
        np.asarray(coo.col, dtype=np.int64),
        np.asarray(coo.data, dtype=np.float64),
    )


def reset_local_connectivity(
    head: IntArray, tail: IntArray, weight: FloatArray, n: int
) -> tuple[IntArray, IntArray, FloatArray]:
    """umap-learn's ``reset_local_connectivity``: row-max-normalise, then fuzzy-union.

    Divides each row by its largest entry (restoring a 1.0 membership for every point's
    strongest edge after the supervised attenuation), then re-symmetrises with the
    probabilistic t-conorm ``A + At - A∘At``. Structural zeros are dropped.
    """
    from scipy import sparse

    a = sparse.coo_matrix((weight, (head, tail)), shape=(n, n)).tocsr()
    a.eliminate_zeros()
    row_max = np.asarray(a.max(axis=1).toarray(), dtype=np.float64).ravel()
    a = sparse.diags(1.0 / np.where(row_max > 0.0, row_max, 1.0)) @ a
    at = a.transpose().tocsr()
    prod = a.multiply(at)
    combined = (a + at - prod).tocsr()
    combined.eliminate_zeros()
    coo = combined.tocoo()
    return (
        np.asarray(coo.row, dtype=np.int64),
        np.asarray(coo.col, dtype=np.int64),
        np.asarray(coo.data, dtype=np.float64),
    )


def curve_fit_ab(xv: FloatArray, yv: FloatArray) -> tuple[float, float]:
    """Least-squares fit of ``1 / (1 + a * x**(2b))`` to the target curve.

    With the offset-exponential target from spread=1.0, min_dist=0.1 this returns
    the canonical UMAP defaults a=1.5769, b=0.8951.
    """
    from scipy.optimize import curve_fit

    def curve(x: FloatArray, a: float, b: float) -> FloatArray:
        return 1.0 / (1.0 + a * x ** (2.0 * b))

    params, _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0), maxfev=10000)
    return float(params[0]), float(params[1])


def connected_components(n: int, head: IntArray, tail: IntArray) -> tuple[int, IntArray]:
    """Number of connected components of the undirected graph, and per-vertex labels."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components as _cc

    adj = sparse.coo_matrix(
        (np.ones(head.shape[0], dtype=np.float64), (head, tail)), shape=(n, n)
    ).tocsr()
    n_components, labels = _cc(adj, directed=False)
    return int(n_components), np.asarray(labels, dtype=np.int64)


def normalized_adjacency_eigenvectors(
    n: int,
    head: IntArray,
    tail: IntArray,
    weight: FloatArray,
    k: int,
    v0: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Leading ``k`` eigenpairs of the normalised adjacency ``D^-1/2 W D^-1/2``.

    Taking the *largest* eigenvectors of the normalised adjacency avoids the
    shift-invert factorisation that finding the *smallest* eigenvectors of the
    Laplacian would need. ``v0`` is pinned so ARPACK is bit-reproducible.
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    w = sparse.coo_matrix((weight, (head, tail)), shape=(n, n)).tocsr()
    degree = np.asarray(w.sum(axis=1), dtype=np.float64).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    d_mat = sparse.diags(d_inv_sqrt)
    norm = (d_mat @ w @ d_mat).tocsr()
    norm = (norm + norm.T) * 0.5  # symmetrise away floating-point asymmetry
    ncv = min(n, max(2 * k + 1, 20))
    vals, vecs = eigsh(norm, k=k, which="LA", v0=v0, ncv=ncv, maxiter=n * 10, tol=0)
    return np.asarray(vals, dtype=np.float64), np.asarray(vecs, dtype=np.float64)


def pdist_euclidean(matrix: FloatArray) -> FloatArray:
    """Condensed pairwise euclidean distances (upper triangle, row-major)."""
    from scipy.spatial.distance import pdist

    return np.asarray(pdist(matrix, metric="euclidean"), dtype=np.float64)


def spearman(a: FloatArray, b: FloatArray) -> float:
    """Spearman rank correlation between two 1-D samples."""
    from scipy.stats import spearmanr

    result = spearmanr(a, b)
    return float(result.statistic)
