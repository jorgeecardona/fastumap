"""Fuzzy simplicial set: turn a kNN graph into a symmetric weighted graph.

The three UMAP steps live here:
  1. smooth_knn_dist  — per-point local scaling (rho_i, sigma_i)
  2. compute_membership — directed edge weights exp(-(d - rho)/sigma)
  3. fuzzy_simplicial_set — the fuzzy-union symmetrisation, via the scipy shim

Plus find_ab_params (output-kernel fit) and make_epochs_per_sample (edge schedule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _scipy
from ._types import FloatArray, IntArray

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


@dataclass(frozen=True)
class FuzzyGraph:
    """A symmetric weighted graph on ``n`` vertices, held as a COO edge list.

    head/tail/weight are parallel arrays over edges. This is the single data
    structure the layout stage consumes, which keeps scipy's sparse types out of
    the rest of the codebase.
    """

    n: int
    head: IntArray
    tail: IntArray
    weight: FloatArray


def smooth_knn_dist(
    knn_dists: FloatArray,
    k: int,
    *,
    n_iter: int = 64,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Per-point ``(sigma, rho)``.

    ``rho_i`` is the distance to the nearest neighbour with positive distance (so
    every point keeps one edge at full weight and nothing is isolated).
    ``sigma_i`` is binary-searched so ``sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)``,
    then floored at ``MIN_K_DIST_SCALE`` times the local mean distance.

    Only ``local_connectivity == 1.0`` (the UMAP default) is matched exactly; the
    argument is accepted for signature parity.
    """
    _ = local_connectivity
    n = knn_dists.shape[0]
    target = np.log2(k) * bandwidth
    mean_distance = float(knn_dists.mean())

    positive = np.where(knn_dists > 0.0, knn_dists, np.inf)
    rho = positive.min(axis=1)
    rho = np.where(np.isinf(rho), 0.0, rho)

    lo = np.zeros(n, dtype=np.float64)
    hi = np.full(n, np.inf, dtype=np.float64)
    mid = np.ones(n, dtype=np.float64)
    offset = knn_dists[:, 1:] - rho[:, None]  # skip self column, like umap-learn

    for _iteration in range(n_iter):
        val = np.where(offset > 0.0, np.exp(-offset / mid[:, None]), 1.0)
        psum = val.sum(axis=1)
        converged = np.abs(psum - target) < SMOOTH_K_TOLERANCE
        over = psum > target
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
        mid_next = np.where(
            over,
            (lo + hi) / 2.0,
            np.where(np.isinf(hi), mid * 2.0, (lo + hi) / 2.0),
        )
        mid = np.where(converged, mid, mid_next)

    sigma = mid
    mean_ith = knn_dists.mean(axis=1)
    floor = np.where(
        rho > 0.0,
        MIN_K_DIST_SCALE * mean_ith,
        MIN_K_DIST_SCALE * mean_distance,
    )
    sigma = np.maximum(sigma, floor)
    return sigma, rho


def compute_membership(
    knn_indices: IntArray,
    knn_dists: FloatArray,
    sigma: FloatArray,
    rho: FloatArray,
    *,
    exclude_self: bool = True,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Directed edge weights ``exp(-(d - rho)/sigma)``, clamped to 1 when d <= rho.

    Returns COO arrays ``(rows, cols, data)``. With ``exclude_self`` (self-kNN), edges
    where a point is its own neighbour are set to weight 0, matching umap-learn. For
    ``transform`` the neighbours come from a *different* set (the training data), so index
    equality is coincidental and ``exclude_self=False`` keeps those edges.
    """
    n, k = knn_indices.shape
    offset = knn_dists - rho[:, None]
    # np.where evaluates both branches, so clamp the exponent's argument at 0 before the
    # exp: where offset <= 0 the result is discarded anyway, and this avoids a spurious
    # `overflow encountered in exp` warning (-offset can be large positive there).
    weight = np.exp(-np.maximum(offset, 0.0) / sigma[:, None])
    vals = np.where(offset <= 0.0, 1.0, weight)

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = knn_indices.reshape(-1).astype(np.int64)
    data = vals.reshape(-1).astype(np.float64)
    if exclude_self:
        data[rows == cols] = 0.0
    return rows, cols, data


def fuzzy_simplicial_set(
    knn_indices: IntArray,
    knn_dists: FloatArray,
    *,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> FuzzyGraph:
    """Full fuzzy simplicial set from a kNN graph, symmetrised by fuzzy union."""
    n, k = knn_indices.shape
    sigma, rho = smooth_knn_dist(knn_dists, k, local_connectivity=local_connectivity)
    rows, cols, data = compute_membership(knn_indices, knn_dists, sigma, rho)
    head, tail, weight = _scipy.fuzzy_union(rows, cols, data, n, set_op_mix_ratio)
    return FuzzyGraph(n=n, head=head, tail=tail, weight=weight)


def categorical_intersection(
    graph: FuzzyGraph, y: IntArray, *, target_weight: float = 0.5
) -> FuzzyGraph:
    """Categorical supervised UMAP: fold class labels ``y`` into the fuzzy graph.

    Ported from umap-learn 0.5.12 (``discrete_metric_simplicial_set_intersection`` /
    ``fast_intersection`` / ``reset_local_connectivity``); the theory paper (arXiv:1802.03426)
    proposes only the simplicial-set-intersection idea as future work (Sec. 7), so the reference
    implementation is the authority here. See umap-learn's "Supervised Dimension Reduction and
    Metric Learning" docs and McInnes et al., JOSS 3(29):861.

    Same-label edges are kept; edges between different labels are multiplied by
    ``exp(-far_dist)`` with ``far_dist = 2.5 / (1 - target_weight)`` (edges are removed outright
    when ``target_weight >= 1``); edges touching an unlabelled point (``y == -1``) are multiplied
    by ``exp(-1)`` (semi-supervised). The attenuated graph is then row-max-normalised and
    re-symmetrised. ``target_weight`` is in ``[0, 1]`` (default 0.5): 0.0 pushes classes apart the
    least, 1.0 severs every inter-class edge.
    """
    if y.shape[0] != graph.n:
        raise ValueError(f"y must have one label per point ({graph.n}), got {y.shape[0]}")
    far_dist = 2.5 / (1.0 - target_weight) if target_weight < 1.0 else 1.0e12
    yi = y[graph.head]
    yj = y[graph.tail]
    unknown = (yi == -1) | (yj == -1)
    different = (~unknown) & (yi != yj)
    w = graph.weight.copy()
    w[unknown] *= np.exp(-1.0)  # unknown_dist = 1.0, not scaled by target_weight (per umap-learn)
    w[different] *= np.exp(-far_dist)
    head, tail, weight = _scipy.reset_local_connectivity(graph.head, graph.tail, w, graph.n)
    return FuzzyGraph(n=graph.n, head=head, tail=tail, weight=weight)


def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit the output kernel ``1/(1 + a x^(2b))`` to the offset exponential target."""
    xv = np.linspace(0.0, spread * 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    return _scipy.curve_fit_ab(xv, yv)


def make_epochs_per_sample(weights: FloatArray, n_epochs: int) -> FloatArray:
    """Edges per epoch: ``weight.max()/weight``, so strong edges fire every epoch.

    Edges that would never be sampled get -1 (a sentinel the optimiser skips).
    """
    result = np.full(weights.shape[0], -1.0, dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    positive = n_samples > 0.0
    with np.errstate(over="ignore"):
        result[positive] = float(n_epochs) / n_samples[positive]
    # A denormal weight makes the reciprocal overflow to inf; fall back to the documented -1
    # sentinel so the contract holds and we neither warn nor raise under np.seterr(all="raise").
    result[~np.isfinite(result)] = -1.0
    return result
