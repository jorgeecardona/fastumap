"""Grading metrics: how much structure a projection preserves.

Every quality claim in this project is a number these functions produce. They are
runtime-importable (numpy + scipy only) so a caller can grade a layout without
pulling in umap-learn or scikit-learn.
"""

from __future__ import annotations

import numpy as np

from . import _scipy
from ._types import FloatArray, IntArray
from .neighbors import knn_euclidean


def neighbor_overlap(original: FloatArray, projected: FloatArray, k: int) -> float:
    """Mean share of each point's ``k`` input-space neighbours kept after projection.

    Chance level is about ``k/(n-1)`` — always compare against a random-layout control
    (see :func:`random_layout`).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    orig = np.ascontiguousarray(original, dtype=np.float64)
    proj = np.ascontiguousarray(projected, dtype=np.float64)
    n = orig.shape[0]
    kk = min(k + 1, n)
    orig_idx, _ = knn_euclidean(orig, kk)
    proj_idx, _ = knn_euclidean(proj, kk)

    total = 0
    for i in range(n):
        o = orig_idx[i][orig_idx[i] != i][:k]
        p = proj_idx[i][proj_idx[i] != i][:k]
        total += int(np.intersect1d(o, p, assume_unique=False).shape[0])
    return total / (n * k)


def global_distance_correlation(
    original: FloatArray,
    projected: FloatArray,
    *,
    sample: int = 2000,
    random_state: int = 0,
) -> float:
    """Spearman correlation of all pairwise distances, before vs after projection.

    For n above ``sample`` a deterministic random subset is used so the condensed
    distance vector stays bounded (its length is quadratic in n).
    """
    orig = np.ascontiguousarray(original, dtype=np.float64)
    proj = np.ascontiguousarray(projected, dtype=np.float64)
    n = orig.shape[0]
    if n > sample:
        rng = np.random.default_rng(random_state)
        chosen: IntArray = np.sort(rng.choice(n, size=sample, replace=False))
        orig = orig[chosen]
        proj = proj[chosen]
    return _scipy.spearman(_scipy.pdist_euclidean(orig), _scipy.pdist_euclidean(proj))


def random_layout(n: int, dimensions: int, *, random_state: int = 0) -> FloatArray:
    """A Gaussian random layout — the control every overlap number is read against."""
    rng = np.random.default_rng(random_state)
    return rng.normal(size=(n, dimensions))
