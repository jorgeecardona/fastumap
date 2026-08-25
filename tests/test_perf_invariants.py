"""Machine-independent performance guards (roadmap: quality).

These assert *work done* — how many passes k-means makes, and how big the largest intermediate
gets — not wall-clock. So they pass identically on a fast laptop or a swap-thrashed CI runner, yet
still fail on a real regression: a change that doubles the Lloyd iterations, or that materialises
the whole n-by-k distance matrix instead of blocking it, breaks them. The loose wall-clock ceilings
in ``test_time_budget.py`` stay only as a gross-regression smoke check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fastumap.clustering as clustering
from fastumap._types import FloatArray, IntArray

# Pristine references captured at import, before any test patches them, so a spy always chains to
# the real implementation regardless of what is currently installed on the module.
_REAL_ASSIGN = clustering._assign  # pyright: ignore[reportPrivateUsage]
_REAL_ARGMIN = np.argmin
_BLOCK: int = clustering._BLOCK  # pyright: ignore[reportPrivateUsage]


def _blobs(n: int, d: int, k: int, *, seed: int = 0, spread: float = 8.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * spread
    y = rng.integers(0, k, n)
    return (centers[y] + rng.standard_normal((n, d))).astype(np.float64)


def _count_assign_passes(x: FloatArray, k: int, monkeypatch: pytest.MonkeyPatch) -> int:
    """Number of nearest-centroid passes k-means makes: one for the k-means++ seeding, then one
    per Lloyd iteration. A host-independent proxy for how much work convergence took."""
    calls = 0

    def counting_assign(x_: FloatArray, centroids: FloatArray) -> IntArray:
        nonlocal calls
        calls += 1
        return _REAL_ASSIGN(x_, centroids)

    monkeypatch.setattr(clustering, "_assign", counting_assign)
    clustering.kmeans(x, k, random_state=0)
    return calls


def _max_distance_block_rows(
    x: FloatArray, centroids: FloatArray, monkeypatch: pytest.MonkeyPatch
) -> int:
    """Largest first-dimension of any 2-D array handed to ``np.argmin`` during assignment — i.e.
    how many rows of the distance matrix exist at once. Blocking pins this at ``_BLOCK``."""
    max_rows = 0

    def spy_argmin(a: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal max_rows
        arr = np.asarray(a)
        if arr.ndim == 2:
            max_rows = max(max_rows, int(arr.shape[0]))
        return _REAL_ARGMIN(a, *args, **kwargs)

    monkeypatch.setattr(np, "argmin", spy_argmin)
    _REAL_ASSIGN(x, centroids)
    return max_rows


def test_kmeans_converges_in_a_bounded_number_of_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Well-separated blobs settle in a handful of Lloyd steps; measured 3 passes (init + 2
    # iterations). Bound at 5 so a change that doubles the iteration count fails, while a +/-1
    # convergence wobble across numpy versions does not. n < _ACCEL_MIN, so this is the numpy path.
    passes = _count_assign_passes(_blobs(3000, 16, 6), 6, monkeypatch)
    assert passes <= 5, f"k-means used {passes} assignment passes to converge (bound 5)"


def test_assignment_is_blocked_never_the_full_n_by_k_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # n is 40k, far above _BLOCK: an unblocked assignment would build a 40000-row distance matrix.
    # The blocked path never holds more than _BLOCK rows at once.
    n, k, d = 40_000, 50, 16
    x = _blobs(n, d, k)
    centroids = x[:k].copy()
    max_rows = _max_distance_block_rows(x, centroids, monkeypatch)
    assert 0 < max_rows <= _BLOCK, (
        f"assignment held a {max_rows}-row distance block; _BLOCK={_BLOCK} and "
        f"n={n} — the full n-by-k matrix is being materialised"
    )


def test_assignment_memory_bound_is_independent_of_n(monkeypatch: pytest.MonkeyPatch) -> None:
    # The machine-independent scaling claim: quadruple n and the peak distance-block stays _BLOCK,
    # so peak memory does not grow with the input. A quadratic assignment would fail this.
    k, d = 40, 16
    small = _blobs(20_000, d, k)
    large = _blobs(80_000, d, k)
    rows_small = _max_distance_block_rows(small, small[:k].copy(), monkeypatch)
    rows_large = _max_distance_block_rows(large, large[:k].copy(), monkeypatch)
    assert rows_small == rows_large == _BLOCK
