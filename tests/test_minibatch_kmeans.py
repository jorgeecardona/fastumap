"""Mini-batch k-means — roadmap Tier 2 (constrained-CPU clustering)."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import _kmeans
from fastumap._types import FloatArray
from fastumap.clustering import kmeans

# The k-means internals this file exercises. Same pattern as test_perf_invariants.py.
_inertia = _kmeans.inertia
_MINIBATCH_MIN: int = _kmeans.MINIBATCH_MIN


def _blobs(n: int = 20_000, dim: int = 32, k: int = 16, seed: int = 0) -> FloatArray:
    """WELL-SEPARATED clusters (centres 8 sigma apart). This is mini-batch's adversarial case, not
    its normal one — see test_minibatch_struggles_on_well_separated_blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    return np.ascontiguousarray(centers[labels] + rng.normal(size=(n, dim)), dtype=np.float64)


def _overlapping(n: int = 20_000, dim: int = 32, k: int = 16, seed: int = 0) -> FloatArray:
    """Clusters that genuinely touch (centres 1.5 sigma apart) — the shape of real embeddings."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=1.5, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    return np.ascontiguousarray(centers[labels] + rng.normal(size=(n, dim)), dtype=np.float64)


def test_minibatch_matches_lloyd_on_overlapping_data() -> None:
    """The bar from the roadmap: inertia within ~5% of full Lloyd. MEASURED at 1.3% on MNIST
    60000x784 with k=64, and it holds here on overlapping synthetic data too."""
    x = _overlapping()
    mb_labels, mb_centroids = kmeans(x, 16, batch_size=2000)
    ll_labels, ll_centroids = kmeans(x, 16, batch_size=0)
    assert len(set(mb_labels.tolist())) == 16
    mb, ll = _inertia(x, mb_labels, mb_centroids), _inertia(x, ll_labels, ll_centroids)
    assert mb <= ll * 1.05


def test_minibatch_matches_lloyd_on_well_separated_blobs() -> None:
    """Mini-batch reaches full-Lloyd inertia on well-separated blobs, and n_init keeps it there.

    With plain k-means++ seeding, mini-batch was 4.8x worse here (20000x32, 16 blobs 8 sigma
    apart): a centroid seeded in a neighbour's blob could not walk back. Greedy k-means++ seeding
    (#30) no longer puts two seeds in one blob, so the single default run already matches Lloyd.
    """
    x = _blobs()
    ll_labels, ll_centroids = kmeans(x, 16, batch_size=0)
    ll = _inertia(x, ll_labels, ll_centroids)

    mb_labels, mb_centroids = kmeans(x, 16, batch_size=2000)
    assert _inertia(x, mb_labels, mb_centroids) <= ll * 1.05
    assert len(set(mb_labels.tolist())) == 16

    r_labels, r_centroids = kmeans(x, 16, batch_size=2000, n_init=5)
    assert _inertia(x, r_labels, r_centroids) <= ll * 1.05


def test_minibatch_n_init_is_deterministic() -> None:
    x = _blobs()
    first = kmeans(x, 16, batch_size=2000, n_init=5, random_state=0)[0]
    second = kmeans(x, 16, batch_size=2000, n_init=5, random_state=0)[0]
    assert np.array_equal(first, second)


def test_minibatch_rejects_bad_n_init() -> None:
    x = _blobs(n=2000)
    with pytest.raises(ValueError, match="n_init must be >= 1"):
        kmeans(x, 16, batch_size=500, n_init=0)


def test_minibatch_is_deterministic_across_calls() -> None:
    """Same seed, same labels — the batch draw and the within-batch update order are both fixed,
    so the thread count cannot change the answer."""
    x = _blobs(n=5000)
    first, _ = kmeans(x, 8, batch_size=500, random_state=7)
    second, _ = kmeans(x, 8, batch_size=500, random_state=7)
    assert np.array_equal(first, second)


def test_batch_size_zero_forces_full_lloyd() -> None:
    """Escape hatch: 0 means full Lloyd at any n, so a caller can opt out of the auto switch."""
    x = _blobs(n=5000)
    forced, _ = kmeans(x, 8, batch_size=0, random_state=3)
    default, _ = kmeans(x, 8, random_state=3)  # n is below _MINIBATCH_MIN, so also full Lloyd
    assert np.array_equal(forced, default)


def test_auto_switch_is_by_n_not_by_core_count() -> None:
    """Below the threshold the default must be full Lloyd, and identical to forcing it. This is
    what keeps labels reproducible across machines — the switch never reads the core count."""
    x = _blobs(n=1000)
    assert x.shape[0] < _MINIBATCH_MIN
    assert np.array_equal(
        kmeans(x, 8, random_state=1)[0], kmeans(x, 8, batch_size=0, random_state=1)[0]
    )


def test_minibatch_labels_every_row() -> None:
    """The final pass is blocked, so an n that is not a multiple of the block must still come back
    fully labelled."""
    x = _blobs(n=7777, dim=16, k=8)
    labels, _ = kmeans(x, 8, batch_size=1000)
    assert labels.shape == (7777,)
    assert labels.min() >= 0 and labels.max() < 8


def test_negative_batch_size_is_rejected() -> None:
    x = _blobs(n=500)
    with pytest.raises(ValueError, match="batch_size must be >= 0"):
        kmeans(x, 4, batch_size=-1)
