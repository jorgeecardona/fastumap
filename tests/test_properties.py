"""Property-based invariants (roadmap: quality).

hypothesis generates arbitrary valid inputs; these assert invariants that must hold for ALL of
them, catching edge cases the fixed-fixture tests miss. Sizes are bounded and deadlines disabled
so the projection SGD has room to run without the suite dragging.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import fastumap.clustering as clustering
from fastumap import cluster, umap_project
from fastumap._types import FloatArray

_UNIT_ROWS = clustering._unit_rows  # pyright: ignore[reportPrivateUsage]


def _matrices(
    min_n: int, max_n: int, min_d: int = 2, max_d: int = 8
) -> st.SearchStrategy[FloatArray]:
    """(n, d) float64 arrays with finite, moderate entries."""
    return hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(min_n, max_n), st.integers(min_d, max_d)),
        elements=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False, width=64),
    )


@settings(max_examples=25, deadline=None)
@given(x=_matrices(20, 120), k=st.integers(2, 6))
def test_cluster_labels_are_always_valid(x: FloatArray, k: int) -> None:
    k = min(k, x.shape[0])
    labels = cluster(x, method="kmeans", n_clusters=k, random_state=0)
    assert labels.shape == (x.shape[0],)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < k
    assert len(np.unique(labels)) <= k


@settings(max_examples=15, deadline=None)
@given(x=_matrices(10, 60))
def test_projection_is_finite_and_shaped(x: FloatArray) -> None:
    xy = umap_project(x, 2, random_state=0)
    assert xy.shape == (x.shape[0], 2)
    assert bool(np.all(np.isfinite(xy)))


@settings(max_examples=25, deadline=None)
@given(x=_matrices(30, 120), k=st.integers(2, 5))
def test_cosine_equals_euclidean_on_unit_rows(x: FloatArray, k: int) -> None:
    # drop near-zero rows (their unit direction is ill-defined) so the invariant is well-posed
    x = x[np.sqrt((x * x).sum(axis=1)) > 1e-6]
    if x.shape[0] < 2:
        return  # nothing left to cluster after dropping degenerate rows
    k = min(k, x.shape[0])
    unit = _UNIT_ROWS(x)  # exactly what cluster(metric="cosine") normalises to internally
    euclid_on_unit = cluster(
        unit, method="kmeans", n_clusters=k, metric="euclidean", random_state=0
    )
    cosine_on_raw = cluster(x, method="kmeans", n_clusters=k, metric="cosine", random_state=0)
    assert np.array_equal(euclid_on_unit, cosine_on_raw)


@settings(max_examples=20, deadline=None)
@given(x=_matrices(20, 80), seed=st.integers(0, 10_000))
def test_clustering_is_deterministic_for_any_seed(x: FloatArray, seed: int) -> None:
    k = min(4, x.shape[0])
    a = cluster(x, method="kmeans", n_clusters=k, random_state=seed)
    b = cluster(x, method="kmeans", n_clusters=k, random_state=seed)
    assert np.array_equal(a, b)
