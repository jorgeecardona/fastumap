"""Hierarchical density clustering (HDBSCAN) — roadmap Tier 2."""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import cluster, dbscan, hdbscan
from fastumap._types import FloatArray, IntArray


def _blobs(n: int = 800, d: int = 8, k: int = 4, seed: int = 0) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * 12.0
    y = rng.integers(0, k, n)
    x = (centers[y] + rng.standard_normal((n, d))).astype(np.float64)
    return x, y.astype(np.int64)


def _mixed_density(seed: int = 0) -> FloatArray:
    """A tight blob beside a diffuse one. The diffuse cluster's own point spacing is larger than
    the gap between the two, so no single eps can split them and keep the diffuse one whole."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((200, 2)) * 0.05
    diffuse = rng.standard_normal((200, 2)) * 3.0 + np.array([9.0, 0.0])
    return np.ascontiguousarray(np.vstack([dense, diffuse]), dtype=np.float64)


def _purity(true: IntArray, pred: IntArray) -> float:
    mask = pred != -1
    if not mask.any():
        return 0.0
    total = sum(
        int(np.bincount(true[mask][pred[mask] == c]).max()) for c in set(pred[mask].tolist())
    )
    return total / int(mask.sum())


def test_hdbscan_recovers_blob_count_without_being_told_k() -> None:
    x, _ = _blobs()
    labels = hdbscan(x, min_cluster_size=25)
    assert len(set(labels.tolist()) - {-1}) == 4


def test_hdbscan_separates_densities_no_single_eps_can() -> None:
    x = _mixed_density()
    labels = hdbscan(x, min_cluster_size=15)
    assert len(set(labels.tolist()) - {-1}) == 2
    assert len(set(labels[:200].tolist())) == 1  # the tight blob comes out whole...
    dense_label = int(labels[0])
    assert dense_label != -1
    diffuse = labels[200:]
    kept = diffuse[diffuse != -1]
    # ...and the diffuse one is one cluster plus its own outliers (a stray point or two land in
    # the tight blob; scikit-learn's HDBSCAN misplaces the same ones).
    counts = np.bincount(kept)
    assert counts.argmax() != dense_label
    assert counts.max() / counts.sum() > 0.95
    assert (diffuse != -1).mean() > 0.7

    # the claim in the docstring, checked: DBSCAN needs one eps for both densities and there
    # isn't one. Every eps either merges the two or shreds the diffuse cluster.
    for eps in np.linspace(0.05, 8.0, 24):
        d_labels = dbscan(x, eps=float(eps), min_samples=10)
        split = len(set(d_labels[:200].tolist())) == 1 and len(set(d_labels[200:].tolist())) == 1
        assert not (split and (d_labels != -1).mean() > 0.9 and d_labels[0] != d_labels[200])


def test_hdbscan_finds_non_convex_clusters_kmeans_cannot() -> None:
    datasets = pytest.importorskip("sklearn.datasets")
    xm, ym = datasets.make_moons(n_samples=600, noise=0.05, random_state=0)
    labels = hdbscan(np.ascontiguousarray(xm, dtype=np.float64), min_cluster_size=25)
    assert len(set(labels.tolist()) - {-1}) == 2
    assert _purity(ym.astype(np.int64), labels) > 0.95


def test_hdbscan_agrees_with_sklearn_hdbscan() -> None:
    """Independent reimplementation, so the check is agreement on labels, not identical code."""
    sk_cluster = pytest.importorskip("sklearn.cluster")
    metrics = pytest.importorskip("sklearn.metrics")
    x = _mixed_density()
    ours = hdbscan(x, min_cluster_size=15)
    theirs = sk_cluster.HDBSCAN(min_cluster_size=15, copy=True).fit_predict(x)
    assert metrics.adjusted_rand_score(theirs, ours) > 0.95


def test_hdbscan_labels_outliers_as_noise() -> None:
    rng = np.random.default_rng(0)
    two_blobs = np.vstack(
        [rng.standard_normal((300, 4)) * 0.3, rng.standard_normal((300, 4)) * 2.0 + 25.0]
    )
    outliers = rng.uniform(-40.0, 60.0, size=(30, 4))
    x = np.ascontiguousarray(np.vstack([two_blobs, outliers]), dtype=np.float64)
    labels = hdbscan(x, min_cluster_size=15)
    assert (labels[-30:] == -1).mean() > 0.5
    assert (labels[:600] == -1).mean() < 0.05


def test_hdbscan_returns_all_noise_when_there_is_only_one_group() -> None:
    """The root is never a cluster: HDBSCAN answers which groups stand out from each other, not
    whether a group exists. scikit-learn's allow_single_cluster=False default does the same."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((300, 4)) * 0.3, dtype=np.float64)
    assert set(hdbscan(x, min_cluster_size=15).tolist()) == {-1}


def test_hdbscan_min_cluster_size_absorbs_small_groups() -> None:
    rng = np.random.default_rng(1)
    big = rng.standard_normal((300, 3)) * 0.5
    other = rng.standard_normal((300, 3)) * 0.5 + np.array([40.0, 0.0, 0.0])
    small = rng.standard_normal((12, 3)) * 0.5 + np.array([20.0, 0.0, 0.0])
    x = np.ascontiguousarray(np.vstack([big, other, small]), dtype=np.float64)
    assert len(set(hdbscan(x, min_cluster_size=5).tolist()) - {-1}) == 3  # 12 >= 5: a cluster
    assert len(set(hdbscan(x, min_cluster_size=30).tolist()) - {-1}) == 2  # 12 < 30: noise


def test_hdbscan_handles_a_disconnected_knn_graph() -> None:
    """Far-apart groups share no neighbours, so the kNN graph is a forest, not a tree."""
    rng = np.random.default_rng(2)
    parts = [rng.standard_normal((60, 3)) * 0.2 + np.array([i * 500.0, 0.0, 0.0]) for i in range(3)]
    x = np.ascontiguousarray(np.vstack(parts), dtype=np.float64)
    labels = hdbscan(x, min_cluster_size=10, n_neighbors=10)
    assert len(set(labels.tolist()) - {-1}) == 3


def test_hdbscan_is_deterministic() -> None:
    x, _ = _blobs(n=400)
    assert np.array_equal(hdbscan(x, min_cluster_size=15), hdbscan(x, min_cluster_size=15))


def test_hdbscan_via_cluster_matches_standalone() -> None:
    x, _ = _blobs(n=400)
    assert np.array_equal(
        cluster(x, method="hdbscan", min_cluster_size=15),
        hdbscan(x, min_cluster_size=15),
    )


def test_hdbscan_cosine_runs_on_unit_rows() -> None:
    x, y = _blobs(n=400, d=16)
    labels = hdbscan(x, min_cluster_size=15, metric="cosine")
    assert _purity(y, labels) > 0.9


def test_hdbscan_rejects_bad_params() -> None:
    x, _ = _blobs(n=200)
    with pytest.raises(ValueError, match="return_centroids is not supported"):
        cluster(x, method="hdbscan", return_centroids=True)
    with pytest.raises(ValueError, match="min_cluster_size must be >= 2"):
        hdbscan(x, min_cluster_size=1)
    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        hdbscan(x, min_samples=0)
    with pytest.raises(ValueError, match="unknown metric"):
        hdbscan(x, metric="manhattan")
    with pytest.raises(ValueError, match="embeddings must be 2-D"):
        hdbscan(np.zeros(10, dtype=np.float64))


def test_cluster_forwards_explicit_n_neighbors() -> None:
    # cluster(method="hdbscan") used to drop n_neighbors silently; an explicit value must now
    # reach hdbscan (the default, n_neighbors=None, still matches the standalone default of 30).
    x, _ = _blobs(n=400)
    assert np.array_equal(
        cluster(x, method="hdbscan", min_cluster_size=15, n_neighbors=10),
        hdbscan(x, min_cluster_size=15, n_neighbors=10),
    )
