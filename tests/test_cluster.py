"""Clustering — our numpy k-means and spectral, no external clustering libraries."""

import numpy as np
import pytest

from fastumap import cluster
from fastumap.clustering import (
    ChooseKResult,
    ClusterResult,
    assign_clusters,
    choose_k,
    kmeans,
)


def _blobs(n: int = 1500, d: int = 20, k: int = 5, spread: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * spread
    y = rng.integers(0, k, n)
    x = (centers[y] + rng.standard_normal((n, d))).astype(np.float64)
    return x, y


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    total = sum(int(np.bincount(true[pred == c]).max()) for c in np.unique(pred))
    return total / len(true)


@pytest.mark.parametrize("method", ["kmeans", "spectral"])
def test_recovers_separated_blobs(method: str) -> None:
    x, y = _blobs()
    labels = cluster(x, method=method, n_clusters=5, random_state=0)
    assert labels.shape == (x.shape[0],)
    assert _purity(y, labels) > 0.95


def test_kmeans_is_deterministic() -> None:
    x, _ = _blobs()
    a = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    b = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    assert np.array_equal(a, b)


def test_cosine_metric_runs() -> None:
    x, _ = _blobs()
    labels = cluster(x, method="kmeans", n_clusters=5, metric="cosine", random_state=0)
    assert labels.shape == (x.shape[0],)


def test_kmeans_standalone_returns_centroids() -> None:
    x, y = _blobs()
    labels, centroids = kmeans(x, 5, random_state=0)
    assert centroids.shape == (5, x.shape[1])
    assert _purity(y, labels) > 0.95


def test_no_n_clusters_picks_k() -> None:
    # n_clusters=None is the documented default; kmeans/spectral pick k via choose_k rather than
    # refusing the call (GitLab #23). Five well-separated blobs should recover ~5 clusters.
    x, _ = _blobs(n=1500, k=5)
    labels = cluster(x, method="kmeans", random_state=0)
    assert labels.shape == (1500,)
    assert 4 <= len(set(labels.tolist())) <= 6


def test_bad_method_raises() -> None:
    x, _ = _blobs(n=100)
    with pytest.raises(ValueError, match="unknown method"):
        cluster(x, method="meanshift", n_clusters=3)


def test_too_many_clusters_raises() -> None:
    x, _ = _blobs(n=10)
    with pytest.raises(ValueError, match="exceeds"):
        cluster(x, method="kmeans", n_clusters=20)


def test_accel_path_recovers_clusters() -> None:
    # n >= _ACCEL_MIN (50k) routes k-means through the built-in native kernel (fastumap._accel);
    # it gives different (f32) labels than numpy but must still recover well-separated clusters.
    pytest.importorskip("fastumap._accel")
    x, y = _blobs(n=60_000, d=32, k=8)
    labels = cluster(x, method="kmeans", n_clusters=8, random_state=0)
    assert labels.shape == (60_000,)
    assert _purity(y, labels) > 0.95


def test_return_centroids_gives_centroids_and_matching_inertia() -> None:
    x, y = _blobs()
    res = cluster(x, method="kmeans", n_clusters=5, random_state=0, return_centroids=True)
    assert isinstance(res, ClusterResult)
    assert res.centroids.shape == (5, x.shape[1])
    assert res.labels.shape == (x.shape[0],)
    assert res.inertia > 0.0 and np.isfinite(res.inertia)
    # inertia is exactly the sum of squared distances to the assigned centroid
    manual = float(((x - res.centroids[res.labels]) ** 2).sum())
    assert abs(res.inertia - manual) <= 1e-6 * manual
    assert _purity(y, res.labels) > 0.95


def test_return_centroids_default_still_returns_a_plain_label_array() -> None:
    x, _ = _blobs(n=200)
    labels = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    assert not isinstance(labels, ClusterResult)
    assert labels.shape == (200,)


def test_return_centroids_works_for_spectral() -> None:
    x, _ = _blobs()
    res = cluster(x, method="spectral", n_clusters=5, random_state=0, return_centroids=True)
    assert isinstance(res, ClusterResult)
    assert res.centroids.shape == (5, x.shape[1])
    assert np.isfinite(res.inertia)


def test_inertia_decreases_as_k_increases() -> None:
    # the number you compare when picking k: more clusters -> tighter fit -> lower inertia.
    x, _ = _blobs()
    coarse = cluster(x, method="kmeans", n_clusters=3, random_state=0, return_centroids=True)
    fine = cluster(x, method="kmeans", n_clusters=8, random_state=0, return_centroids=True)
    assert fine.inertia < coarse.inertia


def test_assign_clusters_reproduces_fit_labels_and_predicts_held_out() -> None:
    x, _ = _blobs(n=1500)
    res = cluster(x[:1000], method="kmeans", n_clusters=5, random_state=0, return_centroids=True)
    # predicting the training rows reproduces the fitted labels (they ARE nearest-centroid)
    assert np.array_equal(assign_clusters(x[:1000], res.centroids), res.labels)
    # held-out rows get the same label a manual argmin over centroids would give
    held = x[1000:]
    pred = assign_clusters(held, res.centroids)
    manual = np.array(
        [int(np.argmin(((held[i] - res.centroids) ** 2).sum(axis=1))) for i in range(len(held))]
    )
    assert np.array_equal(pred, manual)


def test_assign_clusters_honours_cosine_metric() -> None:
    x, _ = _blobs()
    res = cluster(
        x, method="kmeans", n_clusters=5, metric="cosine", random_state=0, return_centroids=True
    )
    pred = assign_clusters(x, res.centroids, metric="cosine")
    assert np.array_equal(pred, res.labels)


def test_assign_clusters_rejects_mismatched_dims() -> None:
    with pytest.raises(ValueError, match="dim"):
        assign_clusters(np.zeros((10, 4)), np.zeros((3, 5)))


@pytest.mark.parametrize("true_k", [3, 5, 7])
def test_choose_k_recovers_true_k_on_separated_blobs(true_k: int) -> None:
    x, _ = _blobs(n=1500, d=16, k=true_k, spread=8.0)
    k = choose_k(x, k_min=2, k_max=10, random_state=0)
    assert abs(k - true_k) <= 1  # the elbow lands on the true k give or take one


def test_choose_k_subsamples_large_n_and_stays_valid() -> None:
    # above max_samples the search runs on a subsample; the returned k must still be in range.
    x, _ = _blobs(n=25_000, d=8, k=4, spread=8.0)
    k = choose_k(x, k_min=2, k_max=8, max_samples=5_000, random_state=0)
    assert 2 <= k <= 8


def test_choose_k_is_deterministic() -> None:
    x, _ = _blobs(n=1200, d=12, k=5)
    assert choose_k(x, random_state=0) == choose_k(x, random_state=0)


def test_choose_k_rejects_bad_range() -> None:
    x, _ = _blobs(n=100)
    with pytest.raises(ValueError, match="k_min"):
        choose_k(x, k_min=1, k_max=5)


def test_choose_k_n_jobs_matches_serial() -> None:
    # the parallel sweep must give the identical k (each fit is seeded and order-independent).
    x, _ = _blobs(n=1200, d=12, k=5)
    serial = choose_k(x, k_min=2, k_max=12, random_state=0, n_jobs=1)
    parallel = choose_k(x, k_min=2, k_max=12, random_state=0, n_jobs=4)
    assert serial == parallel


def test_choose_k_return_evidence() -> None:
    x, _ = _blobs(n=1200, d=12, k=5)
    ev = choose_k(x, k_min=2, k_max=10, random_state=0, return_evidence=True)
    assert isinstance(ev, ChooseKResult)
    assert ev.k == choose_k(x, k_min=2, k_max=10, random_state=0)  # same k as the bare call
    assert ev.candidates == list(range(2, 11))
    assert len(ev.inertias) == len(ev.candidates)
    assert ev.criterion == "inertia_elbow"
    assert ev.n_samples == 1200  # below max_samples, so every point is used
    assert ev.inertias[0] > ev.inertias[-1]  # inertia falls as k grows


def test_choose_k_rejects_bad_n_jobs() -> None:
    x, _ = _blobs(n=100)
    with pytest.raises(ValueError, match="n_jobs"):
        choose_k(x, n_jobs=0)


def test_assign_clusters_return_distances() -> None:
    x, _ = _blobs(n=600, d=10, k=4)
    res = cluster(x, method="kmeans", n_clusters=4, random_state=0, return_centroids=True)
    labels, distances = assign_clusters(x, res.centroids, return_distances=True)
    assert labels.shape == (600,)
    assert distances.shape == (600,)
    assert (distances >= 0.0).all()
    # labels must match the plain (distance-free) call
    assert np.array_equal(labels, assign_clusters(x, res.centroids))


def test_assign_clusters_max_distance_marks_noise() -> None:
    x, _ = _blobs(n=600, d=10, k=4)
    res = cluster(x, method="kmeans", n_clusters=4, random_state=0, return_centroids=True)
    # points very far from every centroid become -1; in-distribution points keep a real label.
    far = np.full((3, 10), 1e6)
    both = np.concatenate([x[:5], far])
    labels, distances = assign_clusters(
        both, res.centroids, max_distance=100.0, return_distances=True
    )
    assert (labels[:5] >= 0).all()  # the real points keep their cluster
    assert (labels[5:] == -1).all()  # the far points are noise
    assert (distances[5:] > 100.0).all()  # raw distance still reported for the -1 rows


def test_assign_clusters_rejects_bad_max_distance() -> None:
    x, _ = _blobs(n=100, d=8, k=3)
    res = cluster(x, method="kmeans", n_clusters=3, random_state=0, return_centroids=True)
    with pytest.raises(ValueError, match="max_distance"):
        assign_clusters(x, res.centroids, max_distance=-1.0)


def test_cluster_result_dispersion_and_sizes() -> None:
    # a diffuse cluster and a tight one of the same size are told apart by dispersion (not by size
    # or total inertia); sizes sum to n. This is the "should I split this cluster?" signal (#28).
    rng = np.random.default_rng(0)
    tight = rng.normal(0.0, 0.2, (200, 8)) + 10.0
    diffuse = rng.normal(0.0, 3.0, (200, 8)) - 10.0
    x = np.concatenate([tight, diffuse])
    r = cluster(x, method="kmeans", n_clusters=2, random_state=0, return_centroids=True)
    assert r.dispersion.shape == (2,)
    assert r.sizes.shape == (2,)
    assert int(r.sizes.sum()) == 400
    assert (r.dispersion >= 0.0).all()
    assert r.dispersion.max() > r.dispersion.min() * 2.0  # the diffuse cluster is clearly looser


def test_public_surface_is_importable() -> None:
    # every name in __all__ must resolve through the lazy __getattr__ (guards the kmeans export).
    import fastumap

    for name in fastumap.__all__:
        assert getattr(fastumap, name) is not None
    assert fastumap.kmeans is kmeans


def test_wide_input_below_50k_rows_uses_accel(monkeypatch: pytest.MonkeyPatch) -> None:
    # 10k x 768 is 7.7M values, over _ACCEL_MIN_CELLS, so it must take the native kernel even
    # though n < _ACCEL_MIN. Numpy was ~3x slower at 24k x 768 on one core.
    from fastumap import _kmeans

    pytest.importorskip("fastumap._accel")
    calls: list[int] = []
    real = _kmeans._accel_kmeans  # pyright: ignore[reportPrivateUsage]
    assert real is not None

    def spy(data: np.ndarray, k: int, n_iter: int, seed: int) -> np.ndarray:
        calls.append(data.shape[0])
        return real(data, k, n_iter, seed)

    monkeypatch.setattr(_kmeans, "_accel_kmeans", spy)
    x, y = _blobs(n=10_000, d=768, k=5)
    labels = cluster(x, method="kmeans", n_clusters=5, random_state=0)
    assert calls == [10_000]
    assert _purity(y, labels) > 0.95
