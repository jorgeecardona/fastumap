"""Approximate kNN backend (fastumap[ann] -> faiss), roadmap #15."""

import numpy as np
import pytest

pytest.importorskip("faiss")  # skip the whole module if the ann extra is not installed

from fastumap import ann_available, umap_project
from fastumap.metrics import neighbor_overlap
from fastumap.neighbors import knn, knn_approx, knn_graph


def test_ann_available_true_when_faiss_present() -> None:
    assert ann_available() is True


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_high_recall_vs_exact(metric: str) -> None:
    x = np.random.default_rng(0).standard_normal((3000, 48))
    exact, _ = knn(x, 15, metric=metric)
    approx, _ = knn_approx(x, 15, metric=metric)
    recall = np.mean([len(set(exact[i]) & set(approx[i])) / 15 for i in range(len(x))])
    assert recall > 0.9, f"{metric} recall too low: {recall}"


def test_approx_is_deterministic() -> None:
    x = np.random.default_rng(1).standard_normal((3000, 48))
    idx_a, dist_a = knn_approx(x, 15)
    idx_b, dist_b = knn_approx(x, 15)
    assert np.array_equal(idx_a, idx_b)
    assert np.array_equal(dist_a, dist_b)


def test_distance_units_match_exact_contract() -> None:
    x = np.random.default_rng(2).standard_normal((1000, 32))
    _, de = knn_approx(x, 5, metric="euclidean")
    assert (de >= 0.0).all()
    assert (np.diff(de, axis=1) >= -1e-9).all()  # each row sorted ascending
    assert de.min() < 1e-6  # self (distance 0) is present
    _, dc = knn_approx(x, 5, metric="cosine")
    assert (dc >= 0.0).all() and dc.max() <= 2.0  # cosine distance in [0, 2]


def test_auto_stays_exact_below_threshold() -> None:
    # n=200 < 4096, so "auto" must use the exact path — bit-identical to brute force.
    x = np.random.default_rng(3).standard_normal((200, 16))
    exact_idx, exact_d = knn(x, 10)
    auto_idx, auto_d = knn_graph(x, 10, mode="auto")
    assert np.array_equal(exact_idx, auto_idx)
    assert np.array_equal(exact_d, auto_d)


def test_exact_mode_equals_brute_force() -> None:
    x = np.random.default_rng(4).standard_normal((300, 16))
    a, da = knn_graph(x, 10, mode="exact")
    b, db = knn(x, 10)
    assert np.array_equal(a, b) and np.array_equal(da, db)


def test_bad_mode_raises() -> None:
    x = np.random.default_rng(5).standard_normal((50, 8))
    with pytest.raises(ValueError, match="knn mode"):
        knn_graph(x, 5, mode="nonsense")


def test_approx_layout_overlap_close_to_exact() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, 48)) * 10.0
    labels = rng.integers(0, 8, size=1200)
    x = (centers[labels] + rng.standard_normal((1200, 48))).astype(np.float64)
    exact = umap_project(x, 2, knn="exact", random_state=42)
    approx = umap_project(x, 2, knn="approx", random_state=42)
    o_exact = neighbor_overlap(x, exact, 15)
    o_approx = neighbor_overlap(x, approx, 15)
    assert o_approx > o_exact - 0.05, f"approx {o_approx} vs exact {o_exact}"
