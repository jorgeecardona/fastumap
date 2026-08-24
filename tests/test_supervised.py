"""Categorical supervised projection (GitLab #13; ported from umap-learn 0.5.12)."""

import numpy as np
import pytest

from fastumap import umap_project
from fastumap.graph import categorical_intersection, fuzzy_simplicial_set
from fastumap.neighbors import knn


def _labelled(n: int = 800, d: int = 32, classes: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, d)) * 1.5
    y = rng.integers(0, classes, n)
    x = (centers[y] + rng.standard_normal((n, d))).astype(np.float64)
    return x, y


def _within_between_ratio(xy: np.ndarray, y: np.ndarray) -> float:
    from scipy.spatial.distance import cdist

    within, between = [], []
    for c in np.unique(y):
        same, other = xy[y == c][:40], xy[y != c][:40]
        within.append(cdist(same, same).mean())
        between.append(cdist(same, other).mean())
    return float(np.mean(within) / np.mean(between))


def test_y_none_is_plain_unsupervised() -> None:
    x, _ = _labelled()
    assert np.array_equal(
        umap_project(x, 2, random_state=42),
        umap_project(x, 2, random_state=42, y=None),
    )


def test_supervised_tightens_classes() -> None:
    x, y = _labelled()
    unsup = _within_between_ratio(umap_project(x, 2, random_state=42), y)
    sup = _within_between_ratio(umap_project(x, 2, random_state=42, y=y), y)
    assert sup < unsup  # labels pull same-class points together


def test_supervised_is_deterministic() -> None:
    x, y = _labelled(n=500)
    a = umap_project(x, 2, random_state=7, y=y)
    b = umap_project(x, 2, random_state=7, y=y)
    assert np.array_equal(a, b)


def test_higher_target_weight_severs_more_cross_edges() -> None:
    x, y = _labelled(n=400, classes=2)
    ki, kd = knn(x, 10)
    graph = fuzzy_simplicial_set(ki, kd)

    def cross_weight(g: object) -> float:
        mask = y[g.head] != y[g.tail]  # type: ignore[attr-defined]
        return float(g.weight[mask].sum())  # type: ignore[attr-defined]

    mild = categorical_intersection(graph, y, target_weight=0.5)  # far_dist = 5
    harsh = categorical_intersection(graph, y, target_weight=0.9)  # far_dist = 25
    assert cross_weight(harsh) <= cross_weight(mild) + 1e-9


def test_unknown_label_is_semi_supervised() -> None:
    x, y = _labelled(n=400)
    y = y.copy()
    y[:100] = -1  # unlabelled: softly down-weighted, not cut
    out = umap_project(x, 2, random_state=42, y=y)
    assert out.shape == (400, 2)


def test_wrong_label_length_raises() -> None:
    x, _ = _labelled(n=100)
    with pytest.raises(ValueError, match="one label per point"):
        umap_project(x, 2, y=np.zeros(50, dtype=np.int64))
