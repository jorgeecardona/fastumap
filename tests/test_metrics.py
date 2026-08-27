"""The grading metrics must separate a real layout from a random control."""

from __future__ import annotations

import pytest

from fastumap import (
    global_distance_correlation,
    neighbor_overlap,
    random_layout,
    umap_project,
)
from fixtures import gaussian_clusters


def test_overlap_beats_random_control() -> None:
    x, _ = gaussian_clusters(n=500, dim=64)
    emb = umap_project(x, 2, random_state=3, n_epochs=200)
    ours = neighbor_overlap(x, emb, 15)
    control = neighbor_overlap(x, random_layout(x.shape[0], 2, random_state=3), 15)
    assert ours > 0.1
    assert ours > 5.0 * control


def test_global_correlation_positive() -> None:
    x, _ = gaussian_clusters(n=400, dim=64)
    emb = umap_project(x, 2, random_state=3, n_epochs=200)
    assert global_distance_correlation(x, emb) > 0.2


def test_identical_layout_full_overlap() -> None:
    x, _ = gaussian_clusters(n=100, dim=16)
    assert neighbor_overlap(x, x[:, :2], 10) >= 0.0  # sanity: no crash, valid range
    assert abs(neighbor_overlap(x, x, 10) - 1.0) < 1e-9  # a copy preserves all neighbours


def test_neighbor_overlap_rejects_k_below_one() -> None:
    # k=0 previously divided by zero; a clear error beats a bare ZeroDivisionError. Regression.
    x, _ = gaussian_clusters(n=50, dim=8)
    with pytest.raises(ValueError, match="k must be"):
        neighbor_overlap(x, x[:, :2], 0)
