"""Shape, finiteness, determinism, and degenerate-input behaviour of the projection."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from fastumap import spectral_project, umap_project
from fastumap._types import FloatArray
from fixtures import (
    barely_above_minimum,
    collinear,
    curved_manifold,
    disconnected_blobs,
    gaussian_clusters,
    identical,
)


def test_shape_2d() -> None:
    x, _ = gaussian_clusters(n=300)
    emb = umap_project(x, 2, random_state=1, n_epochs=100)
    assert emb.shape == (300, 2)
    assert bool(np.all(np.isfinite(emb)))


def test_shape_3d() -> None:
    x, _ = gaussian_clusters(n=200)
    emb = umap_project(x, 3, random_state=1, n_epochs=100)
    assert emb.shape == (200, 3)
    assert bool(np.all(np.isfinite(emb)))


def test_deterministic_same_process() -> None:
    x, _ = gaussian_clusters(n=200)
    first = umap_project(x, 2, random_state=7, n_epochs=100)
    second = umap_project(x, 2, random_state=7, n_epochs=100)
    assert np.array_equal(first, second)


def test_different_seed_differs() -> None:
    x, _ = gaussian_clusters(n=200)
    first = umap_project(x, 2, random_state=1, n_epochs=100)
    second = umap_project(x, 2, random_state=2, n_epochs=100)
    assert not np.array_equal(first, second)


def test_spectral_project_shape() -> None:
    x, _ = gaussian_clusters(n=200)
    emb = spectral_project(x, 2, random_state=1)
    assert emb.shape == (200, 2)
    assert bool(np.all(np.isfinite(emb)))


@pytest.mark.parametrize("factory", [collinear, identical, curved_manifold])
def test_degenerate_inputs_stay_finite(factory: Callable[[], FloatArray]) -> None:
    x = factory()
    emb = umap_project(x, 2, random_state=1, n_epochs=50)
    assert emb.shape == (x.shape[0], 2)
    assert bool(np.all(np.isfinite(emb)))


def test_disconnected_graph_does_not_collapse() -> None:
    x = disconnected_blobs()
    emb = umap_project(x, 2, random_state=1, n_epochs=50)
    assert bool(np.all(np.isfinite(emb)))
    assert float(np.ptp(emb)) > 0.0


def test_barely_above_minimum() -> None:
    x = barely_above_minimum()
    emb = umap_project(x, 2, random_state=1, n_epochs=50)
    assert emb.shape == (5, 2)
    assert bool(np.all(np.isfinite(emb)))
