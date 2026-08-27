"""transform(): place new points into a fitted layout without refitting.

Deterministic, leaves the base layout untouched, and lands new points near training
points of their own cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastumap import fit, transform, umap_project
from fastumap.neighbors import knn_between
from fixtures import gaussian_clusters


def test_fit_embedding_matches_umap_project() -> None:
    x, _ = gaussian_clusters(n=300, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=100)
    direct = umap_project(x, 2, random_state=0, n_epochs=100)
    assert np.array_equal(model.embedding, direct)


def test_transform_is_deterministic() -> None:
    x, _ = gaussian_clusters(n=500, seed=0)
    model = fit(x[:400], 2, random_state=0, n_epochs=150)
    first = transform(model, x[400:])
    second = transform(model, x[400:])
    assert np.array_equal(first, second)
    assert first.shape == (100, 2)
    assert bool(np.all(np.isfinite(first)))


def test_transform_does_not_mutate_base_layout() -> None:
    x, _ = gaussian_clusters(n=300, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=100)
    before = model.embedding.copy()
    transform(model, x[:20])
    assert np.array_equal(model.embedding, before)


def test_transform_places_points_in_correct_cluster() -> None:
    x, labels = gaussian_clusters(n=600, seed=1)
    n_train = 500
    model = fit(x[:n_train], 2, random_state=0, n_epochs=200)
    projected = transform(model, x[n_train:])
    # nearest training point in the embedding should share the new point's true cluster
    nearest, _ = knn_between(projected, model.embedding, 1)
    predicted = labels[:n_train][nearest[:, 0]]
    accuracy = float(np.mean(predicted == labels[n_train:]))
    assert accuracy > 0.8, f"only {accuracy:.2f} of new points landed in their cluster"


def test_transform_feature_mismatch_rejected() -> None:
    x, _ = gaussian_clusters(n=200, dim=32, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=50)
    wrong = np.random.default_rng(0).normal(size=(10, 16))
    with pytest.raises(ValueError, match="features"):
        transform(model, wrong)


def test_transform_single_point() -> None:
    # placing one new point into a fitted layout is well-defined (one query per request is a
    # primary serving path); only fit needs the n>=2 floor. Regression for the n=1 crash.
    x, _ = gaussian_clusters(n=200, dim=12, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=50)
    y = transform(model, x[:1])
    assert y.shape == (1, 2)
    assert np.isfinite(y).all()
