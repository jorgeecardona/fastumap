"""PCA pre-reduction (`pca_dim`): a speed lever for high-dimensional inputs.

Correctness only here (deterministic, valid, quality-preserving on low-rank data); the
speed win at 5000 x 1024 is a benchmark, not a unit test.
"""

from __future__ import annotations

import numpy as np

from fastumap import fit, neighbor_overlap, transform, umap_project
from fixtures import gaussian_clusters


def test_pca_dim_none_is_the_default() -> None:
    x, _ = gaussian_clusters(n=300, dim=64, seed=0)
    default = umap_project(x, 2, random_state=0, n_epochs=100)
    explicit = umap_project(x, 2, random_state=0, n_epochs=100, pca_dim=None)
    assert np.array_equal(default, explicit)


def test_pca_dim_above_n_features_is_a_noop() -> None:
    # pca_dim >= n_features means no reduction, so the result is the plain path byte-for-byte.
    x, _ = gaussian_clusters(n=300, dim=64, seed=0)
    plain = umap_project(x, 2, random_state=0, n_epochs=100)
    noop = umap_project(x, 2, random_state=0, n_epochs=100, pca_dim=200)
    assert np.array_equal(plain, noop)


def test_pca_reduced_is_deterministic_and_finite() -> None:
    x, _ = gaussian_clusters(n=400, dim=128, seed=1)
    first = umap_project(x, 2, random_state=0, n_epochs=100, pca_dim=32)
    second = umap_project(x, 2, random_state=0, n_epochs=100, pca_dim=32)
    assert np.array_equal(first, second)
    assert first.shape == (400, 2)
    assert bool(np.all(np.isfinite(first)))


def test_pca_preserves_overlap_on_low_rank_data() -> None:
    # 8 clusters in 128 dims => low intrinsic rank; reducing to 32 keeps the structure.
    x, _ = gaussian_clusters(n=400, dim=128, seed=2)
    # Mean over 3 seeds: one seed's gap swings between -0.03 and +0.02 with float rounding
    # (CI on Python 3.14 failed at -0.032 on seed 0 alone).
    seeds = range(3)
    full = np.mean(
        [neighbor_overlap(x, umap_project(x, 2, random_state=s, n_epochs=150), 15) for s in seeds]
    )
    reduced = np.mean(
        [
            neighbor_overlap(x, umap_project(x, 2, random_state=s, n_epochs=150, pca_dim=32), 15)
            for s in seeds
        ]
    )
    assert reduced > full - 0.03


def test_transform_with_pca_is_deterministic() -> None:
    x, _ = gaussian_clusters(n=400, dim=128, seed=3)
    model = fit(x[:350], 2, random_state=0, n_epochs=100, pca_dim=32)
    first = transform(model, x[350:])
    second = transform(model, x[350:])
    assert np.array_equal(first, second)
    assert first.shape == (50, 2)
    assert bool(np.all(np.isfinite(first)))
