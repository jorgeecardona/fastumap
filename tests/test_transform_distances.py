"""transform(return_distances=True) — per-point fit measure (GitLab #14)."""

import numpy as np

from fastumap import fit, transform


def test_coords_unchanged_and_distances_shape() -> None:
    x = np.random.default_rng(0).standard_normal((300, 32))
    model = fit(x, 2, random_state=42)
    new = np.random.default_rng(1).standard_normal((20, 32))
    coords_only = transform(model, new)
    coords, dist = transform(model, new, return_distances=True)
    assert np.array_equal(coords, coords_only)  # the flag never changes the coordinates
    assert dist.shape == (20,)
    assert (dist >= 0.0).all()


def test_distance_flags_outliers() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 16))
    model = fit(x, 2, random_state=42)
    inliers = rng.standard_normal((8, 16))
    outliers = rng.standard_normal((8, 16)) + 20.0  # far from the training distribution
    _, d_in = transform(model, inliers, return_distances=True)
    _, d_out = transform(model, outliers, return_distances=True)
    assert d_out.mean() > d_in.mean()
