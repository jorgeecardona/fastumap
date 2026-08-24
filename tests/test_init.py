"""init= starting embedding (GitLab #12)."""

import numpy as np
import pytest

from fastumap import umap_project


def test_init_runs_and_is_deterministic() -> None:
    x = np.random.default_rng(0).standard_normal((300, 24))
    prev = umap_project(x, 2, random_state=42)
    a = umap_project(x, 2, random_state=7, init=prev)
    b = umap_project(x, 2, random_state=7, init=prev)
    assert a.shape == (300, 2)
    assert np.array_equal(a, b)


def test_init_stays_near_the_given_start() -> None:
    # Seeded from the previous layout, the optimiser refines rather than re-laying-out from a
    # fresh spectral start (which could rotate/reflect). The result should stay closer to the
    # given init than a from-scratch fit does.
    x = np.random.default_rng(1).standard_normal((400, 24))
    prev = umap_project(x, 2, random_state=42)
    seeded = umap_project(x, 2, random_state=99, init=prev)
    fresh = umap_project(x, 2, random_state=99)
    assert np.abs(seeded - prev).mean() < np.abs(fresh - prev).mean()


def test_init_wrong_shape_raises() -> None:
    x = np.random.default_rng(2).standard_normal((100, 8))
    with pytest.raises(ValueError, match="init must have shape"):
        umap_project(x, 2, init=np.zeros((50, 2)))
    with pytest.raises(ValueError, match="init must have shape"):
        umap_project(x, 2, init=np.zeros((100, 3)))
