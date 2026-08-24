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


def test_init_biases_toward_the_start_frame() -> None:
    # A warm start: seeded from the previous layout, the result keeps the previous frame far
    # better than a fresh spectral start (which is free to rotate/reflect). The optimiser still
    # moves points, so this is a correlation, not a freeze — seeded tracks prev, fresh does not.
    x = np.random.default_rng(1).standard_normal((400, 24))
    prev = umap_project(x, 2, random_state=42)
    seeded = umap_project(x, 2, random_state=99, init=prev)
    fresh = umap_project(x, 2, random_state=99)

    def frame_corr(a: np.ndarray, b: np.ndarray) -> float:
        return abs(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))

    assert frame_corr(seeded, prev) > frame_corr(fresh, prev) + 0.1


def test_init_is_not_mutated_or_aliased() -> None:
    # #15: the in-place SGD (from #5) must not write through to the caller's init array, and
    # the result must be a fresh array — otherwise "before" and "after" alias and a drift
    # comparison silently reads zero movement.
    x = np.random.default_rng(3).standard_normal((300, 24))
    prev = umap_project(x, 2, random_state=42)
    snapshot = prev.copy()
    result = umap_project(x, 2, random_state=42, init=prev)
    assert not np.shares_memory(result, prev)
    assert np.array_equal(prev, snapshot)


def test_init_wrong_shape_raises() -> None:
    x = np.random.default_rng(2).standard_normal((100, 8))
    with pytest.raises(ValueError, match="init must have shape"):
        umap_project(x, 2, init=np.zeros((50, 2)))
    with pytest.raises(ValueError, match="init must have shape"):
        umap_project(x, 2, init=np.zeros((100, 3)))
