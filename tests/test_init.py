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
    # A gentle warm start holds the previous frame: with init= and a low initial_alpha the SGD
    # barely moves, so the layout stays where init put it. init sets where the optimiser STARTS,
    # not where it ends (GitLab #16), so a full-alpha warm start drifts away over the schedule.
    # Asserted as absolute closeness (robust across platforms), plus that the gentle touch holds
    # the frame far better than the full-alpha one — a fixed margin over a fresh start is too seed-
    # and float-sensitive to hold on every machine (it was, and failed CI on Linux).
    x = np.random.default_rng(1).standard_normal((400, 24))
    prev = umap_project(x, 2, random_state=42)
    gentle = umap_project(x, 2, random_state=99, init=prev, initial_alpha=0.01)
    full = umap_project(x, 2, random_state=99, init=prev, initial_alpha=1.0)

    def frame_corr(a: np.ndarray, b: np.ndarray) -> float:
        return abs(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))

    assert frame_corr(gentle, prev) > 0.8  # the gentle warm start stays on the previous frame
    assert frame_corr(gentle, prev) > frame_corr(full, prev)  # the light touch is why it holds


def test_initial_alpha_reduces_drift_from_init() -> None:
    # The principled version of the same thing (GitLab #16 option 2): a lower initial_alpha makes
    # gentle corrections across the full schedule, so a warm start stays closer to its init than
    # one run at the default full learning rate.
    x = np.random.default_rng(1).standard_normal((400, 24))
    prev = umap_project(x, 2, random_state=42)

    def move(emb: np.ndarray) -> float:
        return float(np.median(np.linalg.norm(emb - prev, axis=1)))

    gentle = umap_project(x, 2, random_state=99, init=prev, initial_alpha=0.05)
    full = umap_project(x, 2, random_state=99, init=prev, initial_alpha=1.0)
    assert move(gentle) < move(full)


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
