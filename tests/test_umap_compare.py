"""Compare against umap-learn as a test-only reference.

Skipped (not failed) when umap-learn is absent, so a clean runtime install still runs
the rest of the suite. We assert we are within a tolerance band of the reference, never
coordinate equality: different init and update order rotate/reflect equally-good layouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import gaussian_clusters

umap = pytest.importorskip("umap")

from fastumap import neighbor_overlap, umap_project  # noqa: E402


@pytest.mark.parametrize("n", [1000])
def test_overlap_within_tolerance_of_umap_learn(n: int) -> None:
    x, _ = gaussian_clusters(n=n, dim=64, seed=0)
    ours = umap_project(x, 2, random_state=42)
    reference = np.asarray(
        umap.UMAP(n_components=2, random_state=42).fit_transform(x), dtype=np.float64
    )
    ov_ours = neighbor_overlap(x, ours, 15)
    ov_ref = neighbor_overlap(x, reference, 15)
    # We should be in the same ballpark, not necessarily ahead yet (roadmap #5/#6).
    assert ov_ours > ov_ref - 0.05
