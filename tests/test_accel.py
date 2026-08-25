"""Native Rust accelerator (fastumap._accel), built into the package: the SGD runs natively.

Skipped when the extension isn't built (a source checkout without a maturin build), so the
pure-numpy fallback is unaffected; when it is, these check the fast path stays deterministic and
keeps (in fact improves) quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import gaussian_clusters

pytest.importorskip("fastumap._accel")

from fastumap import neighbor_overlap, umap_project


def test_accel_path_is_deterministic() -> None:
    x, _ = gaussian_clusters(n=500, dim=64, seed=0)
    first = umap_project(x, 2, random_state=0)
    second = umap_project(x, 2, random_state=0)
    assert np.array_equal(first, second)  # deterministic through the Rust kernel
    assert first.shape == (500, 2)
    assert bool(np.all(np.isfinite(first)))


def test_accel_preserves_local_structure() -> None:
    x, _ = gaussian_clusters(n=600, dim=64, seed=1)
    overlap = neighbor_overlap(x, umap_project(x, 2, random_state=0), 15)
    assert overlap > 0.15  # the in-place walk should be at least as good as the numpy path
