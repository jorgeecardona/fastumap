"""Machine-independent speedup guard (roadmap: quality).

The native SGD kernel must be at least as fast as the numpy fallback for the SAME algorithm. Both
are timed in-process on the same input and compared as a ratio, so the host speed cancels — no
absolute wall-clock budget that would flake on a slow or loaded runner. Skipped when the native
kernel is absent (an unbuilt source checkout).
"""

from __future__ import annotations

import time

import pytest

import fastumap.layout as layout
from fastumap import umap_project
from fastumap._types import FloatArray
from fixtures import gaussian_clusters

pytest.importorskip("fastumap._accel")


def _best_project_seconds(x: FloatArray, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        umap_project(x, 2, random_state=0)
        best = min(best, time.perf_counter() - start)
    return best


def test_native_sgd_is_not_slower_than_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same input, same algorithm (chunk_count=1); only the kernel differs. Forcing _accel_optimize
    # to None routes optimize_layout through the numpy fallback. The native path does umap-learn's
    # in-place walk in Rust and is ~1.7-1.9x faster, so requiring it to be no slower than numpy
    # (with 25% slack for timing noise, best-of-3) is a robust, host-independent check.
    x, _ = gaussian_clusters(n=600, dim=64, seed=0)
    monkeypatch.setattr(layout, "_accel_optimize", None)  # force the numpy fallback
    numpy_seconds = _best_project_seconds(x)
    monkeypatch.undo()  # restore the native kernel
    assert getattr(layout, "_accel_optimize") is not None  # sanity: the accel is back  # noqa: B009
    native_seconds = _best_project_seconds(x)
    assert native_seconds <= numpy_seconds * 1.25, (
        f"native SGD {native_seconds:.3f}s was slower than numpy {numpy_seconds:.3f}s"
    )
