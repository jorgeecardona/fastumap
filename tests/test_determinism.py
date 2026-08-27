"""Determinism across processes: same input + seed => bit-identical output bytes."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_SNIPPET = textwrap.dedent(
    """
    import sys
    import numpy as np
    from fastumap import umap_project

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 32))
    emb = umap_project(x, 2, random_state=123, n_epochs=100)
    sys.stdout.buffer.write(emb.tobytes())
    """
)


def _run() -> bytes:
    out = subprocess.run([sys.executable, "-c", _SNIPPET], capture_output=True, check=True)
    return out.stdout


def test_two_processes_identical_bytes() -> None:
    assert _run() == _run()


_KMEANS_SNIPPET = textwrap.dedent(
    """
    import sys
    import numpy as np
    from fastumap.clustering import kmeans

    # uniform data: many points sit near cluster boundaries, so a low-order-bit change in a
    # centroid can flip a label. n >= 50k routes through the native full-Lloyd kernel.
    rng = np.random.default_rng(3)
    x = rng.random((55000, 16))
    labels, centroids = kmeans(x, 12, n_iter=40, random_state=0)
    sys.stdout.buffer.write(labels.astype(np.int64).tobytes())
    sys.stdout.buffer.write(np.ascontiguousarray(centroids, dtype=np.float32).tobytes())
    """
)


def _run_kmeans(threads: str) -> bytes:
    env = {**os.environ, "RAYON_NUM_THREADS": threads}
    out = subprocess.run(
        [sys.executable, "-c", _KMEANS_SNIPPET], capture_output=True, check=True, env=env
    )
    return out.stdout


def test_accel_kmeans_bit_identical_across_thread_counts() -> None:
    # The native full-Lloyd path must give bit-identical labels AND centroids whatever the core
    # count. Its parallel reductions (k-means++ seed selection and the centroid means) are
    # order-sensitive, so before the fixed-order f64 fix a machine with a different core count
    # picked a different seed and diverged the whole run.
    pytest.importorskip("fastumap._accel")
    assert _run_kmeans("1") == _run_kmeans("4")
