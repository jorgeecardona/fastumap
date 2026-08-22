"""Determinism across processes: same input + seed => bit-identical output bytes."""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
