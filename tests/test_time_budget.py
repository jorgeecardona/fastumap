"""Wall-clock guard, measured under a single-core cap so it means something on 0.5 vCPU.

These ceilings are deliberately generous — they exist to catch a *regression* (a change
that makes a fit several times slower), not to certify absolute performance. Measured
single-thread on a laptop: n=1000 x 1024 ~7.7 s, n=5000 x 1024 ~39.5 s. The SGD scatter
(np.add.at) dominates; see the perf issue for the optimisation that would tighten these.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# seconds, single-thread, dim=1024. ~3x the measured time, to absorb CI variance.
_CEILINGS: dict[int, float] = {1000: 25.0, 5000: 120.0}

_MEASURE = textwrap.dedent(
    """
    import time
    import numpy as np
    from fastumap import umap_project

    x = np.random.default_rng(0).normal(size=({n}, 1024))
    start = time.perf_counter()
    umap_project(x, 2, random_state=0)
    print(time.perf_counter() - start)
    """
)

_SINGLE_CORE_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@pytest.mark.parametrize(("n", "ceiling"), list(_CEILINGS.items()))
def test_wall_clock_under_budget(n: int, ceiling: float) -> None:
    env = {**os.environ, **_SINGLE_CORE_ENV}
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE.format(n=n)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    elapsed = float(out.stdout.strip().splitlines()[-1])
    assert elapsed < ceiling, f"n={n} x1024 fit took {elapsed:.1f}s (budget {ceiling}s)"
