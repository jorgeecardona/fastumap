"""Peak-RSS ceiling: the worst case (5000 points x 1024 dims) must stay bounded.

The n-by-n distance matrix is never materialised (512-row blocked kNN), so the fit's
peak memory over baseline stays small. Measured on Linux ~102 MB over the import+input
baseline; the bound below leaves headroom for CI variance. This is a Linux/Fargate claim
(x86 manylinux OpenBLAS), so the test is skipped elsewhere.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Documented bound (see README "Guarantees"). Measured ~102 MB; 200 MB leaves headroom.
_PEAK_RSS_BUDGET_MB = 200

_MEASURE = textwrap.dedent(
    """
    import resource
    import numpy as np
    from fastumap import umap_project

    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 1024))
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    umap_project(x, 2, random_state=0)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print((peak_kb - baseline_kb) / 1024.0)  # MB the fit added over baseline
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="peak-RSS bound is a Linux/Fargate claim")
def test_peak_rss_at_5000x1024_under_budget() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE], capture_output=True, text=True, check=True
    )
    added_mb = float(out.stdout.strip().splitlines()[-1])
    assert added_mb < _PEAK_RSS_BUDGET_MB, (
        f"5000x1024 fit added {added_mb:.0f} MB peak RSS (budget {_PEAK_RSS_BUDGET_MB} MB)"
    )
