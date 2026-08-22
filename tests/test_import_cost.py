"""The whole reason this library exists: `import fastumap` must be cheap.

Measured in a fresh subprocess so nothing is already cached in this interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_IMPORT_BUDGET_S = 0.20

_MEASURE = textwrap.dedent(
    """
    import time, importlib
    start = time.perf_counter()
    importlib.import_module("fastumap")
    print(time.perf_counter() - start)
    """
)


def _time_import() -> float:
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE], capture_output=True, text=True, check=True
    )
    return float(out.stdout.strip().splitlines()[-1])


def test_import_under_budget() -> None:
    _time_import()  # warm the .pyc cache so we measure import, not compilation
    best = min(_time_import() for _ in range(5))
    assert best < _IMPORT_BUDGET_S, f"import fastumap took {best * 1000:.0f}ms (budget 200ms)"


def test_import_does_not_pull_numpy() -> None:
    # The lazy public API means even numpy must not load until a projection runs.
    code = 'import sys; import fastumap; print("numpy" in sys.modules)'
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
