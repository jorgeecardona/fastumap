"""Guard the thesis: the runtime install stays numpy+scipy, no JIT stack."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib

_HEAVY = ("numba", "llvmlite", "umap", "sklearn", "torch")


def test_importing_fastumap_pulls_no_jit_stack() -> None:
    code = (
        "import sys, fastumap; "
        f"bad=[m for m in {_HEAVY!r} if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"import fastumap loaded: {out.stdout.strip()}"


def test_runtime_dependencies_are_clean() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    deps = " ".join(data["project"]["dependencies"]).lower()
    for bad in ("numba", "llvmlite", "scikit-learn", "sklearn", "umap", "torch"):
        assert bad not in deps, f"{bad} leaked into runtime dependencies"
