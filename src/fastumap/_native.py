"""Load the compiled ``fastumap._accel`` extension, or fail with how to get it."""

from __future__ import annotations

import importlib
from types import ModuleType


def load_accel() -> ModuleType:
    """Return the ``fastumap._accel`` module.

    Every wheel and every sdist install compiles it, so it is missing only in an unbuilt source
    checkout. That case raises instead of running the numpy code paths without telling anyone.

    Raises:
        ImportError: the extension was never built.
    """
    try:
        return importlib.import_module("fastumap._accel")
    except ImportError as exc:
        raise ImportError(
            "fastumap's compiled kernel (fastumap._accel) is missing. Install a wheel with "
            "`pip install fastumap`, or build it in a source checkout with `uv sync`."
        ) from exc
