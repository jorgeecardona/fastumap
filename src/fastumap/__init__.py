"""fastumap — a numba-free UMAP in pure numpy + scipy.

The public surface is deliberately small: ``umap_project`` and ``spectral_project``
for layouts, and the three grading metrics.

Imports are lazy on purpose. The whole reason this library exists is import cost, so
``import fastumap`` must not drag in numpy or scipy at module load. The names below are
bound on first access via ``__getattr__``; the ``TYPE_CHECKING`` block gives type
checkers and IDEs the real signatures without triggering those imports at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "global_distance_correlation",
    "neighbor_overlap",
    "random_layout",
    "spectral_project",
    "umap_project",
]

if TYPE_CHECKING:
    from .metrics import global_distance_correlation, neighbor_overlap, random_layout
    from .projection import spectral_project, umap_project


def __getattr__(name: str) -> object:
    if name in ("umap_project", "spectral_project"):
        from . import projection

        return getattr(projection, name)
    if name in ("neighbor_overlap", "global_distance_correlation", "random_layout"):
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
