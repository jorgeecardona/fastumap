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

__all__ = [
    "UMAPModel",
    "__version__",
    "accelerator_active",
    "fit",
    "global_distance_correlation",
    "neighbor_overlap",
    "random_layout",
    "spectral_project",
    "transform",
    "umap_project",
]

if TYPE_CHECKING:
    # pyproject is the single source of truth for the version (like musil); __version__ is
    # derived from the installed distribution metadata at first access, so it can never
    # drift from what was published. Declared here for type checkers.
    __version__: str

    from .layout import accelerator_active
    from .metrics import global_distance_correlation, neighbor_overlap, random_layout
    from .projection import UMAPModel, fit, spectral_project, transform, umap_project


def __getattr__(name: str) -> object:
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("fastumap")
        except PackageNotFoundError:  # not installed as a distribution (rare, e.g. raw src)
            return "0+unknown"
    if name in ("umap_project", "spectral_project", "fit", "transform", "UMAPModel"):
        from . import projection

        return getattr(projection, name)
    if name in ("neighbor_overlap", "global_distance_correlation", "random_layout"):
        from . import metrics

        return getattr(metrics, name)
    if name == "accelerator_active":
        from . import layout

        return layout.accelerator_active
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
