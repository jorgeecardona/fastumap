"""fastumap: fast, import-light UMAP and clustering.

The public surface is small on purpose. ``umap_project`` / ``spectral_project`` build layouts.
``cluster`` / ``assign_clusters`` do clustering. There are also the grading metrics.

Imports are lazy on purpose too. This library exists to keep import cost low. So
``import fastumap`` must not drag in numpy or scipy (or the native kernel) at module load. The
names below are bound on first access via ``__getattr__``. The ``TYPE_CHECKING`` block gives
type checkers and IDEs the real signatures, without triggering those imports at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ChooseKResult",
    "ClusterResult",
    "UMAPModel",
    "__version__",
    "accelerator_active",
    "ann_available",
    "assign_clusters",
    "choose_k",
    "cluster",
    "dbscan",
    "fit",
    "global_distance_correlation",
    "hdbscan",
    "kmeans",
    "neighbor_overlap",
    "random_layout",
    "spectral_project",
    "transform",
    "umap_project",
]

if TYPE_CHECKING:
    # pyproject is the single source of truth for the version (like musil). __version__ is
    # read from the installed distribution metadata on first access, so it can never drift
    # from what was published. Declared here for type checkers.
    __version__: str

    from .clustering import (
        ChooseKResult,
        ClusterResult,
        assign_clusters,
        choose_k,
        cluster,
        dbscan,
        hdbscan,
        kmeans,
    )
    from .layout import accelerator_active
    from .metrics import global_distance_correlation, neighbor_overlap, random_layout
    from .neighbors import ann_available
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
    if name == "ann_available":
        from . import neighbors

        return neighbors.ann_available
    if name in (
        "cluster",
        "ChooseKResult",
        "ClusterResult",
        "assign_clusters",
        "choose_k",
        "dbscan",
        "hdbscan",
        "kmeans",
    ):
        from . import clustering

        return getattr(clustering, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
