"""densMAP — density-preserving UMAP variant (roadmap Tier 2)."""

from __future__ import annotations

import numpy as np

from fastumap import umap_project
from fastumap._types import FloatArray
from fastumap.neighbors import knn_graph


def _varying_density_data() -> FloatArray:
    # one dense blob + one sparse blob far away: plain UMAP flattens the density gap, densMAP keeps it.
    rng = np.random.default_rng(0)
    tight = rng.standard_normal((300, 20)) * 0.5
    loose = rng.standard_normal((300, 20)) * 5.0 + np.array([30.0] + [0.0] * 19)
    return np.ascontiguousarray(np.vstack([tight, loose]), dtype=np.float64)


def _density_corr(orig: FloatArray, emb: FloatArray) -> float:
    _, do = knn_graph(orig, 15, mode="exact")
    _, de = knn_graph(emb, 15, mode="exact")
    ro = np.log(do[:, 1:].mean(axis=1) + 1e-8)
    re = np.log(de[:, 1:].mean(axis=1) + 1e-8)
    return float(np.corrcoef(ro, re)[0, 1])


def test_densmap_preserves_density_far_better_than_plain() -> None:
    x = _varying_density_data()
    plain = umap_project(x, 2, random_state=0, densmap=False)
    dens = umap_project(x, 2, random_state=0, densmap=True)
    assert bool(np.all(np.isfinite(dens)))
    plain_corr = _density_corr(x, plain)
    dens_corr = _density_corr(x, dens)
    assert dens_corr > plain_corr + 0.3  # densMAP keeps dense/sparse distinct
    assert dens_corr > 0.6


def test_densmap_output_shape_and_determinism() -> None:
    x = _varying_density_data()
    a = umap_project(x, 2, random_state=0, densmap=True)
    b = umap_project(x, 2, random_state=0, densmap=True)
    assert a.shape == (600, 2)
    assert np.array_equal(a, b)  # numpy densMAP path is deterministic
