"""Initialisation and optimisation of the low-dimensional embedding.

Two steps:

- spectral_layout gives the deterministic starting positions. These are the leading
  eigenvectors of the normalised fuzzy graph, guarded against disconnected components.
- optimize_layout runs the attract/repel SGD. UMAP's cross-entropy objective reduces
  to that SGD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _scipy
from ._native import load_accel
from ._types import FloatArray, IntArray
from .graph import FuzzyGraph, make_epochs_per_sample


@dataclass(frozen=True)
class _DensityStats:
    """Per-epoch densMAP quantities, held fixed while points move during the epoch."""

    R: FloatArray  # z-scored log original-space radius per vertex (fixed for the whole run)
    x: FloatArray  # log embedding radius per vertex this epoch (umap's dens_re_sum)
    phi_sum: FloatArray  # sum of the low-D kernel phi at each vertex this epoch
    re_std: float  # sqrt(var(x) + dens_var_shift)
    re_mean: float  # mean(x)
    re_cov: float  # dot(x, R) / (n - 1)
    mu_tot: float  # total membership mass, sum(graph.weight)


# Native accelerator, the compiled `fastumap._accel` submodule: a Rust reimplementation of the SGD
# that does umap-learn's true in-place walk. A missing extension raises at import.
_accel_optimize = load_accel().optimize_layout


def accelerator_active() -> bool:
    """Return ``True``: the native ``fastumap._accel`` kernel is required.

    A missing extension makes the import raise ``ImportError``, so a running fastumap always has
    the kernel. Kept so existing callers keep working. Note that ``chunk_count > 1`` still runs
    the numpy layout path, which gives a different layout for the same seed.
    """
    return True


_GRID_SPACING = 3.0
_SCALE = 10.0
_INIT_NOISE = 1e-4
_GRAD_CLIP = 4.0


def _scatter_add(target: FloatArray, idx: IntArray, values: FloatArray) -> None:
    """In-place ``target[idx] += values``, accumulating duplicate indices.

    ``np.add.at`` is a slow element-by-element scatter. This replaces it with one compiled
    ``np.bincount`` reduction per output dimension. The embedding is only 2-3 dims, so this
    is a handful of bincounts. It dominates the SGD's speed. Deterministic.
    """
    n, dim = target.shape
    for d in range(dim):
        target[:, d] += np.bincount(idx, weights=values[:, d], minlength=n)


def _repel_grad(
    self_pos: FloatArray,
    neg_pos: FloatArray,
    *,
    a: float,
    b: float,
    gamma: float,
    alpha: float,
) -> FloatArray:
    """Repulsion gradient of one negative-sample draw: push ``self_pos`` away from ``neg_pos``.

    Shared by optimize_layout and optimize_transform, which drew negatives from different arrays
    (the moving embedding vs the frozen training layout) but computed this gradient identically.
    A zero-distance pair (coincident points) gets the ``_GRAD_CLIP`` default push, matching umap's
    handling. The caller applies any self-repulsion mask, since only optimize_layout can sample a
    point against itself.
    """
    diff_n = self_pos - neg_pos
    d2n: FloatArray = np.einsum("ij,ij->i", diff_n, diff_n)
    repel_coeff = np.zeros_like(d2n)
    positive = d2n > 0.0
    d2p = d2n[positive]
    repel_coeff[positive] = (2.0 * gamma * b) / ((0.001 + d2p) * (1.0 + a * np.power(d2p, b)))
    return (
        np.where(
            positive[:, None],
            np.clip(repel_coeff[:, None] * diff_n, -_GRAD_CLIP, _GRAD_CLIP),
            _GRAD_CLIP,
        )
        * alpha
    )


def spectral_layout(graph: FuzzyGraph, dim: int, *, rng: np.random.Generator) -> FloatArray:
    """Spectral initialisation, rescaled to max|coord|=10 with tiny Gaussian noise."""
    n_components, labels = _scipy.connected_components(graph.n, graph.head, graph.tail)
    if n_components == 1:
        emb = _spectral_component(graph.head, graph.tail, graph.weight, graph.n, dim)
    else:
        emb = _multi_component_layout(graph, dim, labels, n_components, rng)
    return _noisy_scale(emb, rng)


def _spectral_component(
    head: IntArray, tail: IntArray, weight: FloatArray, n: int, dim: int
) -> FloatArray:
    """Leading ``dim`` non-trivial eigenvectors of one connected component.

    A component may be too small for ARPACK to extract ``dim+1`` eigenpairs. Then fall back
    to a deterministic spread along the axes, so nothing collapses to a point.
    """
    k = dim + 1
    if n <= k + 1:
        base = np.zeros((n, dim), dtype=np.float64)
        for i in range(n):
            base[i, i % dim] = float(i)
        return base
    v0 = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
    vals, vecs = _scipy.normalized_adjacency_eigenvectors(n, head, tail, weight, k, v0)
    order: IntArray = np.argsort(vals)[::-1]
    ordered: FloatArray = vecs[:, order]
    # skip the trivial leading eigenvector (constant / D^1/2), take the next `dim`.
    return np.ascontiguousarray(ordered[:, 1 : dim + 1], dtype=np.float64)


def _multi_component_layout(
    graph: FuzzyGraph,
    dim: int,
    labels: IntArray,
    n_components: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Lay out each connected component on its own, then offset them on a grid.

    Why: for a disconnected graph the leading eigenvectors are just component indicators.
    Without this step, each component collapses to a single point.
    """
    n = graph.n
    emb = np.zeros((n, dim), dtype=np.float64)
    for c in range(n_components):
        in_comp: np.ndarray[tuple[int], np.dtype[np.bool_]] = labels == c
        idx: IntArray = np.where(in_comp)[0]
        m = int(idx.shape[0])
        remap = np.full(n, -1, dtype=np.int64)
        remap[idx] = np.arange(m, dtype=np.int64)
        edge_mask = in_comp[graph.head]
        sub_head = remap[graph.head[edge_mask]]
        sub_tail = remap[graph.tail[edge_mask]]
        sub_weight = graph.weight[edge_mask]
        if m > dim + 2:
            comp = _spectral_component(sub_head, sub_tail, sub_weight, m, dim)
        else:
            comp = rng.normal(scale=1.0, size=(m, dim))
        max_abs = float(np.max(np.abs(comp))) if m > 0 else 0.0
        if max_abs > 0.0:
            comp = comp / max_abs
        # grid offset so components sit apart in the plane
        side = int(np.ceil(np.sqrt(n_components)))
        offset = np.zeros(dim, dtype=np.float64)
        offset[0] = float(c % side) * _GRID_SPACING
        if dim > 1:
            offset[1] = float(c // side) * _GRID_SPACING
        emb[idx] = comp + offset
    return emb


def _noisy_scale(emb: FloatArray, rng: np.random.Generator) -> FloatArray:
    max_abs = float(np.max(np.abs(emb)))
    scaled = emb * (_SCALE / max_abs) if max_abs > 0.0 else emb
    noisy = scaled + rng.normal(loc=0.0, scale=_INIT_NOISE, size=scaled.shape)
    return np.ascontiguousarray(noisy, dtype=np.float64)


def optimize_layout(
    embedding: FloatArray,
    graph: FuzzyGraph,
    n_epochs: int,
    *,
    a: float,
    b: float,
    rng: np.random.Generator,
    negative_sample_rate: int = 5,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    chunk_count: int = 1,
    dens_lambda: float = 0.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
    dens_R: FloatArray | None = None,
) -> FloatArray:
    """Attract/repel SGD over the graph edges.

    ``chunk_count`` controls in-epoch feedback:

    - ``chunk_count <= 1`` (default): every gradient in an epoch is computed from one
      snapshot and applied together. Fully reproducible. This is the historical behaviour.
    - ``chunk_count > 1``: the epoch's due edges are shuffled and processed in that many
      chunks. Each chunk is computed from the positions the previous chunks already moved.
      This is a partial-feedback middle ground between the snapshot update and umap-learn's
      in-place (Hogwild) walk.

    Attraction pulls both ends of a due edge. Repulsion pushes the head away from
    ``negative_sample_rate`` uniform samples. Every gradient component is clamped to +/-4.

    densMAP: when ``dens_lambda > 0`` and ``dens_R`` (the z-scored log original-space radius per
    vertex) is given, a density-preserving term is added to the attraction gradient for the last
    ``dens_frac`` fraction of epochs. It pushes each point's local embedding radius to correlate
    with its original radius, so dense/sparse regions stay dense/sparse. This forces the numpy
    path (the Rust kernel does plain UMAP). Faithful port of umap-learn's densmap.
    """
    head = graph.head
    tail = graph.tail
    epochs_per_sample = make_epochs_per_sample(graph.weight, n_epochs)
    dens_active = dens_lambda > 0.0 and dens_R is not None

    # Native fast path: the Rust kernel does the true in-place walk. chunk_count>1's in-epoch
    # feedback is a numpy-only approximation, and the Rust kernel has no densMAP term, so only
    # take this path for the plain default.
    if _accel_optimize is not None and chunk_count <= 1 and not dens_active:
        seed = int(rng.integers(0, 2**63 - 1))
        result = _accel_optimize(
            np.ascontiguousarray(embedding, dtype=np.float64),
            head,
            tail,
            epochs_per_sample,
            n_epochs,
            a,
            b,
            gamma,
            negative_sample_rate,
            initial_alpha,
            seed,
        )
        return np.ascontiguousarray(result, dtype=np.float64)

    epoch_of_next = epochs_per_sample.copy()
    # A copy, not ascontiguousarray. emb is accumulated in place (emb += ...). An already
    # C-contiguous float64 input (e.g. the caller's `init`) would otherwise be mutated (#15).
    emb = np.array(embedding, dtype=np.float64, order="C")
    n_vertices = emb.shape[0]

    def edge_delta(
        edge_idx: IntArray, positions: FloatArray, alpha: float, density: _DensityStats | None
    ) -> FloatArray:
        h = head[edge_idx]
        t = tail[edge_idx]
        delta = np.zeros_like(positions)

        pos_h = positions[h]  # gathered once; reused by attraction and every repulsion draw
        diff = pos_h - positions[t]
        d2: FloatArray = np.einsum("ij,ij->i", diff, diff)
        attract_coeff = np.zeros_like(d2)
        nonzero = d2 > 0.0
        d2_nz = d2[nonzero]
        attract_coeff[nonzero] = (-2.0 * a * b * np.power(d2_nz, b - 1.0)) / (
            1.0 + a * np.power(d2_nz, b)
        )
        grad = np.clip(attract_coeff[:, None] * diff, -_GRAD_CLIP, _GRAD_CLIP) * alpha
        _scatter_add(delta, h, grad)
        _scatter_add(delta, t, -grad)

        # densMAP: extra force on this attraction edge that nudges the local embedding radius
        # toward the original radius. Faithful to umap-learn's density gradient (dens_R is the
        # z-scored log original radius; x/phi_sum/std/mean/cov are this epoch's embedding stats).
        if density is not None:
            safe = np.where(nonzero, d2, 1.0)  # avoid pow(0, b-1) blowups; masked to 0 below
            phi = 1.0 / (1.0 + a * np.power(safe, b))
            dphi = a * b * np.power(safe, b - 1.0) * phi
            common = 1.0 - b * (1.0 - phi)
            drt = (phi / density.phi_sum[t]) * (common / np.exp(density.x[t]) + dphi)
            drh = (phi / density.phi_sum[h]) * (common / np.exp(density.x[h]) + dphi)
            re_std_sq = density.re_std * density.re_std
            w_t = density.R[t] - density.re_cov * (density.x[t] - density.re_mean) / re_std_sq
            w_h = density.R[h] - density.re_cov * (density.x[h] - density.re_mean) / re_std_sq
            grad_cor = (
                dens_lambda
                * density.mu_tot
                * (w_t * drt + w_h * drh)
                / (graph.weight[edge_idx] * density.re_std)
                / n_vertices
            )
            dens_grad = np.clip(2.0 * grad_cor[:, None] * diff, -_GRAD_CLIP, _GRAD_CLIP) * alpha
            dens_grad[~nonzero] = 0.0  # d2==0 => diff==0 => no density direction
            _scatter_add(delta, h, dens_grad)
            _scatter_add(delta, t, -dens_grad)

        # One blocked draw instead of negative_sample_rate sequential ones: PCG64 fills the
        # (rate, m) buffer in the same stream order, so the samples are bit-identical.
        samples: IntArray = rng.integers(
            0, n_vertices, size=(negative_sample_rate, edge_idx.shape[0])
        )
        for k_samp in samples:
            grad_n = _repel_grad(pos_h, positions[k_samp], a=a, b=b, gamma=gamma, alpha=alpha)
            grad_n[k_samp == h] = 0.0  # a vertex never repels itself (umap `continue`)
            _scatter_add(delta, h, grad_n)

        return delta

    dens_mu_tot = float(graph.weight.sum())

    def density_stats(positions: FloatArray) -> _DensityStats:
        # One pass over all edges: the embedding local radius per vertex (weighted by the low-D
        # kernel phi), plus the correlation stats. Fixed while points move during the epoch.
        assert dens_R is not None
        d = positions[head] - positions[tail]
        d2f: FloatArray = np.einsum("ij,ij->i", d, d)
        phi = 1.0 / (1.0 + a * np.power(d2f, b))
        pd2 = phi * d2f
        re_num = np.bincount(head, weights=pd2, minlength=n_vertices) + np.bincount(
            tail, weights=pd2, minlength=n_vertices
        )
        phi_sum = np.bincount(head, weights=phi, minlength=n_vertices) + np.bincount(
            tail, weights=phi, minlength=n_vertices
        )
        x = np.log(1e-8 + re_num / phi_sum)
        return _DensityStats(
            R=dens_R,
            x=x,
            phi_sum=phi_sum,
            re_std=float(np.sqrt(np.var(x) + dens_var_shift)),
            re_mean=float(np.mean(x)),
            re_cov=float(np.dot(x, dens_R) / (n_vertices - 1)),
            mu_tot=dens_mu_tot,
        )

    for epoch in range(n_epochs):
        alpha = initial_alpha * (1.0 - epoch / n_epochs)
        due: IntArray = np.where((epoch_of_next <= epoch) & (epochs_per_sample > 0.0))[0]
        if due.shape[0] == 0:
            continue

        # densMAP turns on for the last dens_frac fraction of epochs (a plain-UMAP warmup first).
        density = (
            density_stats(emb) if dens_active and (epoch + 1) / n_epochs > 1.0 - dens_frac else None
        )

        if chunk_count <= 1:
            emb += edge_delta(due, emb, alpha, density)  # in place: no fresh (n, dim) alloc
        else:
            shuffled = due[rng.permutation(due.shape[0])]
            for chunk in np.array_split(shuffled, chunk_count):
                if chunk.shape[0] == 0:
                    continue
                emb += edge_delta(chunk, emb, alpha, density)

        epoch_of_next[due] += epochs_per_sample[due]

    return emb


def optimize_transform(
    new_embedding: FloatArray,
    fixed_embedding: FloatArray,
    head: IntArray,
    tail: IntArray,
    weight: FloatArray,
    n_epochs: int,
    *,
    a: float,
    b: float,
    rng: np.random.Generator,
    negative_sample_rate: int = 5,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
) -> FloatArray:
    """Place new points into a fixed layout: only ``new_embedding`` moves.

    Edges run from new points to their training neighbours. ``head`` indexes
    ``new_embedding``; ``tail`` indexes the frozen ``fixed_embedding``. Attraction pulls
    each new point toward its neighbours. Repulsion pushes it from ``negative_sample_rate``
    random training points. The training layout never moves, so the picture stays stable.
    """
    epochs_per_sample = make_epochs_per_sample(weight, n_epochs)
    epoch_of_next = epochs_per_sample.copy()
    emb = np.array(new_embedding, dtype=np.float64, order="C")  # copy: accumulated in place (#15)
    ref = fixed_embedding
    n_ref = ref.shape[0]

    for epoch in range(n_epochs):
        alpha = initial_alpha * (1.0 - epoch / n_epochs)
        due: IntArray = np.where((epoch_of_next <= epoch) & (epochs_per_sample > 0.0))[0]
        if due.shape[0] == 0:
            continue

        h = head[due]
        t = tail[due]
        delta = np.zeros_like(emb)

        emb_h = emb[h]  # gathered once; reused by attraction and every repulsion draw
        diff = emb_h - ref[t]
        d2: FloatArray = np.einsum("ij,ij->i", diff, diff)
        attract_coeff = np.zeros_like(d2)
        nonzero = d2 > 0.0
        d2_nz = d2[nonzero]
        attract_coeff[nonzero] = (-2.0 * a * b * np.power(d2_nz, b - 1.0)) / (
            1.0 + a * np.power(d2_nz, b)
        )
        grad = np.clip(attract_coeff[:, None] * diff, -_GRAD_CLIP, _GRAD_CLIP) * alpha
        _scatter_add(delta, h, grad)  # only the new points move; ref is frozen

        # One blocked draw instead of negative_sample_rate sequential ones: PCG64 fills the
        # (rate, m) buffer in the same stream order, so the samples are bit-identical.
        samples: IntArray = rng.integers(0, n_ref, size=(negative_sample_rate, due.shape[0]))
        for k_samp in samples:
            grad_n = _repel_grad(emb_h, ref[k_samp], a=a, b=b, gamma=gamma, alpha=alpha)
            _scatter_add(delta, h, grad_n)

        emb = emb + delta
        epoch_of_next[due] += epochs_per_sample[due]

    return emb
