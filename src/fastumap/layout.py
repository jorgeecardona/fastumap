"""Initialisation and optimisation of the low-dimensional embedding.

spectral_layout gives the deterministic starting positions (leading eigenvectors of
the normalised fuzzy graph, guarded against disconnected components); optimize_layout
runs the attract/repel SGD that UMAP's cross-entropy objective reduces to.
"""

from __future__ import annotations

import numpy as np

from . import _scipy
from ._types import FloatArray, IntArray
from .graph import FuzzyGraph, make_epochs_per_sample

_GRID_SPACING = 3.0
_SCALE = 10.0
_INIT_NOISE = 1e-4
_GRAD_CLIP = 4.0


def _scatter_add(target: FloatArray, idx: IntArray, values: FloatArray) -> None:
    """In-place ``target[idx] += values`` accumulating duplicate indices.

    Replaces ``np.add.at`` (a slow element-by-element scatter) with one compiled
    ``np.bincount`` reduction per output dimension. The embedding is only 2-3 dims, so
    this is a handful of bincounts and dominates the SGD's speed. Deterministic.
    """
    n, dim = target.shape
    for d in range(dim):
        target[:, d] += np.bincount(idx, weights=values[:, d], minlength=n)


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

    For a component too small for ARPACK to extract ``dim+1`` eigenpairs, fall back
    to a deterministic spread along the axes so nothing collapses to a point.
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

    Without this, the leading eigenvectors of a disconnected graph are component
    indicators and each component collapses to a single point.
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
) -> FloatArray:
    """Attract/repel SGD over the graph edges.

    ``chunk_count`` controls in-epoch feedback. With ``chunk_count <= 1`` (default) every
    gradient in an epoch is computed from one snapshot and applied together — fully
    reproducible and the historical behaviour. With ``chunk_count > 1`` the epoch's due
    edges are shuffled and processed in that many chunks, each computed from the positions
    the previous chunks already moved — a partial-feedback middle ground between the
    snapshot update and umap-learn's in-place (Hogwild) walk. Attraction pulls both ends of
    a due edge; repulsion pushes the head away from ``negative_sample_rate`` uniform
    samples. Every gradient component is clamped to +/-4.
    """
    head = graph.head
    tail = graph.tail
    epochs_per_sample = make_epochs_per_sample(graph.weight, n_epochs)
    epoch_of_next = epochs_per_sample.copy()
    emb = np.ascontiguousarray(embedding, dtype=np.float64)
    n_vertices = emb.shape[0]

    def edge_delta(edge_idx: IntArray, positions: FloatArray, alpha: float) -> FloatArray:
        h = head[edge_idx]
        t = tail[edge_idx]
        delta = np.zeros_like(positions)

        diff = positions[h] - positions[t]
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

        for _sample in range(negative_sample_rate):
            k_samp: IntArray = rng.integers(0, n_vertices, size=edge_idx.shape[0])
            diff_n = positions[h] - positions[k_samp]
            d2n: FloatArray = np.einsum("ij,ij->i", diff_n, diff_n)
            repel_coeff = np.zeros_like(d2n)
            positive = d2n > 0.0
            d2p = d2n[positive]
            repel_coeff[positive] = (2.0 * gamma * b) / (
                (0.001 + d2p) * (1.0 + a * np.power(d2p, b))
            )
            grad_n = (
                np.where(
                    positive[:, None],
                    np.clip(repel_coeff[:, None] * diff_n, -_GRAD_CLIP, _GRAD_CLIP),
                    _GRAD_CLIP,
                )
                * alpha
            )
            grad_n[k_samp == h] = 0.0  # a vertex never repels itself (umap `continue`)
            _scatter_add(delta, h, grad_n)

        return delta

    for epoch in range(n_epochs):
        alpha = initial_alpha * (1.0 - epoch / n_epochs)
        due: IntArray = np.where((epoch_of_next <= epoch) & (epochs_per_sample > 0.0))[0]
        if due.shape[0] == 0:
            continue

        if chunk_count <= 1:
            emb = emb + edge_delta(due, emb, alpha)
        else:
            shuffled = due[rng.permutation(due.shape[0])]
            for chunk in np.array_split(shuffled, chunk_count):
                if chunk.shape[0] == 0:
                    continue
                emb = emb + edge_delta(chunk, emb, alpha)

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

    Edges run from new points (``head`` indexes ``new_embedding``) to their training
    neighbours (``tail`` indexes the frozen ``fixed_embedding``). Attraction pulls each
    new point toward its neighbours; repulsion pushes it from ``negative_sample_rate``
    random training points. The training layout never moves, so the picture stays stable.
    """
    epochs_per_sample = make_epochs_per_sample(weight, n_epochs)
    epoch_of_next = epochs_per_sample.copy()
    emb = np.ascontiguousarray(new_embedding, dtype=np.float64)
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

        diff = emb[h] - ref[t]
        d2: FloatArray = np.einsum("ij,ij->i", diff, diff)
        attract_coeff = np.zeros_like(d2)
        nonzero = d2 > 0.0
        d2_nz = d2[nonzero]
        attract_coeff[nonzero] = (-2.0 * a * b * np.power(d2_nz, b - 1.0)) / (
            1.0 + a * np.power(d2_nz, b)
        )
        grad = np.clip(attract_coeff[:, None] * diff, -_GRAD_CLIP, _GRAD_CLIP) * alpha
        _scatter_add(delta, h, grad)  # only the new points move; ref is frozen

        for _sample in range(negative_sample_rate):
            k_samp: IntArray = rng.integers(0, n_ref, size=due.shape[0])
            diff_n = emb[h] - ref[k_samp]
            d2n: FloatArray = np.einsum("ij,ij->i", diff_n, diff_n)
            repel_coeff = np.zeros_like(d2n)
            positive = d2n > 0.0
            d2p = d2n[positive]
            repel_coeff[positive] = (2.0 * gamma * b) / (
                (0.001 + d2p) * (1.0 + a * np.power(d2p, b))
            )
            grad_n = (
                np.where(
                    positive[:, None],
                    np.clip(repel_coeff[:, None] * diff_n, -_GRAD_CLIP, _GRAD_CLIP),
                    _GRAD_CLIP,
                )
                * alpha
            )
            _scatter_add(delta, h, grad_n)

        emb = emb + delta
        epoch_of_next[due] += epochs_per_sample[due]

    return emb
