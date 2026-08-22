"""Grading harness: one command, one table, four layouts, many seeds.

    uv run python -m fastumap.grade [--n 1000] [--seeds 5] [--dim 64] [--clusters 8]

Prints neighbor_overlap@15 and the global pairwise-distance Spearman correlation for
fastumap, umap-learn, PCA and a random control, as mean +/- std over the seeds. This
is a dev/bench tool: it imports umap-learn and scikit-learn, which are NOT runtime
dependencies, so it lives behind the same test/bench extras.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._types import FloatArray
from .metrics import global_distance_correlation, neighbor_overlap, random_layout
from .projection import umap_project


@dataclass(frozen=True)
class Row:
    name: str
    overlap: list[float]
    correlation: list[float]


def _clusters(n: int, dim: int, clusters: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(clusters, dim))
    labels = rng.integers(0, clusters, size=n)
    return np.ascontiguousarray(centers[labels] + rng.normal(size=(n, dim)), dtype=np.float64)


def _pca(x: FloatArray, dim: int) -> FloatArray:
    centered = x - x.mean(axis=0, keepdims=True)
    # economy SVD; deterministic, no sklearn needed for the baseline.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return np.ascontiguousarray(centered @ vt[:dim].T, dtype=np.float64)


def _umap_learn(x: FloatArray, dim: int, seed: int) -> FloatArray | None:
    try:
        import umap
    except ImportError:
        return None
    model = umap.UMAP(n_components=dim, random_state=seed)
    return np.ascontiguousarray(model.fit_transform(x), dtype=np.float64)


def grade(n: int, seeds: int, dim: int, clusters: int, k: int = 15) -> list[Row]:
    layouts: dict[str, Callable[[FloatArray, int], FloatArray | None]] = {
        "fastumap": lambda x, s: umap_project(x, 2, random_state=s),
        "umap-learn": lambda x, s: _umap_learn(x, 2, s),
        "pca": lambda x, _s: _pca(x, 2),
        "random": lambda x, s: random_layout(x.shape[0], 2, random_state=s),
    }
    rows: dict[str, Row] = {name: Row(name, [], []) for name in layouts}
    for seed in range(seeds):
        x = _clusters(n, dim, clusters, seed)
        for name, fn in layouts.items():
            emb = fn(x, seed)
            if emb is None:
                continue
            rows[name].overlap.append(neighbor_overlap(x, emb, k))
            rows[name].correlation.append(global_distance_correlation(x, emb))
    return [r for r in rows.values() if r.overlap]


def _fmt(values: list[float]) -> str:
    if not values:
        return "   n/a    "
    mean = statistics.mean(values)
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} +/- {spread:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade fastumap against umap-learn, PCA, random.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    rows = grade(args.n, args.seeds, args.dim, args.clusters)
    print(
        f"\nn={args.n}  seeds={args.seeds}  dim={args.dim}  clusters={args.clusters}"
        "  (overlap@15, higher better; global dist corr, higher better)\n"
    )
    print(f"{'layout':<12} {'overlap@15':<18} {'global dist corr':<18}")
    print("-" * 48)
    for row in rows:
        print(f"{row.name:<12} {_fmt(row.overlap):<18} {_fmt(row.correlation):<18}")
    print()


if __name__ == "__main__":
    main()
