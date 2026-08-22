"""Real-data benchmark: fastumap vs umap-learn vs PCA on MNIST / Fashion-MNIST.

    uv run python bench/mnist.py [--dataset mnist|fashion] [--sizes 5000,10000,20000]

Downloads once into bench/data/ (gitignored) via scikit-learn's OpenML fetcher, caches as
.npy, then reports wall-clock, overlap@15 and the global pairwise-distance Spearman
correlation for each method at each size. scikit-learn and umap-learn are bench-only deps
(not in the runtime closure). umap-learn is skipped with a note if it is not installed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fastumap import global_distance_correlation, neighbor_overlap, umap_project
from fastumap._types import FloatArray

_CACHE = Path(__file__).resolve().parent / "data"
_OPENML = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}


def load(dataset: str) -> FloatArray:
    """Load the dataset as (n, 784) float64, caching the raw array under bench/data/."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached = _CACHE / f"{dataset}.npy"
    if cached.exists():
        return np.asarray(np.load(cached), dtype=np.float64)
    from sklearn.datasets import fetch_openml

    # parser="liac-arff" avoids the pandas requirement of the default "auto" parser.
    data = fetch_openml(_OPENML[dataset], version=1, as_frame=False, parser="liac-arff")
    x = np.ascontiguousarray(data.data, dtype=np.float64)
    np.save(cached, x)
    return x


def _pca(x: FloatArray, dim: int) -> FloatArray:
    centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return np.ascontiguousarray(centered @ vt[:dim].T, dtype=np.float64)


def _umap_learn(x: FloatArray, seed: int) -> FloatArray | None:
    try:
        import umap
    except ImportError:
        return None
    return np.ascontiguousarray(
        umap.UMAP(n_components=2, random_state=seed).fit_transform(x), dtype=np.float64
    )


def _row(name: str, x: FloatArray, emb: FloatArray | None, dt: float) -> str:
    if emb is None:
        return f"  {name:<12} {'(not installed)':>12}"
    ov = neighbor_overlap(x, emb, 15)
    corr = global_distance_correlation(x, emb)
    return f"  {name:<12} {dt:8.1f}s   overlap@15={ov:.3f}   dist_corr={corr:.3f}"


def benchmark(dataset: str, sizes: list[int], seed: int = 42) -> None:
    full = load(dataset)
    rng = np.random.default_rng(seed)
    print(f"\n== {dataset} ({full.shape[0]} x {full.shape[1]}) ==")
    for n in sizes:
        if n > full.shape[0]:
            print(f"\n-- n={n}: only {full.shape[0]} rows available, skipping --")
            continue
        idx = rng.choice(full.shape[0], size=n, replace=False)
        x = np.ascontiguousarray(full[idx])
        print(f"\n-- n={n} --")
        t = time.perf_counter()
        ours = umap_project(x, 2, random_state=seed)
        print(_row("fastumap", x, ours, time.perf_counter() - t))
        t = time.perf_counter()
        pca = _pca(x, 2)
        print(_row("pca", x, pca, time.perf_counter() - t))
        t = time.perf_counter()
        ref = _umap_learn(x, seed)
        print(_row("umap-learn", x, ref, time.perf_counter() - t))


def main() -> None:
    parser = argparse.ArgumentParser(description="fastumap real-data benchmark")
    parser.add_argument("--dataset", choices=list(_OPENML), default="mnist")
    parser.add_argument("--sizes", default="5000,10000,20000")
    args = parser.parse_args()
    benchmark(args.dataset, [int(s) for s in args.sizes.split(",")])


if __name__ == "__main__":
    main()
