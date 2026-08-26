"""The 1M x 128 clustering probe, run inside a CPU-capped container by bench/minibatch_1m.sh.

The claim under test, from the roadmap: cluster 1M x 128 into ~64 groups in under 10 s on a
2-vCPU container, with inertia within ~5% of full Lloyd. Full Lloyd at that size is ~154
core-seconds, so it cannot meet the bar; mini-batch samples a batch per iteration instead.

Data is 1M x 128 Gaussian clusters with *overlapping* spread, not well-separated blobs. That is
deliberate. Mini-batch's known weakness is the perfectly-separated case, where one misplaced
centroid pair wrecks the objective and Lloyd's full recompute does not; overlapping data is both
harder for it and closer to the embeddings people actually cluster.

Prints a markdown row per implementation to stdout; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fastumap._types import FloatArray


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def make_data(n: int, dim: int, k: int, seed: int = 0) -> FloatArray:
    """Overlapping Gaussian clusters: centre spread 1.5 sigma, so the clusters genuinely touch."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=1.5, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    return np.ascontiguousarray(centers[labels] + rng.normal(size=(n, dim)), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="1M x 128 clustering probe under a CPU cap")
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--cap", default="?", help="vCPU cap, for the row label")
    parser.add_argument("--skip-lloyd", action="store_true", help="Lloyd is minutes at 1M")
    parser.add_argument("--sift", action="store_true", help="use the cached SIFT1M descriptors")
    args = parser.parse_args()

    import os

    from fastumap import clustering

    if args.sift:
        # The canonical 1M x 128 benchmark set (Jegou et al.), what faiss benchmarks on — a real
        # distribution rather than anything this script could invent.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cluster_bench import _download_sift, _read_fvecs  # pyright: ignore[reportPrivateUsage]

        _stderr(f"reading SIFT1M ... (cpu_count={os.cpu_count()})")
        x = _read_fvecs(_download_sift(), args.n)
    else:
        _stderr(f"building {args.n} x {args.dim} ... (cpu_count={os.cpu_count()})")
        x = make_data(args.n, args.dim, args.k)
    _stderr(f"data {x.shape}, {x.nbytes / 1e9:.2f} GB")

    start = time.perf_counter()
    mb_labels, mb_centroids = clustering.kmeans(x, args.k)  # auto: mini-batch above the threshold
    mb_s = time.perf_counter() - start
    mb_inertia = clustering._inertia(x, mb_labels, mb_centroids)  # pyright: ignore[reportPrivateUsage]
    groups = len(set(mb_labels.tolist()))
    _stderr(f"mini-batch {mb_s:7.2f}s  inertia={mb_inertia:.5e}  groups={groups}")
    print(f"| {args.cap} | mini-batch | {mb_s:.1f} s | {mb_inertia:.4e} | {groups} |")

    if args.skip_lloyd:
        return
    start = time.perf_counter()
    ll_labels, ll_centroids = clustering.kmeans(x, args.k, batch_size=0)
    ll_s = time.perf_counter() - start
    ll_inertia = clustering._inertia(x, ll_labels, ll_centroids)  # pyright: ignore[reportPrivateUsage]
    _stderr(f"lloyd      {ll_s:7.2f}s  inertia={ll_inertia:.5e}")
    print(
        f"| {args.cap} | full Lloyd | {ll_s:.1f} s | {ll_inertia:.4e} | "
        f"{len(set(ll_labels.tolist()))} |"
    )
    print(
        f"| {args.cap} | ratio | {mb_s / ll_s:.2f}x time | {mb_inertia / ll_inertia:.3f}x inertia | |"
    )


if __name__ == "__main__":
    main()
