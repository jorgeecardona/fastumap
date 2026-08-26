"""What fastumap costs on a small, CPU-throttled container — the shape production actually sees.

This is the *inside-the-container* half of the probe. `bench/constrained.sh` is the driver: it
runs this file under `docker --cpus=0.5`, `--cpus=1` and `--cpus=2` and collects the rows into a
dated markdown table under docs/bench/.

Why a separate probe at all: every other benchmark here runs on a workstation with cores to
spare, and the README's constrained-CPU claim ("minutes on 0.5-2 vCPU") had nothing behind it.
Jorge's live Fargate ECS numbers are the ground truth for the SGD path — 250 / 500 / 1000 / 5000
points at 7.6 / 28 / 83 / 270 s, with one case over 1000 s — and clustering under a cap had never
been measured at all. This reproduces that curve locally so a change to it is visible.

`docker --cpus` sets a CFS quota, which is how Fargate's cgroup meters CPU, so the throttling
here is the same mechanism rather than an approximation of it.

Prints one markdown table row per measurement to stdout, and nothing else, so the driver can
concatenate them. Anything it wants a human to see goes to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_cost_ms() -> float:
    """Cost of `import fastumap` in this interpreter. Only meaningful once per process."""
    start = time.perf_counter()
    importlib.import_module("fastumap")
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="constrained-CPU probe (runs inside a container)")
    parser.add_argument("--cap", default=os.environ.get("CPUS", "?"), help="vCPU cap, for the row")
    parser.add_argument("--sizes", default="250,500,1000,5000")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--clusters", type=int, default=8, help="k for the clustering half")
    parser.add_argument(
        "--numpy-path",
        action="store_true",
        help="force the pure-numpy SGD by disabling the native kernel, to measure what a "
        "pre-0.2.1 install (no compiled _accel) would have cost on the same container",
    )
    parser.add_argument("--skip-cluster", action="store_true", help="time the fit only")
    args = parser.parse_args()

    ms = import_cost_ms()
    _stderr(f"import fastumap: {ms:.0f} ms")

    import numpy as np

    from fastumap import accelerator_active, cluster, umap_project

    if args.numpy_path:
        # layout._accel_optimize is read at call time, so clearing it routes the SGD through the
        # pure-numpy path — the same code a pre-0.2.1 install (before the kernel was merged in)
        # would have run. Nothing else changes: same data, same seed, same container.
        from fastumap import layout

        layout._accel_optimize = None  # pyright: ignore[reportPrivateUsage]
    _stderr(f"native kernel active: {accelerator_active()}  cpu_count={os.cpu_count()}")
    print(f"| {args.cap} | import | | {ms:.0f} ms | |")

    for n in (int(s) for s in args.sizes.split(",")):
        x = np.ascontiguousarray(
            np.random.default_rng(0).normal(size=(n, args.dim)), dtype=np.float64
        )

        start = time.perf_counter()
        umap_project(x, 2, random_state=0)
        fit_s = time.perf_counter() - start
        _stderr(f"  fit     n={n:<6} {fit_s:8.1f} s")

        if args.skip_cluster:
            print(f"| {args.cap} | {n} | {args.dim} | {fit_s:.1f} s | (skipped) |")
            continue

        # k-means on the same input, so the two halves are comparable at each n. Clustering does
        # not need a layout first, which is exactly why it is worth timing separately.
        start = time.perf_counter()
        cluster(x, method="kmeans", n_clusters=args.clusters, random_state=0)
        cluster_s = time.perf_counter() - start
        _stderr(f"  cluster n={n:<6} {cluster_s:8.1f} s")

        print(f"| {args.cap} | {n} | {args.dim} | {fit_s:.1f} s | {cluster_s:.2f} s |")


if __name__ == "__main__":
    main()
