"""One command that regenerates every benchmark number the READMEs quote.

    uv run python bench/report.py            # or: make bench-report

The READMEs used to carry numbers pasted by hand from ad-hoc runs, so a reader had no way to
re-derive them and no way to notice when they drifted. This writes a dated markdown report under
docs/bench/ instead: the same two tables, plus the machine and the package versions they were
measured on, so "26 s at n=5000" is a claim someone else can check.

Two tables:

- **Projection quality** — fastumap vs umap-learn vs PCA on MNIST at several sizes: wall time,
  neighbour overlap@15, and the global pairwise-distance Spearman correlation. This is the
  README's "Quality relative to umap-learn" table.
- **Clustering** — k-means wall time *and* inertia (the k-means objective), native kernel vs the
  numpy path, so speed is never quoted without a quality number beside it. Synthetic 128-d data by
  default; `--sift` uses the real SIFT1M descriptors that bench/cluster_bench.py downloads.

scikit-learn and umap-learn are bench-only dependencies. A missing umap-learn is reported in the
table as "(not installed)" rather than silently dropped — a table with a hole in it is honest, a
table that quietly omits the comparison is not.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np

from fastumap import accelerator_active, global_distance_correlation, neighbor_overlap
from fastumap._types import FloatArray

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "docs" / "bench"
_SEED = 42

# bench/ is a directory of scripts, not a package. Put it on sys.path so this one can reuse
# mnist.py's OpenML cache and cluster_bench.py's SIFT reader instead of copying them.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _versions() -> dict[str, str]:
    """Every version that can move a number in this report."""
    out: dict[str, str] = {}
    for name in ("fastumap", "numpy", "scipy", "umap-learn", "scikit-learn", "faiss-cpu"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "(not installed)"
    return out


def _cpu_model() -> str:
    """/proc/cpuinfo on Linux, sysctl on macOS, platform's guess elsewhere."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _load_average() -> tuple[str, bool]:
    """1/5/15-minute load average, and whether the machine was busy enough to distort wall times.

    Every timing here is wall-clock, so a contended machine inflates all of them — and unevenly,
    since BLAS-heavy PCA and single-threaded SGD lose different amounts. Recording the load is what
    makes a slow row readable as "the box was busy" instead of "the library got slower".
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return "(unavailable)", False
    cores = os.cpu_count() or 1
    return f"{one:.2f} / {five:.2f} / {fifteen:.2f} (over {cores} cores)", one > cores * 0.7


def _total_ram_gb() -> str:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 1024 / 1024:.1f} GB"
    except (OSError, ValueError):
        pass
    return "(unknown)"


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


def projection_rows(dataset: str, sizes: list[int]) -> list[str]:
    """One markdown row per (size, method): wall time, overlap@15, global distance correlation."""
    from mnist import load  # bench/mnist.py owns the OpenML download + .npy cache

    from fastumap import umap_project

    full = load(dataset)
    rng = np.random.default_rng(_SEED)
    rows: list[str] = []
    for n in sizes:
        if n > full.shape[0]:
            print(f"  n={n}: only {full.shape[0]} rows available, skipping")
            continue
        idx = rng.choice(full.shape[0], size=n, replace=False)
        x = np.ascontiguousarray(full[idx])
        fits: list[tuple[str, Callable[[FloatArray], FloatArray | None]]] = [
            ("fastumap", lambda a: umap_project(a, 2, random_state=_SEED)),
            ("umap-learn", lambda a: _umap_learn(a, _SEED)),
            ("pca", lambda a: _pca(a, 2)),
        ]
        for name, fit in fits:
            start = time.perf_counter()
            emb = fit(x)
            elapsed = time.perf_counter() - start
            if emb is None:
                rows.append(f"| {n} | {name} | (not installed) | | |")
                print(f"  n={n:<6} {name:<11} (not installed)")
                continue
            overlap = neighbor_overlap(x, emb, 15)
            corr = global_distance_correlation(x, emb)
            rows.append(f"| {n} | {name} | {elapsed:.0f} s | {overlap:.3f} | {corr:.3f} |")
            print(f"  n={n:<6} {name:<11} {elapsed:7.1f}s  overlap={overlap:.3f}  corr={corr:.3f}")
    return rows


def _inertia(x: FloatArray, labels: np.ndarray, k: int) -> float:
    """Sum of squared distances to each point's assigned centroid — the k-means objective."""
    total = 0.0
    for c in range(k):
        pts = x[labels == c]
        if len(pts):
            total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def _cluster_data(n: int, use_sift: bool) -> tuple[FloatArray, str]:
    if use_sift:
        from cluster_bench import _download_sift, _read_fvecs  # pyright: ignore[reportPrivateUsage]

        return _read_fvecs(_download_sift(), n), "SIFT1M descriptors"
    # Synthetic but shaped like SIFT (128-d, clustered), so the report runs with no download.
    rng = np.random.default_rng(_SEED)
    centers = rng.normal(scale=8.0, size=(64, 128))
    labels = rng.integers(0, 64, size=n)
    x = centers[labels] + rng.normal(size=(n, 128))
    return np.ascontiguousarray(x, dtype=np.float64), "synthetic 128-d blobs"


def cluster_rows(n: int, k: int, n_iter: int, use_sift: bool) -> tuple[list[str], str]:
    """k-means wall time and inertia for the native kernel and the numpy path."""
    from fastumap.clustering import kmeans as numpy_kmeans

    x, source = _cluster_data(n, use_sift)
    rows: list[str] = []
    try:
        accel = importlib.import_module("fastumap._accel")
    except ImportError:
        rows.append(f"| accel (Rust) | {n} | {k} | (kernel not built) | |")
    else:
        start = time.perf_counter()
        labels = accel.kmeans(np.ascontiguousarray(x), k, n_iter, _SEED)
        elapsed = time.perf_counter() - start
        inertia = _inertia(x, labels, k)
        rows.append(f"| accel (Rust) | {n} | {k} | {elapsed:.2f} s | {inertia:.3e} |")
        print(f"  accel k-means {elapsed:7.2f}s  inertia={inertia:.3e}")
    start = time.perf_counter()
    np_labels, _ = numpy_kmeans(x, k, n_iter=n_iter, random_state=_SEED)
    elapsed = time.perf_counter() - start
    inertia = _inertia(x, np_labels, k)
    rows.append(f"| numpy | {n} | {k} | {elapsed:.2f} s | {inertia:.3e} |")
    print(f"  numpy k-means {elapsed:7.2f}s  inertia={inertia:.3e}")
    return rows, source


def render(
    dataset: str,
    sizes: list[int],
    proj: list[str],
    clust: list[str],
    cluster_source: str,
    command: str,
) -> str:
    versions = _versions()
    version_lines = "\n".join(f"| {name} | {ver} |" for name, ver in versions.items())
    load, busy = _load_average()
    warning = (
        "\n> **This run was measured on a busy machine.** The load average below is a large"
        " fraction of the core count, so every wall time here is inflated and the inflation is not"
        " even across methods. The quality columns (overlap@15, global corr) are deterministic and"
        " unaffected. Re-run on an idle machine before quoting the timings.\n"
        if busy
        else ""
    )
    return f"""---
title: fastumap benchmark report
date: {date.today().isoformat()}
generated_by: bench/report.py
---

# fastumap benchmark report — {date.today().isoformat()}
{warning}
Generated, not hand-written. Regenerate with:

```
{command}
```

Every number below comes from that one run on the machine described here. Comparing two reports
only means something when the machine rows match.

## Machine

| | |
|---|---|
| CPU | {_cpu_model()} |
| logical cores | {os.cpu_count() or 0} |
| load average (1/5/15 min) | {load} |
| RAM | {_total_ram_gb()} |
| platform | {platform.platform()} |
| python | {platform.python_version()} |
| native kernel active | {accelerator_active()} |

## Versions

| package | version |
|---|---|
{version_lines}

## Projection quality — {dataset} (784-dimensional)

fastumap vs umap-learn vs PCA at n = {", ".join(str(s) for s in sizes)}, seed {_SEED}.

- **overlap@15** — the fraction of each point's 15 input-space neighbours still among its 15
  nearest neighbours in the embedding. Local structure. Higher is better.
- **global corr** — Spearman correlation between input-space and embedding pairwise distances.
  Global structure. Higher is better.

| n | method | wall time | overlap@15 | global corr |
|---|---|---:|---:|---:|
{chr(10).join(proj)}

umap-learn's wall times in a single process reuse the numba compilation from its first fit; a
fresh process pays that compilation (roughly 20 s) on every invocation. fastumap has no
compilation step.

## Clustering — k-means

{cluster_source}. Inertia is the k-means objective (sum of squared distances to the assigned
centroid): lower is tighter. It sits beside the wall time on purpose — a faster k-means that
optimises less is not faster.

| implementation | n | k | wall time | inertia |
|---|---|---:|---:|---:|
{chr(10).join(clust)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="regenerate the fastumap benchmark report")
    parser.add_argument("--dataset", choices=("mnist", "fashion"), default="mnist")
    parser.add_argument("--sizes", default="5000,10000,20000", help="projection sizes")
    parser.add_argument("--kmeans-n", type=int, default=200_000)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--sift", action="store_true", help="cluster real SIFT1M (downloads ~1 GB)")
    parser.add_argument("--out", default=str(_OUT))
    parser.add_argument("--skip-projection", action="store_true")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    command = (
        f"uv run python bench/report.py --dataset {args.dataset} --sizes {args.sizes}"
        f" --kmeans-n {args.kmeans_n} --k {args.k} --iter {args.iter}"
        + (" --sift" if args.sift else "")
    )

    proj: list[str] = []
    if not args.skip_projection:
        print(f"projection quality ({args.dataset}) ...")
        proj = projection_rows(args.dataset, sizes)
    print("clustering ...")
    clust, cluster_source = cluster_rows(args.kmeans_n, args.k, args.iter, args.sift)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.dataset}-{date.today().isoformat()}.md"
    path.write_text(render(args.dataset, sizes, proj, clust, cluster_source, command))
    print(f"\nwrote {path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
