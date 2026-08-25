"""Real-data clustering benchmark on SIFT1M (1,000,000 x 128 SIFT descriptors).

SIFT1M is the canonical ANN / clustering benchmark (Jégou et al.; what faiss benchmarks on),
so "1M x 128" here is a real distribution, not synthetic blobs. Downloads + caches the dataset
under bench/data/ (gitignored), then times fastumap's clustering and reports the k-means
objective (inertia) so speed is never quoted without a quality number beside it.

    uv run python bench/cluster_bench.py [--n 1000000] [--k 64] [--iter 20]
"""

from __future__ import annotations

import argparse
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
# ftp.irisa.fr is the primary host; these HTTPS mirrors carry the same sift.tar.gz.
_URLS = [
    "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift.tar.gz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
]


def _download_sift() -> Path:
    _DATA.mkdir(exist_ok=True)
    base = _DATA / "sift_base.fvecs"
    if base.exists():
        return base
    tar = _DATA / "sift.tar.gz"
    if not tar.exists():
        last: Exception | None = None
        for url in _URLS:
            try:
                print(f"downloading {url} ...")
                urllib.request.urlretrieve(url, tar)
                break
            except Exception as exc:  # try the next mirror
                last = exc
                print(f"  failed: {exc}")
        else:
            raise RuntimeError(f"could not fetch SIFT1M from any mirror: {last}")
    with tarfile.open(tar) as t:
        member = next(m for m in t.getmembers() if m.name.endswith("sift_base.fvecs"))
        member.name = Path(member.name).name  # flatten
        t.extract(member, _DATA)
    return base


def _read_fvecs(path: Path, limit: int | None) -> np.ndarray:
    """.fvecs: each record is int32 dim, then `dim` float32 values."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)[:, 1:]  # drop the per-row dim prefix
    vecs = rows.view(np.float32)
    return np.ascontiguousarray(vecs[:limit] if limit else vecs, dtype=np.float64)


def _inertia(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Sum of squared distances to each point's assigned centroid — the k-means objective."""
    total = 0.0
    for c in range(k):
        pts = x[labels == c]
        if len(pts):
            total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--iter", type=int, default=20)
    args = ap.parse_args()

    x = _read_fvecs(_download_sift(), args.n)
    print(f"SIFT1M: {x.shape[0]} x {x.shape[1]}")

    from fastumap.clustering import kmeans as np_kmeans

    try:
        import importlib

        accel = importlib.import_module("fastumap._accel")  # the compiled kernels, built in
        t = time.perf_counter()
        labels = accel.kmeans(np.ascontiguousarray(x), args.k, args.iter, 42)
        dt = time.perf_counter() - t
        print(f"accel  k-means: {dt:6.2f}s  inertia={_inertia(x, labels, args.k):.3e}")
    except ImportError:
        print("accel  k-means: (native kernel not built)")

    if args.n <= 200_000:  # the numpy path is far slower; only time it on a tractable slice
        t = time.perf_counter()
        labels, _ = np_kmeans(x, args.k, n_iter=args.iter, random_state=42)
        dt = time.perf_counter() - t
        print(f"numpy  k-means: {dt:6.2f}s  inertia={_inertia(x, labels, args.k):.3e}")


if __name__ == "__main__":
    main()
