"""Benchmark: does a native Rust UMAP (annembed_rs) beat umap-learn on import time,
and is its embedding quality good enough to skip a faithful port?

Measures (1) import time in a fresh interpreter, (2) fit time + trustworthiness.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time

PY = sys.executable


def resolve_annembed_import() -> str | None:
    """annembed installs as `annembed_rs` but may import as `annembed` or `annembed_rs`."""
    for name in ("annembed", "annembed_rs"):
        out = subprocess.run([PY, "-c", f"import {name}"], capture_output=True, text=True)
        if out.returncode == 0:
            return name
    return None


def time_import(mod: str, runs: int = 4) -> tuple[list[float] | None, str | None]:
    times: list[float] = []
    for _ in range(runs):
        code = (
            "import time,importlib;"
            "t=time.perf_counter();"
            f"importlib.import_module('{mod}');"
            "print(time.perf_counter()-t)"
        )
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        if out.returncode != 0:
            tail = (out.stderr or "").strip().splitlines()
            return None, tail[-1] if tail else "import failed"
        times.append(float(out.stdout.strip().splitlines()[-1]))
    return times, None


def run_annembed(mod: str, X, dim: int = 2):
    """Try the several call shapes annembed exposes; return an (n, dim) array."""
    import numpy as np

    ae = __import__(mod)
    api = [a for a in dir(ae) if not a.startswith("_")]
    print(f"  annembed public API: {api}")

    embed = getattr(ae, "embed", None)
    if embed is None:
        raise RuntimeError(f"no `embed` function found; API={api}")

    last: Exception | None = None
    # 1) direct numpy array
    for kwargs in ({"dim": dim}, {"dim": dim, "asked": dim}, {}):
        try:
            arr = embed(np.ascontiguousarray(X, dtype=np.float32), **kwargs)
            return np.asarray(arr)
        except Exception as e:
            last = e
    # 2) CSV-file path (README example is file-based)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        np.savetxt(f, X, delimiter=",")
        path = f.name
    for kwargs in ({"dim": dim}, {}):
        try:
            arr = embed(path, **kwargs)
            return np.asarray(arr)
        except Exception as e:
            last = e
    raise RuntimeError(f"could not call annembed.embed; last error: {last!r}")


def main() -> None:
    ann = resolve_annembed_import()
    print("annembed import name:", ann or "NOT INSTALLED")

    print("\n== import time (fresh interpreter, seconds; first run = cold disk cache) ==")
    for mod in [m for m in ("umap", ann) if m]:
        times, err = time_import(mod)
        if err:
            print(f"  {mod:12s} FAILED: {err}")
        else:
            assert times is not None
            print(
                f"  {mod:12s} min={min(times):.3f}  median={statistics.median(times):.3f}"
                f"  runs={[round(t, 3) for t in times]}"
            )

    from sklearn.datasets import load_digits
    from sklearn.manifold import trustworthiness

    X, _y = load_digits(return_X_y=True)
    print(f"\n== quality on digits {X.shape} (trustworthiness, higher=better, max 1.0) ==")

    try:
        import umap

        t = time.perf_counter()
        emb = umap.UMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
        ).fit_transform(X)
        tw = trustworthiness(X, emb, n_neighbors=15)
        print(f"  umap-learn   fit={time.perf_counter() - t:6.2f}s  trustworthiness={tw:.4f}")
    except Exception as e:
        print("  umap-learn   FAILED:", repr(e))

    if ann:
        try:
            t = time.perf_counter()
            emb2 = run_annembed(ann, X)
            dt = time.perf_counter() - t
            tw2 = trustworthiness(X, emb2, n_neighbors=15)
            print(f"  {ann:12s} fit={dt:6.2f}s  trustworthiness={tw2:.4f}  shape={emb2.shape}")
        except Exception as e:
            print(f"  {ann:12s} FAILED:", repr(e))


if __name__ == "__main__":
    main()
