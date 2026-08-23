# fastumap-accel

Optional native accelerator for [fastumap](../): a Rust reimplementation of the SGD layout
kernel (`optimize_layout`), built as a Python extension with PyO3 + maturin.

The base `fastumap` package is pure numpy + scipy and stays that way — it works everywhere
with no build toolchain. When `fastumap_accel` is importable, `fastumap` uses it for the
optimisation loop; otherwise it falls back to the numpy path. Same public API, same results
within the usual rotation/reflection tolerance.

## Why Rust

- **Concurrency**: the loop runs with the GIL released (`py.allow_threads`), so concurrent
  `asyncio.to_thread` calls run in parallel; the borrow checker guarantees no data races.
- **Lambda/ECS**: a tiny `.so` (no 172 MB llvmlite), imports in milliseconds (no runtime
  JIT → fast cold start), single-threaded under fractional vCPU. maturin builds x86_64 and
  aarch64 (Graviton) manylinux wheels.
- **Fidelity + speed**: does umap-learn's true *in-place* (Hogwild) edge walk, which the
  pure-numpy path only approximates with a per-epoch snapshot.

## Build

```bash
maturin develop --release          # into the active venv (dev)
maturin build --release            # wheel for the current platform
# maturin build --release --zig --target aarch64-unknown-linux-gnu   # Graviton
```

## Status

Scaffold — not yet built or benchmarked. Next: first `maturin develop` (may need minor
PyO3 0.22 API tweaks), a determinism + quality test against the numpy path, wire the
optional import into `fastumap/layout.py`, then the CI wheel matrix.
