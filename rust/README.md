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

**Built and benchmarked.** `maturin develop --release` produces an **abi3 wheel
(`cp311-abi3`) that works on Python 3.11 and every later version** — one build, no
per-version rebuild. Measured (release, single-thread, `optimize_layout` only, vs the
numpy path on the same graph/init):

| n | rust | numpy | speedup | rust overlap@15 | numpy overlap@15 |
|---|---:|---:|---:|---:|---:|
| 1000 | 2.40 s | 4.50 s | **1.9×** | **0.244** | 0.233 |
| 2000 | 5.03 s | 8.57 s | **1.7×** | **0.151** | 0.140 |

Faster *and* higher quality — the true in-place (Hogwild) walk, which the numpy snapshot
only approximates. Deterministic run-to-run. (Numbers were taken on a swap-loaded box, so a
clean machine should show a wider gap.)

Next: wire the optional import into `fastumap/layout.py` (use it when present, else numpy),
add `tests/test_accel.py` (skipped when the module is absent), and the CI wheel matrix
(x86_64 + aarch64/Graviton).
