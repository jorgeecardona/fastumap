# fastumap-accel

The optional native accelerator for [fastumap](https://gitlab.com/jorgeecardona/fastumap): a
Rust reimplementation of the SGD layout kernel (`optimize_layout`), built as a Python extension
with PyO3 and maturin.

It is not usually installed directly — `pip install fastumap[accel]` pulls it in. The base
fastumap package remains pure NumPy and SciPy and installs without a build toolchain; when this
wheel is present, fastumap uses it for the optimisation loop and otherwise falls back to the
NumPy path. `fastumap.accelerator_active()` reports which path is in use.

## Rationale

- **Concurrency.** The loop runs with the GIL released, so concurrent `asyncio.to_thread`
  calls execute in parallel, and Rust's borrow checker precludes data races.
- **Fast cold start.** The extension is a small `.so` that imports in milliseconds — no runtime
  JIT and no 172 MB of llvmlite — so it preserves fastumap's fast import even on a
  fractional-vCPU host. Prebuilt wheels cover x86_64 and aarch64/Graviton.
- **Fidelity.** It performs umap-learn's true in-place (Hogwild) edge walk, which the NumPy path
  only approximates with a per-epoch snapshot — yielding both higher speed and slightly higher
  overlap. Consequently it produces a different (higher-quality) layout than the NumPy fallback
  for the same seed; see fastumap's Reproducibility section.

## Benchmarks

Release build, single-threaded, `optimize_layout` only, against the NumPy path on identical
graph and initialisation:

| n | Rust | NumPy | speedup | overlap@15 (Rust / NumPy) |
|---|---:|---:|---:|---:|
| 1000 | 2.40 s | 4.50 s | 1.9× | 0.244 / 0.233 |
| 2000 | 5.03 s | 8.57 s | 1.7× | 0.151 / 0.140 |

Output is deterministic run to run. These figures were taken on a swap-loaded machine; an
unloaded host shows a wider margin. The distribution is a single **abi3** wheel (`cp311-abi3`)
compatible with Python 3.11 and later — one build, no per-version rebuild.

## Building

```bash
maturin develop --release          # build into the active virtualenv (development)
maturin build --release            # wheel for the current platform
maturin sdist                      # source distribution
# maturin build --release --zig --target aarch64-unknown-linux-gnu   # aarch64 / Graviton
```

MIT licensed, as fastumap.
