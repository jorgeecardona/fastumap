# fastumap native kernels (`fastumap._accel`)

The compiled Rust extension built into [fastumap](https://gitlab.com/jorgeecardona/fastumap): the
SGD layout kernel (`optimize_layout`) and the k-means kernel (`kmeans`), built as a Python
extension with PyO3 and maturin and shipped **inside the fastumap wheel** — there is no separate
package. On a platform with a prebuilt wheel it is always present; from an sdist it needs a Rust
toolchain to compile, and without one fastumap falls back to its pure-NumPy path.
`fastumap.accelerator_active()` reports which path is in use.

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

Run from the repository root — maturin reads the root `pyproject.toml` (its `[tool.maturin]`
points `manifest-path` at this `rust/Cargo.toml`) and builds the whole `fastumap` wheel, Python
plus extension:

```bash
maturin develop --release          # build + install into the active virtualenv (development)
maturin build --release            # wheel for the current platform
maturin sdist                      # source distribution
# maturin build --release --zig --target aarch64-unknown-linux-gnu   # aarch64 / Graviton
```

MIT licensed, as fastumap.
