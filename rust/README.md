# fastumap-accel

The optional native speedup for [fastumap](https://gitlab.com/jorgeecardona/fastumap). It's
one piece of fastumap rewritten in Rust — the SGD layout loop (`optimize_layout`) — built as a
small Python extension with PyO3 + maturin.

You don't really install this on its own; `pip install fastumap[accel]` pulls it in. The base
fastumap package stays pure numpy + scipy and runs everywhere with no build toolchain, and when
this wheel happens to be present fastumap uses it for the optimisation loop, otherwise it falls
back to the numpy path. Same API, same results within the usual rotation/reflection tolerance.

## Why it's worth having

- It runs the loop with the GIL released, so concurrent `asyncio.to_thread` calls actually run
  in parallel, and Rust's borrow checker rules out data races.
- It's a tiny `.so` that imports in milliseconds — no runtime JIT, no 172 MB of llvmlite — so
  it keeps fastumap's fast cold start even on a fractional-vCPU box. Prebuilt wheels cover
  x86_64 and aarch64 (Graviton).
- It does umap-learn's true *in-place* (Hogwild) edge walk, which the pure-numpy path only
  approximates with a per-epoch snapshot — so it comes out a little faster *and* slightly
  higher quality.

Measured (release build, single thread, `optimize_layout` only, against the numpy path on the
same graph and init):

| n | rust | numpy | speedup | rust overlap@15 | numpy overlap@15 |
|---|---:|---:|---:|---:|---:|
| 1000 | 2.40 s | 4.50 s | **1.9×** | **0.244** | 0.233 |
| 2000 | 5.03 s | 8.57 s | **1.7×** | **0.151** | 0.140 |

Deterministic run to run. (These were taken on a swap-loaded machine, so a clean one should
show a wider gap.) It's one **abi3** wheel (`cp311-abi3`) that works on Python 3.11 and every
later version — one build, no per-version rebuild.

## Building it yourself

```bash
maturin develop --release          # into the active venv (dev)
maturin build --release            # wheel for the current platform
# maturin build --release --zig --target aarch64-unknown-linux-gnu   # Graviton
```

MIT licensed, same as fastumap.
