---
title: "#10 Determinism test across processes (bit-identical bytes)"
status: open
area: production
owner: null
filed: 2026-08-22
scope: S
verify: test
---

Same input and seed must produce bit-identical output across processes, not just within one. Two sources of drift: ARPACK's start vector and the negative sampler.

Done when: two subprocesses produce identical bytes for the same input, on Linux and macOS in CI.

MEASURED: tests/test_determinism.py spawns two subprocesses and asserts identical emb.tobytes(); passes on Linux. ARPACK v0 pinned (src/fastumap/_scipy.py normalized_adjacency_eigenvectors) and negative sampler seeded via np.random.default_rng(random_state). macOS CI leg still to confirm.