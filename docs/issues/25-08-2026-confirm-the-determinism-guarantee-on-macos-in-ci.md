---
title: Confirm the determinism guarantee on macOS in CI
status: open
area: quality
owner: null
filed: 2026-08-25
blocked_by:
  - macos-windows-wheels-in-ci
scope: S
verify: test
---

The determinism issue (done, fixed_in the ARPACK-v0 + seeded-sampler work) notes "macOS CI leg still to confirm". tests/test_determinism.py asserts bit-identical bytes across two subprocesses but only the Linux leg runs. The README's cross-process reproducibility claim is therefore unverified on macOS, where a different BLAS/threading backend could perturb the SGD or the eigensolver.

Add a macOS CI leg (naturally piggybacks on the macOS wheels job) running test_determinism + test_projection.

Done when: the determinism test passes on macOS in CI (test result observed).