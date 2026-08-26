---
title: Broaden the x86_64 wheel glibc floor (manylinux2014)
status: in-progress
area: production
owner: bronze-sentinel-79061
filed: 2026-08-25
scope: S
verify: inspect
---

The x86_64 wheel is tagged manylinux_2_34 (MEASURED: fastumap-0.2.1-cp311-abi3-manylinux_2_34_x86_64.whl, produced by the debian-bookworm builder whose glibc is 2.36). That excludes hosts with glibc < 2.34 — Ubuntu 20.04 (2.31), Amazon Linux 2, older CentOS — which then fall back to the sdist + Rust. The aarch64 wheel is already manylinux2014 (2_17) because the zig cross-build targets an older glibc.

Build the x86_64 wheel against an older glibc too: a manylinux2014 builder image, maturin --compatibility manylinux2014, or a zig-target x86_64 build mirroring the aarch64 path.

Done when: the x86_64 wheel on PyPI carries a manylinux_2_17 / manylinux2014 tag (INSPECTED on the PyPI files list).