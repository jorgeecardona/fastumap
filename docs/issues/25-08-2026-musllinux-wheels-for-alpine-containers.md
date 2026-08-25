---
title: musllinux wheels for Alpine containers
status: done
area: production
owner: null
filed: 2026-08-25
fixed_in: 9293cb1
scope: S
verify: inspect
---

Only manylinux wheels are built (.gitlab-ci.yml). Alpine-based images (musl libc, common for small containers) match no wheel and fall back to sdist + Rust. Add musllinux_1_2 x86_64 (and aarch64) abi3 wheels via maturin targeting *-unknown-linux-musl, or a musl builder image.

Done when: PyPI shows musllinux wheels and pip install fastumap works on python:3.12-alpine with no compiler (INSPECTED).