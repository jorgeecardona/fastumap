---
title: macOS + Windows wheels in CI
status: open
area: production
owner: null
filed: 2026-08-25
scope: M
verify: inspect
---

The 0.2.1 merge ships Linux x86_64 + aarch64 abi3 wheels only (.gitlab-ci.yml ci job). macOS and Windows users must build from the sdist, which needs a Rust toolchain — a regression from the pre-merge pure-Python install that ran anywhere with no compiler.

Add macOS (x86_64 + arm64) and Windows (x86_64) abi3 wheels. GitLab's Linux runners can't cross-build these cleanly; likely needs a GitHub Actions mirror or cibuildwheel on hosted macOS/Windows runners. abi3 means one wheel per platform covers 3.11+.

Done when: PyPI shows macosx_* and win_amd64 abi3 wheels for the next release, and pip install fastumap needs no compiler on those platforms (INSPECTED on the PyPI files list).