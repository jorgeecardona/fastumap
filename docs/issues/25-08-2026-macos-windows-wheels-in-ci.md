---
title: macOS + Windows wheels in CI
status: in-progress
area: production
owner: bronze-sentinel-79061
filed: 2026-08-25
scope: M
verify: inspect
---

The 0.2.1 merge ships Linux x86_64 + aarch64 abi3 wheels only (.gitlab-ci.yml ci job). macOS and Windows users must build from the sdist, which needs a Rust toolchain — a regression from the pre-merge pure-Python install that ran anywhere with no compiler.

Add macOS (x86_64 + arm64) and Windows (x86_64) abi3 wheels. GitLab's Linux runners can't cross-build these cleanly; likely needs a GitHub Actions mirror or cibuildwheel on hosted macOS/Windows runners. abi3 means one wheel per platform covers 3.11+.

Done when: PyPI shows macosx_* and win_amd64 abi3 wheels for the next release, and pip install fastumap needs no compiler on those platforms (INSPECTED on the PyPI files list).
## Update 2026-08-26 — Windows done; macOS is plan-gated, not a technical problem

The issue guessed "GitLab's Linux runners can't cross-build these cleanly; likely needs a GitHub
Actions mirror". Half right. VERIFIED against current GitLab docs, and the two halves differ:

WINDOWS — available on the Free plan, cost factor 1 (the same rate as Linux small). Implemented as
the `wheels-windows` job: tag `saas-windows-medium-amd64`, no GitHub mirror needed. Notes that cost
time to rediscover: it is not a Docker executor (an `image:` key is rejected), the shell is
PowerShell, Python 3.13 is preinstalled but Rust is NOT (rustup-init runs in the job). It publishes
its own wheel with `twine --skip-existing` rather than handing an artifact to `ci` — PyPI accepts
extra files on an existing version, so the two publishes do not need to be coordinated.

MACOS — blocked, and NOT by anything in this repo. GitLab's hosted macOS runners are
`Tier: Premium, Ultimate`. On a Free namespace they cannot run at all; this is a plan gate, so
verifying a payment method does not unlock them. MEASURED on this project: the runners are visible
via the API (`saas-macos-medium-m1`, online) but visibility is not entitlement.

Two real options, both Jorge's call:
  1. **GitLab for Open Source program.** fastumap is MIT and the project is public, so it looks
     eligible. Grants Ultimate, which includes the macOS runners. Free, but an application.
  2. **A GitHub Actions mirror** for the macOS wheel only, publishing to the same PyPI project via
     a second trusted publisher. Works today, costs nothing, adds a second CI system to maintain.
  (A third, not recommended: pay for Premium at ~$29/user/month for one wheel.)

Cost note if option 1 lands: macOS runners bill at **cost factor 6** (`saas-macos-medium-m1`) or 12
(`saas-macos-large-m2pro`), against Windows' 1. A 10-minute macOS build spends 60 compute minutes,
so it belongs on a manual or tag-only trigger, not on every push to main.

This issue stays OPEN until PyPI shows a `macosx_*` wheel.

VERIFIED 2026-08-26, pipeline #92 (sha 870d71c): `wheels-windows` went green on its first run and
published. INSPECTED on the PyPI files list for 0.2.17:

    fastumap-0.2.17-cp311-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    fastumap-0.2.17-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    fastumap-0.2.17-cp311-abi3-musllinux_1_2_aarch64.whl
    fastumap-0.2.17-cp311-abi3-musllinux_1_2_x86_64.whl
    fastumap-0.2.17-cp311-abi3-win_amd64.whl      <- new
    fastumap-0.2.17.tar.gz

So `pip install fastumap` on Windows x86_64 no longer needs a compiler. Only the macOS half is
left, and it is an account-plan decision, not code.
