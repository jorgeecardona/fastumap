---
title: macOS + Windows wheels in CI
status: done
fixed_in: 0.2.33
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

## Update 2026-08-26 (2) — the macOS workflow is written and waiting on two human steps

Jorge asked for the macOS half "if possible". It is not possible to finish from here: it needs a
GitHub account and a PyPI settings change, and neither `gh` nor a macOS host exists in this
environment. What IS done is everything up to those two steps.

ADDED: `.github/workflows/macos-wheels.yml`. Inert on GitLab; it activates the moment the repo is
mirrored to GitHub. Structure adapted from `maturin generate-ci github --platform macos`, with two
deliberate changes: no free-threaded (3.14t) build, because fastumap ships cp311-abi3 and one wheel
per arch already covers 3.11+; and no sdist job, because GitLab publishes the sdist and two sources
uploading the same filename would collide. It builds x86_64 (`macos-15-intel`) and aarch64
(`macos-latest`) and publishes ONLY those two wheels, via OIDC trusted publishing with
`skip-existing: true`, so it composes with the GitLab publish instead of fighting it. Triggers are
`workflow_dispatch` and `v*` tags — never on every push.

REMAINING, both one-time and both human:
  1. Mirror the repo to GitHub. A push mirror from GitLab
     (Settings > Repository > Mirroring repositories) keeps it in sync with no ongoing work.
  2. Add a SECOND PyPI trusted publisher, alongside the GitLab one:
     pypi.org > fastumap > Manage > Publishing > Add a new publisher > GitHub, with
     owner=<github username>, repository=fastumap, workflow=macos-wheels.yml, environment=pypi.
     A project may have several trusted publishers; this does not disturb the GitLab publish.

The alternative path stays open and is strictly less work if it is granted: the GitLab for Open
Source program gives Ultimate, which includes the hosted macOS runners, and then the macOS build
could move into .gitlab-ci.yml beside the others and this workflow could be deleted. fastumap is
MIT and public, so it looks eligible. Cost note if that lands: macOS bills at cost factor 6, so it
belongs on a tag-only trigger, not on every push.

Still OPEN until PyPI shows a `macosx_*` wheel. The Windows half is done and verified.

## Update 2026-11-11 — macOS moves to GitLab

The cardona-oss group is now on GitLab's Open Source plan (Ultimate, until 2027-10-18), so the
hosted macOS runners are available. Added the `wheels-macos` job (`saas-macos-medium-m1`,
`macos-15-xcode-16`): builds aarch64 + x86_64 abi3 wheels on one runner, publishes with OIDC and
`--skip-existing`. Runs on a release (pyproject.toml change); manual on other main pushes. Deleted
`.github/workflows/macos-wheels.yml`, since no GitHub mirror is needed.

Still OPEN until PyPI shows a `macosx_*` wheel.

## Done 2026-11-11

INSPECTED on PyPI, 0.2.33: `macosx_11_0_arm64` and `macosx_11_0_x86_64` abi3 wheels, the arm64 one
contains `fastumap/_accel.abi3.so`. `wheels-macos` and `wheels-windows` now run after `ci`
(`needs: ci`), so a platform wheel is never published before the tests pass and the Linux files exist.
