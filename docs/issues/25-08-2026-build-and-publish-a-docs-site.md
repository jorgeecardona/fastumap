---
title: Build and publish a docs site
status: open
area: harness
owner: null
filed: 2026-08-25
scope: M
verify: inspect
---

pyproject declares a `docs` optional-dependency group but no site is built or published; all documentation lives in README.md + rust/README.md. The deeper material (reproducibility, guarantees, the full API, the accelerator internals) would be more discoverable as a browsable site.

A `pages` job existed in the initial commit but was DROPPED in 68e609e ("...drop pages"), so there is no Pages site today. This restores it AND makes it genuinely useful — not a bare generated index but a navigable site (getting started, API, clustering, reproducibility/guarantees, the accelerator internals), styled and with working cross-links.

Wire a docs build (mkdocs or similar, using the existing docs extra) into CI and publish to GitLab Pages on main.

Done when: a CI `pages` job builds the docs and publishes a navigable, styled site to Pages on main (INSPECTED at the Pages URL).