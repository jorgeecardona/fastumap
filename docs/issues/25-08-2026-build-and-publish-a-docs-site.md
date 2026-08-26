---
title: Build and publish a docs site
status: in-progress
area: harness
owner: bronze-sentinel-79061
filed: 2026-08-25
scope: M
verify: inspect
---

pyproject declares a `docs` optional-dependency group but no site is built or published; all documentation lives in README.md + rust/README.md. The deeper material (reproducibility, guarantees, the full API, the accelerator internals) would be more discoverable as a browsable site.

A `pages` job existed in the initial commit but was DROPPED in 68e609e ("...drop pages"), so there is no Pages site today. This restores it AND makes it genuinely useful — not a bare generated index but a navigable site (getting started, API, clustering, reproducibility/guarantees, the accelerator internals), styled and with working cross-links.

Wire a docs build (mkdocs or similar, using the existing docs extra) into CI and publish to GitLab Pages on main.

Done when: a CI `pages` job builds the docs and publishes a navigable, styled site to Pages on main (INSPECTED at the Pages URL).
## Update 2026-08-26 — built and deployed, blocked on ONE project setting

DONE (landed 2f21179 + ade75d3): a `pages` job builds an 11-page MkDocs Material site and CI
pipeline #87 deployed it. VERIFIED by unzipping the artifact GitLab actually published, not from a
local build: index, projection, clustering, server, performance, large-inputs, accelerator,
benchmarks, guarantees, api, contributing — every nav cross-link resolves, 31 generated signatures
on the API page. The job installs only the `docs` extra (`--no-install-project` / `--no-sync`), so
it needs no Rust toolchain and builds in ~2 s.

Every prose page pulls its text out of README.md with pymdownx snippet sections, so the site
cannot drift from the README. Add a page by marking a README section and adding it to `nav`.

BLOCKED, and it is not a code change: the project is `visibility: public` but
`pages_access_level: private`, so https://fastumap-942a24.gitlab.io/ redirects to a GitLab login.
An API call to flip it was denied — it is an outward-facing visibility change, so it is Jorge's.

Remaining: Settings > General > Visibility > Pages > "Everyone". Then load the URL and close this.
Nothing is exposed that is not already public — every page is derived from the public README and
public source docstrings.

NOTE: the site is served from the unique domain https://fastumap-942a24.gitlab.io/, NOT the
path-based https://jorgeecardona.gitlab.io/fastumap/ (which 403s).
