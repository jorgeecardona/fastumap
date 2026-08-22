---
title: "#18 Publish to PyPI (OIDC Trusted Publishing), independent-reimplementation notice"
status: open
area: release
owner: null
filed: 2026-08-22
scope: M
verify: probe
---

Publish to PyPI via GitLab OIDC Trusted Publishing (no stored token; scripts/pypi-publish + scripts/release, gated by CI/CD var PYPI_READY=1). Cite the UMAP paper (arXiv 1802.03426) and lmcinnes/umap; state plainly this is an independent reimplementation, not affiliated, not a drop-in replacement. MIT license.

Done when: `pip install fastumap` works, the PyPI page shows the notice, and the release pipeline is green. probe.

BLOCKED on: PyPI Trusted Publisher config (needs Jorge's PyPI login via browser; namespace jorgeecardona, project fastumap, workflow ref main, environment pypi). Until then CI runs :ci only on main.