---
title: Documentation content pass for 0.2.x (post-merge)
status: done
area: harness
owner: null
filed: 2026-08-25
fixed_in: 22ed12b
scope: M
verify: inspect
---

The 0.2.x maturin merge changed the story (native kernels shipped inside the wheel, clustering added, "pure NumPy and SciPy" dropped) and the docs were edited piecemeal during the merge. Do one deliberate content pass:

(a) Verify README.md + rust/README.md + every docstring match 0.2.x — no leftover "optional accelerator", "pip install fastumap[accel]", "separate wheel", or "pure numpy + scipy".
(b) Add a short migration note for anyone who installed fastumap-accel: it is gone, upgrade to fastumap>=0.2.1 (same API, kernels now built in).
(c) Expand API/examples coverage — clustering, cosine, supervised, persistence, transform — which currently live only as terse README snippets.
(d) The in-repo blog draft docs/blog/numba-free-umap.md still frames fastumap as pure-numpy; update it or note it belongs on the personal blog (Jorge's stated destination).

Done when: a reader following the docs on a fresh 0.2.1 install hits no claim contradicted by the code, and a fastumap-accel migration note exists (INSPECTED).