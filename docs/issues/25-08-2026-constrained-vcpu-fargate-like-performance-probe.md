---
title: Constrained-vCPU (Fargate-like) performance probe
status: open
area: production
owner: null
filed: 2026-08-25
scope: M
verify: probe
---

The README claims "~10s at 1M multi-core, minutes on 0.5–2 vCPU" but nothing validates the constrained-vCPU behavior. Jorge's live Fargate ECS numbers are the ground truth for the SGD path (250/500/1000/5000 pts = 7.6/28/83/270s, one case >1000s); clustering under a vCPU cap is entirely untested. bench/fargate.sh exists but is a single 0.5-vCPU import+fit check.

Add a reproducible constrained-CPU probe (cgroup / docker --cpus) that records fit AND cluster wall-clock at several n under 0.5, 1, and 2 vCPU.

Done when: a script emits a table of times under each cap whose shape matches Jorge's observed Fargate curve (PROBED).