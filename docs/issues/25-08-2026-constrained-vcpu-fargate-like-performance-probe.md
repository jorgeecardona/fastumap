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
## Update 2026-08-26 — probe built and measured; the shape does NOT match production

DONE: `make constrained` (bench/constrained.sh + bench/constrained.py) runs `umap_project` AND
`cluster` at n=250/500/1000/5000 x 1024 under `docker --cpus=0.5`, `1` and `2` and writes a dated
table to docs/bench/. It installs a wheel built from the tree, so the container needs no Rust and
what is measured is the artifact PyPI ships. Reproducible: two consecutive runs agreed within ~10%
at every point.

MEASURED at 0.5 vCPU (docs/bench/constrained-2026-08-26.md), against the ground truth in this issue:

| n | Fargate (observed) | this probe |
|---|---:|---:|
| 250 | 7.6 s | 3.5 s |
| 500 | 28 s | 2.8 s |
| 1000 | 83 s | 5.6 s |
| 5000 | 270 s | 33.2 s |

So the done-condition ("whose shape matches Jorge's observed Fargate curve") is NOT met, and this
issue stays open for that reason. It is not just an offset — the shapes differ. Production's
250->500 step is 3.7x; the probe's is 0.87x (n=500 is *faster* than n=250, so fixed startup cost
dominates at small n). Production's 5000/250 ratio is 35x; the probe's is 11x.

INFERRED causes, in order of likely size:
1. Hardware, not mechanism. `--cpus=0.5` reproduces Fargate's CFS quota exactly, but half a core
   of an i9-9880H is far more compute than half a shared Fargate vCPU.
2. Thread spread. The container still reports all 16 host cores (VERIFIED: the probe prints
   `cpu_count=16` under `--cpus=2`), so BLAS and rayon spread the quota across many threads. A
   Fargate micro-VM reports its small real core count.
3. Possibly a different code path entirely — see the open question below.

OPEN QUESTION, and it decides whether this is expected or a regression: which fastumap version
produced 7.6/28/83/270? If that run predates 0.2.1 it used the numpy SGD fallback, and comparing
it against a kernel-enabled probe compares two different implementations. Needs Jorge.

Also measured for the first time: clustering under a cap. At 0.5 vCPU, n=5000x1024, `cluster`
takes 20.7 s against `fit`'s 33.2 s — about two-thirds the cost of a full projection, not the
rounding error it is usually assumed to be.

## Update 2026-08-26 (2) — the open question is ANSWERED, and it was not the code path

The previous update guessed the gap might be a different code path: if the Fargate run predated
0.2.1 it used the pure-numpy SGD, not the Rust kernel. TESTED directly with a new
`bench/constrained.py --numpy-path` flag, which clears `layout._accel_optimize` so the SGD runs
through numpy — same data, same seed, same container. `accelerator_active` printed False,
confirming the fallback really ran.

MEASURED, all at `docker --cpus=0.5`, 1024 dims:

| n | Fargate (observed) | probe, Rust kernel | probe, numpy path |
|---|---:|---:|---:|
| 250 | 7.6 s | 3.5 s | 4.5 s |
| 500 | 28 s | 2.8 s | 4.7 s |
| 1000 | 83 s | 5.6 s | 8.7 s |
| 5000 | 270 s | 33.2 s | 47.4 s |

DISPROVEN. The numpy path costs about 1.4x the kernel, not the 5-10x the gap needs, and the shape
is unchanged: 250->500 is still flat (4.5 -> 4.7 s) where production's is 3.7x. So Jorge no longer
needs to dig up which version produced those numbers — the answer does not change the conclusion.

What is left is the environment, hypotheses 1 and 2 from the previous update: half a core of an
i9-9880H is far more compute than half a shared Fargate vCPU, and the container still reports all
16 host cores so BLAS and rayon spread the quota across threads a Fargate micro-VM does not have.

CONSEQUENCE for this issue's done-condition. "A table whose shape matches Jorge's observed Fargate
curve" is not reachable from this host by any code change — matching Fargate's absolute numbers
needs Fargate. The probe that exists is reproducible (two runs within ~10% at every point) and
self-checking (it prints its own numbers beside production's on every run), which makes it a good
DRIFT detector on a fixed host. Whether that is enough to call this done, or whether it should be
re-scoped to "run the probe ON Fargate and compare", is Jorge's call — so it stays open rather than
being closed against a bar it does not clear.
