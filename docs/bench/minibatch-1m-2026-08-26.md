---
title: fastumap mini-batch k-means at 1M
date: 2026-08-26
generated_by: bench/minibatch_1m.sh
---

# Mini-batch k-means at 1M — 2026-08-26

The bar: **1M x 128 into ~64 groups in under 10 s on 2 vCPU, inertia within ~5% of full
Lloyd.** Regenerate with:

```
CAPS="2" N=1000000 K=64 sh bench/minibatch_1m.sh
```

Each row is one `docker run --cpus=<cap> --memory=6g` on `python:3.13-slim-trixie` with the wheel
`fastumap-0.2.16-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`. Data: SIFT1M descriptors (Jegou et al.), the canonical 1M x 128 set.

| vCPU cap | implementation | wall time | inertia | groups |
|---|---|---:|---:|---:|
| 2 | mini-batch | 5.2 s | 6.6601e+10 | 64 |
| 2 | full Lloyd | 25.3 s | 6.5935e+10 | 64 |
| 2 | ratio | 0.21x time | 1.010x inertia | |

## Host

| | |
|---|---|
| host CPU | Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz |
| host cores | 4 |
| host load at start | 4.54 4.27 4.46 |

A local `--cpus` cap is the same mechanism Fargate uses but not the same hardware; see
the constrained-vCPU probe report for how far apart those are in absolute terms.
