#!/bin/sh
# Reproduce the AWS Fargate / Lambda constraint locally so import cost and per-call time
# mean what they will in production. `--cpus` sets a CFS quota (throttles total CPU-time),
# exactly how Fargate's cgroup meters it, and `--memory` caps RAM. Defaults mirror the
# incident: 512 units = 0.5 vCPU, 2 GB, on the deployment's base image.
#
#   sh bench/fargate.sh                 # 0.5 vCPU, 2 GB (Fargate)
#   CPUS=1 MEM=3g sh bench/fargate.sh   # other shapes (e.g. a Lambda size)
set -e
CPUS="${CPUS:-0.5}"
MEM="${MEM:-2g}"
IMAGE="${IMAGE:-python:3.13-slim-trixie}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== fastumap under --cpus=$CPUS --memory=$MEM on $IMAGE =="
docker run --rm --cpus="$CPUS" --memory="$MEM" -v "$ROOT":/w -w /w "$IMAGE" sh -c '
    pip install -q --disable-pip-version-check . >/dev/null
    python - <<PY
import importlib, time
start = time.perf_counter()
importlib.import_module("fastumap")
print(f"import fastumap: {(time.perf_counter() - start) * 1000:.0f} ms")

import numpy as np
from fastumap import umap_project

for n in (1000, 5000):
    x = np.random.default_rng(0).normal(size=(n, 1024))
    start = time.perf_counter()
    umap_project(x, 2, random_state=0)
    print(f"fit n={n} x1024: {time.perf_counter() - start:.1f} s")
PY
'
