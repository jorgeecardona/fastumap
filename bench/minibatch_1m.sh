#!/bin/sh
# The roadmap's mini-batch bar, checked under a real CPU cap:
#   1M x 128 into ~64 groups in under 10 s on 2 vCPU, inertia within ~5% of full Lloyd.
#
#   sh bench/minibatch_1m.sh              # 2 vCPU on SIFT1M (needs the cached descriptors)
#   CAPS="1 2 4" sh bench/minibatch_1m.sh
#
# Like bench/constrained.sh this installs a wheel built from this tree, so the container needs no
# Rust toolchain and measures the artifact PyPI ships. `--cpus` is a CFS quota, the same mechanism
# Fargate meters with. Memory is generous because 1M x 128 as float64 is ~1 GB before any copy.
set -e
CAPS="${CAPS:-2}"
MEM="${MEM:-6g}"
N="${N:-1000000}"
K="${K:-64}"
IMAGE="${IMAGE:-python:3.13-slim-trixie}"
DATASET="${DATASET:---sift}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/docs/bench/minibatch-1m-$(date +%Y-%m-%d).md}"

WHEEL=$(ls "$ROOT"/dist/*.whl 2>/dev/null | head -1 || true)
[ -n "$WHEEL" ] || { echo "no wheel in dist/ — run: make build" >&2; exit 1; }
WHEEL_NAME=$(basename "$WHEEL")

ROWS=$(mktemp)
trap 'rm -f "$ROWS"' EXIT
for CAP in $CAPS; do
    echo "== --cpus=$CAP --memory=$MEM ==" >&2
    docker run --rm --cpus="$CAP" --memory="$MEM" \
        -v "$ROOT":/w -w /w "$IMAGE" sh -c \
        "pip install -q --disable-pip-version-check /w/dist/$WHEEL_NAME >/dev/null && \
         python /w/bench/minibatch_1m.py --n $N --k $K --cap $CAP $DATASET" >>"$ROWS"
done

mkdir -p "$(dirname "$OUT")"
{
    echo "---"
    echo "title: fastumap mini-batch k-means at 1M"
    echo "date: $(date +%Y-%m-%d)"
    echo "generated_by: bench/minibatch_1m.sh"
    echo "---"
    echo
    echo "# Mini-batch k-means at 1M — $(date +%Y-%m-%d)"
    echo
    echo "The bar: **1M x 128 into ~$K groups in under 10 s on 2 vCPU, inertia within ~5% of full"
    echo "Lloyd.** Regenerate with:"
    echo
    echo '```'
    echo "CAPS=\"$CAPS\" N=$N K=$K sh bench/minibatch_1m.sh"
    echo '```'
    echo
    echo "Each row is one \`docker run --cpus=<cap> --memory=$MEM\` on \`$IMAGE\` with the wheel"
    echo "\`$WHEEL_NAME\`. Data: SIFT1M descriptors (Jegou et al.), the canonical 1M x 128 set."
    echo
    echo "| vCPU cap | implementation | wall time | inertia | groups |"
    echo "|---|---|---:|---:|---:|"
    cat "$ROWS"
    echo
    echo "## Host"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| host CPU | $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown) |"
    echo "| host cores | $(nproc 2>/dev/null || echo unknown) |"
    echo "| host load at start | $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown) |"
    echo
    echo "A local \`--cpus\` cap is the same mechanism Fargate uses but not the same hardware; see"
    echo "the constrained-vCPU probe report for how far apart those are in absolute terms."
} >"$OUT"
echo "" >&2
echo "wrote $OUT" >&2
