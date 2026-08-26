#!/bin/sh
# Constrained-vCPU probe: what fastumap costs on a Fargate-sized container.
#
#   sh bench/constrained.sh                       # 0.5, 1 and 2 vCPU (or: make constrained)
#   CAPS="0.5 1" SIZES=250,500 sh bench/constrained.sh
#
# Runs bench/constrained.py under `docker --cpus=<cap>` once per cap and writes one dated markdown
# table to docs/bench/. `--cpus` sets a CFS quota, which is how Fargate's cgroup meters CPU, so the
# throttling is the same mechanism production uses rather than an approximation of it.
#
# It installs a WHEEL built from this working tree, not `pip install .`: the container then needs
# no Rust toolchain, the install takes seconds instead of minutes, and what is measured is the
# same artifact users get from PyPI. Build one first with:
#
#   maturin build --release -m rust/Cargo.toml -o dist       # or: make build
set -e
CAPS="${CAPS:-0.5 1 2}"
SIZES="${SIZES:-250,500,1000,5000}"
DIM="${DIM:-1024}"
MEM="${MEM:-2g}"
IMAGE="${IMAGE:-python:3.13-slim-trixie}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/docs/bench/constrained-$(date +%Y-%m-%d).md}"

# Jorge's live AWS Fargate ECS numbers for the SGD path, at 0.5 vCPU: "n:seconds". These are the
# ground truth the probe is checked against, so drift shows up as a row moving away from them.
PRODUCTION="250:7.6 500:28 1000:83 5000:270"

WHEEL=$(ls "$ROOT"/dist/*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHEEL" ]; then
    echo "no wheel in dist/ — build one first:" >&2
    echo "  maturin build --release -m rust/Cargo.toml -o dist" >&2
    exit 1
fi
WHEEL_NAME=$(basename "$WHEEL")
echo "wheel: $WHEEL_NAME" >&2

ROWS=$(mktemp)
CMP=$(mktemp)
trap 'rm -f "$ROWS" "$CMP"' EXIT

for CAP in $CAPS; do
    echo "== --cpus=$CAP --memory=$MEM on $IMAGE ==" >&2
    docker run --rm --cpus="$CAP" --memory="$MEM" \
        -v "$ROOT":/w -w /w -e "CPUS=$CAP" "$IMAGE" sh -c \
        "pip install -q --disable-pip-version-check /w/dist/$WHEEL_NAME >/dev/null && \
         python /w/bench/constrained.py --sizes $SIZES --dim $DIM" >>"$ROWS"
done

# Line up this run's 0.5-vCPU fit times against the production numbers, size by size.
for PAIR in $PRODUCTION; do
    N=${PAIR%%:*}
    PROD=${PAIR##*:}
    HERE=$(awk -F'|' -v want="$N" '
        { gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/^ +| +$/, "", $5) }
        $2 == "0.5" && $3 == want { print $5; exit }
    ' "$ROWS")
    printf '| %s | %s s | %s |\n' "$N" "$PROD" "${HERE:-n/a}" >>"$CMP"
done

mkdir -p "$(dirname "$OUT")"
{
    echo "---"
    echo "title: fastumap constrained-vCPU probe"
    echo "date: $(date +%Y-%m-%d)"
    echo "generated_by: bench/constrained.sh"
    echo "---"
    echo
    echo "# Constrained-vCPU probe — $(date +%Y-%m-%d)"
    echo
    echo "Generated, not hand-written. Regenerate with:"
    echo
    echo '```'
    echo "CAPS=\"$CAPS\" SIZES=$SIZES DIM=$DIM sh bench/constrained.sh"
    echo '```'
    echo
    echo "Each row is one \`docker run --cpus=<cap> --memory=$MEM\` on \`$IMAGE\`, installing the"
    echo "wheel \`$WHEEL_NAME\` built from this tree. \`--cpus\` sets a CFS quota, the same mechanism"
    echo "AWS Fargate's cgroup uses to meter CPU."
    echo
    echo "\`fit\` is \`umap_project(x, 2)\`; \`cluster\` is \`cluster(x, method=\"kmeans\", n_clusters=8)\`"
    echo "on the same input. Clustering never computes a layout first, which is why it is timed"
    echo "separately rather than folded into the fit."
    echo
    echo "| vCPU cap | n | dims | fit | cluster |"
    echo "|---|---:|---:|---:|---:|"
    cat "$ROWS"
    echo
    echo "## Against production"
    echo
    echo "The reason this probe exists. The live AWS Fargate ECS numbers for the SGD path are the"
    echo "ground truth it is checked against:"
    echo
    echo "| n | Fargate (observed, 0.5 vCPU) | this run (0.5 vCPU) |"
    echo "|---|---:|---:|"
    cat "$CMP"
    echo
    echo "A local \`--cpus\` cap reproduces Fargate's *mechanism* (a CFS quota) but not its"
    echo "*hardware*: half a core of the host CPU below is far more compute than half a shared"
    echo "Fargate vCPU, and the container still sees every host core, so BLAS and rayon spread the"
    echo "quota over many threads instead of one. Expect this probe to be faster in absolute terms"
    echo "and flatter at small n, where fixed startup cost dominates. What it is good for is"
    echo "**drift**: the same command on the same host, run again after a change."
    echo
    echo "## Host"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| host CPU | $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown) |"
    echo "| host cores | $(nproc 2>/dev/null || echo unknown) |"
    echo "| host load at start | $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown) |"
    echo "| image | $IMAGE |"
    echo "| memory cap | $MEM |"
    echo
    echo "The host load matters: \`--cpus\` bounds the container's CPU-time but cannot reserve it,"
    echo "so a busy host inflates every row here. Compare runs whose host rows match."
} >"$OUT"

echo "" >&2
echo "wrote $OUT" >&2
