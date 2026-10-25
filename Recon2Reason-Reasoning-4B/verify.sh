#!/usr/bin/env bash
# Public self-evaluation. Keep inference separate from benchmark timings.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
if pgrep -f '^bash bench.sh$|^python -m sglang.benchmark.serving' >/dev/null; then
    echo 'Benchmark is active; run accuracy after it completes.' >&2
    exit 2
fi
python scripts/test_verify.py
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${EVAL_OUT:-results/public-accuracy-$STAMP}"
mkdir -p "$OUT"
for DATASET in ocrbench mmmu; do
    python scripts/verify.py --dataset "$DATASET" --output "$OUT/$DATASET" \
        --workers "${EVAL_WORKERS:-8}" --max-tokens "${EVAL_MAX_TOKENS:-2048}" \
        --limit "${EVAL_LIMIT:-0}" --model "$SERVED_NAME" --url "http://$HOST:$PORT" \
        2>&1 | tee "$OUT/$DATASET.log"
done
echo "Public self-evaluation artifacts: $OUT"
