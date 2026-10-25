#!/usr/bin/env bash
# Diagnostic by default. The full matrix is local pre-validation, not SCNet evidence.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
python scripts/prepare_bench_alias.py
BENCH_PROFILE="${BENCH_PROFILE:-matrix}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${BENCH_OUTPUT_DIR:-bench/$BENCH_PROFILE-$STAMP}"
mkdir -p "$OUT"
if [ "$BENCH_PROFILE" = smoke ]; then
    INPUTS=(512); CONCURRENCIES=(1 4); OUTPUT_LEN=128; IMAGE_RES="${IMAGE_RES:-336x448}"
elif [ "$BENCH_PROFILE" = matrix ]; then
    INPUTS=(4096 16384); CONCURRENCIES=(1 8 64); OUTPUT_LEN=1024; IMAGE_RES="${IMAGE_RES:-1080p}"
else
    echo 'BENCH_PROFILE must be smoke or matrix' >&2; exit 2
fi
cp env.sh "$OUT/env.sh"
cp "$RUN_RESULTS_DIR/runtime-config.json" "$OUT/runtime-config.json"
printf 'profile=%s\nimage_resolution=%s\nsource=%s self-test; not official acceptance\n' "$BENCH_PROFILE" "$IMAGE_RES" "${EXECUTION_PLATFORM:-self-owned BW100}" > "$OUT/config.txt"
FAILED=0
for ILEN in "${INPUTS[@]}"; do
    for C in "${CONCURRENCIES[@]}"; do
        N=$(( C*2 > 8 ? C*2 : 8 ))
        for ROUND in 1 2; do
            TAG="round${ROUND}_${ILEN}_${OUTPUT_LEN}_c${C}"
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m sglang.benchmark.serving --backend sglang \
                --host "$HOST" --port "$PORT" --model "$MODEL_PATH" \
                --served-model-name "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
                --flush-cache --dataset-name image --image-count 1 --image-resolution "$IMAGE_RES" \
                --random-input-len "$ILEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio 1 \
                --max-concurrency "$C" --num-prompts "$N" --seed 1 \
                --output-file "$OUT/round${ROUND}.jsonl" > "$OUT/$TAG.log" 2>&1 \
                || { echo "FAIL: $OUT/$TAG.log"; FAILED=1; }
            python scripts/check_bench_row.py "$OUT/round${ROUND}.jsonl" "$ILEN" "$OUTPUT_LEN" "$C" "$N" \
                >> "$OUT/$TAG.log" 2>&1 || { echo "INCOMPLETE: $OUT/$TAG.log"; FAILED=1; }
        done
    done
done
if [ -f "$OUT/round2.jsonl" ]; then python scripts/summarize_bench.py "$OUT"; fi
if [ "$BENCH_PROFILE" = matrix ] && [ "$FAILED" = 0 ]; then python scripts/audit_matrix.py "$OUT"; fi
printf 'Artifacts: %s\n' "$OUT"
exit "$FAILED"
