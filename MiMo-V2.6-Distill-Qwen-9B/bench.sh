#!/usr/bin/env bash
# Default competition load matrix, or BENCH_PROFILE=smoke for a short check.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
PROFILE="${BENCH_PROFILE:-matrix}"
case "$PROFILE" in
 smoke) INPUTS=(512); CONCURRENCIES=(1 4); OLEN=128 ;;
 matrix) INPUTS=(4096 16384); CONCURRENCIES=(1 8 64); OLEN=1024 ;;
 *) echo 'BENCH_PROFILE must be smoke or matrix' >&2; exit 2 ;;
esac
OUT="bench/$PROFILE-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p bench
mkdir "$OUT"
echo "Artifacts: $OUT"
cp "$RUN_RESULTS_DIR/runtime-config.json" "$OUT/"
export BENCH_OUT="$OUT" BENCH_PROFILE="$PROFILE"
python - <<'PY'
import json,os
from pathlib import Path
Path(os.environ['BENCH_OUT'],'protocol.json').write_text(json.dumps({
 'profile':os.environ['BENCH_PROFILE'],'dataset':'random-ids','tokenize_prompt':True,
 'seed':1,'flush_prefix_cache_each_case':True,'rounds':2,
 'requests':'max(8, concurrency*2)','official_scnet_validation':False,'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned')},indent=2))
PY
FAILED=0
for ILEN in "${INPUTS[@]}"; do
 for C in "${CONCURRENCIES[@]}"; do
  N=$((C*2 > 8 ? C*2 : 8))
  for ROUND in 1 2; do
   TAG="round${ROUND}_${ILEN}_${OLEN}_c${C}"
   curl --noproxy '*' -fsS --max-time 4 "http://$HOST:$PORT/health" >/dev/null
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m sglang.benchmark.serving \
   --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL_PATH" \
   --served-model-name "$SERVED_NAME" --tokenizer "$MODEL_PATH" --dataset-name random-ids --tokenize-prompt --flush-cache \
   --random-input-len "$ILEN" --random-output-len "$OLEN" --random-range-ratio 1 \
   --max-concurrency "$C" --num-prompts "$N" --seed 1 \
   --output-file "$OUT/round${ROUND}.jsonl" > "$OUT/$TAG.log" 2>&1 \
   || { echo "FAIL $TAG"; FAILED=1; }
  done
 done
done
python scripts/audit_bench.py "$OUT" --profile "$PROFILE" || FAILED=1
echo "$OUT"
exit "$FAILED"
