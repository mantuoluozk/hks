#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
input="${ALIGNMENT_INPUT_ROOT:-/root/private_data/hks/models/Qwen3-ForcedAligner-0.6B/data}"
test -d "$input/examples"
test -f "$input/official-20261030/formal/conditions/forced_aligner.jsonl"
mkdir -p data
for name in examples official-20261030; do
 if [ ! -e "data/$name" ]; then ln -s "$input/$name" "data/$name"; fi
done
python scripts/run_official_conditions.py --conditions data/official-20261030/formal/conditions/forced_aligner.jsonl --output results/input-check --validate-inputs
