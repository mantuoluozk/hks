#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./env.sh
conditions="${ALIGN_CONDITIONS:-$PWD/data/official-20261030/formal/conditions/forced_aligner.jsonl}"
output="${OFFICIAL_OUTPUT_DIR:-$PWD/results/t03t04-$(date -u +%Y%m%dT%H%M%SZ)}"
python scripts/run_official_conditions.py --conditions "$conditions" --output "$output"
python scripts/verify_official_conditions.py --conditions "$conditions" --job "$output"
