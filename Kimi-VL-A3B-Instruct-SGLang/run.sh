#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p results/runtime
echo "$$" > results/runtime/server.pid
exec bash serve-formal.sh
