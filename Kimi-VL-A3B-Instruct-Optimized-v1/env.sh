#!/usr/bin/env bash
export MODEL_PATH="${MODEL_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/weights}"
export HKS_PROFILE="${HKS_PROFILE:-optimized}"
export PORT=30004
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
