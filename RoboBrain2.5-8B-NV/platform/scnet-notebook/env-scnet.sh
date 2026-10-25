#!/usr/bin/env bash
# Source before running a model inside an SCNet Notebook.
export HKS_ROOT="${HKS_ROOT:-/root/private_data/hks}"
mountpoint -q /root/private_data || { echo 'Shared storage is not mounted' >&2; return 1; }
# Vendor env scripts reference optional unset variables. Restore caller strictness.
_hks_had_nounset=0
case "$-" in *u*) _hks_had_nounset=1; set +u ;; esac
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
if [ "$_hks_had_nounset" = 1 ]; then set -u; fi
unset _hks_had_nounset
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HF_HOME="$HKS_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$HKS_ROOT/cache/xdg"
export TRITON_CACHE_DIR="$HKS_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HKS_ROOT/cache/torchinductor"
mkdir -p "$HKS_ROOT"/{models,weights,datasets,backups,cache,platform/scnet-notebook/results}
