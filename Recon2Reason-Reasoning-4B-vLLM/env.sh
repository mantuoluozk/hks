#!/usr/bin/env bash
export MODEL_ID=BAAI/Recon2Reason-Reasoning-4B
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B}"
export IMAGE=appstore.scnet.cn:5000/aihub/dcu/zack233/recon2reason-vllm:v1.0.0
export IMAGE_VERSION=v1.0.0
export FRAMEWORK=vLLM CARD=BW PRECISION=BF16 TP=1
export HOST=127.0.0.1 PORT=30123 SERVED_NAME=recon2reason-4b
export MAX_LEN="${MAX_LEN:-32768}"
export MEM_FRAC="${MEM_FRAC:-0.80}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
