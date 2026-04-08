#!/usr/bin/env bash

# ============================================
# vLLM Server Launch Script for kimik2.5 fp4
# ============================================
# This script launches a vLLM server with optimized settings
# for AMD MI355X GPUs running FP4 models
#
# Required Environment Variables:
#   MODEL: Model path (e.g., moonshotai/Kimi-K2.5)
#   PORT: Server port (default: 8888)
#   TP: Tensor parallel size (e.g., 4, 8)
#   MAX_MODEL_LEN: Maximum model length (default: 16384)
#
# Optional Environment Variables:
#   GPU_MEMORY_UTIL: GPU memory utilization (default: 0.95)
#   BLOCK_SIZE: Block size for paged attention (default: 64)
#   SERVER_LOG: Path to server log file (auto-generated if not set)

# ============================================
# Validate Required Environment Variables
# ============================================

if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL environment variable is required"
    echo "Example: export MODEL=moonshotai/Kimi-K2.5"
    exit 1
fi

if [[ -z "$PORT" ]]; then
    echo "WARNING: PORT not set, using default: 8888"
    PORT=8888
fi

if [[ -z "$TP" ]]; then
    echo "ERROR: TP (tensor parallel size) environment variable is required"
    echo "Example: export TP=8"
    exit 1
fi

if [[ -z "$MAX_MODEL_LEN" ]]; then
    echo "WARNING: MAX_MODEL_LEN not set, using default: 16384"
    MAX_MODEL_LEN=16384
fi

# ============================================
# Ray compatibility (match InferenceX kimik2.5_fp4_mi355x.sh)
# ============================================
if [[ -n "$ROCR_VISIBLE_DEVICES" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ============================================
# Set Default Values
# ============================================

GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.95}
BLOCK_SIZE=${BLOCK_SIZE:-64}

# ============================================
# Create Server Log File
# ============================================

if [[ -z "$SERVER_LOG" ]]; then
    SERVER_LOG=$(mktemp /tmp/vllm-server-XXXXXX.log)
    echo "INFO: Server log file: $SERVER_LOG"
fi

# ============================================
# Launch vLLM Server
# ============================================

echo "============================================"
echo "Launching vLLM Server"
echo "============================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Util: $GPU_MEMORY_UTIL"
echo "Block Size: $BLOCK_SIZE"
echo "Log File: $SERVER_LOG"
echo "============================================"

echo ""
echo "⚠️  NOTE: First launch may take 20+ minutes due to JIT compilation of kernels"
echo "⚠️  Server will run in FOREGROUND. Press Ctrl+C to stop."
echo "⚠️  Server is ready when you see: 'Uvicorn running on http://0.0.0.0:$PORT'"
echo ""
echo "============================================"
echo "Starting Server..."
echo "============================================"
echo ""

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization $GPU_MEMORY_UTIL \
--max-model-len $MAX_MODEL_LEN \
--block-size=$BLOCK_SIZE \
--disable-log-requests \
--trust-remote-code \
--mm-encoder-tp-mode data
set +x
