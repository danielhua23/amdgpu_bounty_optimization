#!/bin/bash

# ============================================
# Multi-Concurrency Environment Variables
# kimi fp4 Benchmark - All CONC Values
# ============================================
# This script sets environment variables for multi-concurrency testing
# Use with: ./kimi_benchmark submit <team> -isl <value> -osl <value>
#
# Usage:
#   source all_conc_var.sh
#   bash launch_vllm_server.sh  # Start server first
#   ./kimi_benchmark submit "YourTeam" -isl 8192 -osl 1024   # Only supported case; CONC=4,32,128

# ============================================
# Server Configuration
# ============================================

export MODEL="amd/Kimi-K2.5-MXFP4"
export PORT=8888
export TP=8

# ============================================
# Model Configuration
# ============================================

export MAX_MODEL_LEN=16384

# ============================================
# Testing Configuration
# ============================================

# Only ISL=8192, OSL=1024 (8k/1k) is supported; CONC=4,32,128 when using -isl 8192 -osl 1024

export RANDOM_RANGE_RATIO=1.0

# ============================================
# Team Configuration (Optional)
# ============================================

# Set your team name here (optional, can override with command line)
# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Multi-Concurrency Environment Variables Set"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO"
echo ""
echo "Next steps:"
echo "1. bash launch_vllm_server.sh  # Start server"
echo "2. Wait for server to be ready (see 'Uvicorn running...')"
echo "3. Run multi-concurrency test (only 8k/1k):"
echo "   ./kimi_benchmark submit \"YourTeam\" -isl 8192 -osl 1024"
echo "============================================"
