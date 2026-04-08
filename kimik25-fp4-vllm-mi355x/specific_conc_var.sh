#!/bin/bash

# ============================================
# Single Configuration Environment Variables
# kimi fp4 Benchmark - Specific CONC Value
# ============================================
# This script sets environment variables for single configuration testing
# Use with: ./kimi_benchmark submit <team>
#
# Usage:
#   source specific_conc_var.sh
#   bash launch_vllm_server.sh  # Start server first
#   ./kimi_benchmark submit "YourTeam"

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
# Testing Configuration - Single Configuration
# ============================================

# Input/Output Sequence Lengths (only 8192/1024 supported)
export ISL=8192
export OSL=1024

# Concurrency Level: 4, 32, 128
export CONC=4

# Random Range Ratio
export RANDOM_RANGE_RATIO=1.0

export NUM_PROMPTS=$(( CONC * 10 ))

# Result Filename
export RESULT_FILENAME="result_isl${ISL}_osl${OSL}_conc${CONC}"

# ============================================
# Team Configuration (Optional)
# ============================================

# Set your team name here (optional, can override with command line)
# export TEAM_NAME_ENV="YourTeam"

# ============================================
# Summary
# ============================================

echo "============================================"
echo "Single Configuration Environment Variables Set"
echo "============================================"
echo "MODEL:        $MODEL"
echo "PORT:         $PORT"
echo "TP:           $TP"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo ""
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "CONC:         $CONC"
echo "NUM_PROMPTS:  $NUM_PROMPTS"
echo "RESULT_FILE:  $RESULT_FILENAME.json"
echo ""
echo "Next steps:"
echo "1. bash launch_vllm_server.sh  # Start server"
echo "2. Wait for server to be ready (see 'Uvicorn running...')"
echo "3. Run test: ./kimi_benchmark submit \"YourTeam\""
echo "============================================"