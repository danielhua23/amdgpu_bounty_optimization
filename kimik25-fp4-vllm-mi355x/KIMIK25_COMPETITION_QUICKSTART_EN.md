# 🏆 KIMIK25 Competition Quick Start Guide

## 📑 Table of Contents

- [Objective](#objective)
- [Core Files](#core-files)
- [Quick Start (5 Steps)](#quick-start-5-steps)
  - [1️⃣ Prepare Working Directory (on Host Machine)](#1️⃣-prepare-working-directory-on-host-machine)
  - [2️⃣ Launch Development Container](#2️⃣-launch-development-container)
  - [3️⃣ Install Latest Editable vLLM in Container](#3️⃣-install-latest-editable-vllm-in-container)
  - [4️⃣ Example: How to Recompile After Code Modifications](#4️⃣-example-how-to-recompile-after-code-modifications)
  - [5️⃣ Test Optimization Results](#5️⃣-test-optimization-results)
- [Test Mode Comparison](#test-mode-comparison)
- [Two Testing Approaches Comparison](#two-testing-approaches-comparison)
- [Evaluation Criteria](#evaluation-criteria)
  - [Performance Metrics (Primary)](#performance-metrics-primary)
  - [Accuracy Requirements (Must Meet)](#accuracy-requirements-must-meet)
  - [Baseline Comparison 📊](#baseline-comparison-)
- [Optimization Directions](#optimization-directions)
- [Development Tips](#development-tips)
- [FAQ](#faq)
- [Recommended Workflow](#recommended-workflow)
- [Resource Links](#resource-links)

---

## Objective

Optimize inference performance for KIMIK2.5-1T FP4 model on AMD MI355X GPUs while maintaining model accuracy.

## Model Specifics
- **Model**: `amd/Kimi-K2.5-MXFP4` (fp4 quantized)
- **Framework**: Not limited to one framework; vLLM, SGLang, etc. are all acceptable
- **Features**: Uses AMD AITER optimized MoE and attention kernels

## Important Notice
- **Only one benchmark case**: **ISL=8192, OSL=1024 (8k/1k)**. Multi-conc mode only accepts `-isl 8192 -osl 1024`.
- **CONC values**: **4, 32, 128** only.

## Core Files

| File | Purpose |
|------|------|
| `amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/launch_vllm_server.sh` | Launch vLLM server |
| `amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/kimi_benchmark` | Run tests and submit results |
| `amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/all_conc_var.sh` | Multi-concurrency test environment variables |
| `amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/specific_conc_var.sh` | Single configuration test environment variables |

## Quick Start (5 Steps)

### 1️⃣ Prepare Working Directory (on Host Machine)

```bash
# Create working directory on host machine
mkdir -p ~/competition
cd ~/competition

# Clone vLLM (you will optimize based on this)
git clone https://github.com/vllm-project/vllm.git

# Clone AITER (AMD GPU operator library)
git clone --recursive https://github.com/ROCm/aiter.git

# Clone scripts repository
git clone https://github.com/danielhua23/amdgpu_bounty_optimization.git
```

### 2️⃣ Launch Development Container

**Note**: Replace `HF_TOKEN` with your Hugging Face Token.

```bash
docker run -it \
  --name vllm-dev-kimi \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /docker/huggingface/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -v ~/competition/aiter:/workspace/aiter \
  -v ~/competition/vllm:/workspace/vllm \
  -e HF_TOKEN=your_huggingface_token_here \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.15.1
```

**Mount Instructions**:
- Host `~/competition/*` → Container `/workspace/*`
- Code modifications on host machine take effect immediately in container (and vice versa)
- Test scripts are located in `/workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/` directory

### 3️⃣ Install Latest Editable vLLM in Container

> refer to https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#build-wheel-from-source

```bash
# Uninstall existing libraries in container
pip uninstall -y aiter vllm

# Install AITER
cd /workspace/aiter
python3 setup.py develop
```

Verify AITER installation:
```bash
root@mi355:/workspace# pip list | grep aiter
aiter                             0.1.7.post3.dev39+g1f5b378dc        /workspace/aiter
```

Install vLLM:
```bash
# Enter vLLM directory
cd /workspace/vllm

# Upgrade pip
pip install --upgrade pip

# Install vLLM (editable mode)
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip install "numpy<2"
# Install dependencies
pip install -r requirements/rocm.txt
# Build vLLM for MI GPU
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
python3 setup.py develop

# Verify
python -c "import vllm; print(vllm.__file__)"
# Expected output: /workspace/vllm/vllm/__init__.py
```

>note: you might meet some **error** like : ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. transformers 4.56.2 requires huggingface-hub<1.0,>=0.34.0, but you have huggingface-hub 1.1.7 which is incompatible..**Just ignore it**
### 4️⃣ Example: How to Recompile After Code Modifications

```bash
# Edit in container or host machine (VS Code)
# Example: Optimize scheduler
cd /workspace/vllm
vim vllm/engine/llm_engine.py

# If you modified Python code, no recompile needed (editable mode auto-applies)

# If you modified C++/CUDA/HIP extensions, need to recompile:
cd /workspace/vllm
rm -r ./build
rm -r ./vllm.egg-info
pip uninstall -y vllm
python3 setup.py clean
python3 setup.py develop
```

### 5️⃣ Test Optimization Results

#### Recommended Workflow ⭐

```
Development Phase (Rapid Iteration)
  ↓
1. Single Config Test & Submit (Approach 1)
   - Use submit mode to test single CONC config (~15-20 mins)
   - Auto-submit to Leaderboard, view ranking in real-time
  ↓
2. Multi-Concurrency Batch Test & Submit (Approach 2)
   - Use submit mode to test all CONC configs(~1-2 hours/ISL-OSL)
   - Auto-submit all results
  ↓
Done! View Leaderboard rankings in real-time 🎉
```

**Why use submit mode directly?**
- ✅ **All-in-one**: submit = accuracy test + performance test + auto-submit
- ✅ **Real-time feedback**: See Leaderboard ranking immediately, rapid iteration
- ✅ **Save time**: No need to run perf then submit, just submit directly

---

#### Approach 1: Single Config Test (Quick Validation) ⚡

**Use Case**: Quickly validate single configuration performance during development

```bash
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x

# 1. Load environment variables (no manual export needed)
source specific_conc_var.sh

# 2. Launch vLLM server (first launch needs 20+ mins for JIT compilation)
bash launch_vllm_server.sh

# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti vllm-dev bash
source specific_conc_var.sh

# 4. Recommended: Test and submit directly (~15-20 mins) ⭐
./kimi_benchmark submit "YourTeam"

# Optional: Quick accuracy validation only (~5-10 mins)
./kimi_benchmark acc

# Optional: Test performance without submitting (~15-20 mins)
./kimi_benchmark perf
```

**Environment Variables**: `specific_conc_var.sh` sets:
- `MODEL`, `PORT`, `TP` (server configuration)
- `ISL`, `OSL`, `CONC` (test configuration)
- `MAX_MODEL_LEN`, `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME` (test parameters)

**Tip**: All `.sh` scripts are located in `/workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/` directory

---

#### Approach 2: Multi-Concurrency Batch Testing (Test All CONC with One Command) 🚀

**Use Case**: Batch test all CONC values and submit to Leaderboard

**Only 3 commands to auto-test all configurations and submit! ⭐**

```bash
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x

# 1. Load environment variables (no manual export needed)
source all_conc_var.sh

# 2. Launch vLLM server (first launch needs 20+ mins for JIT compilation)
bash launch_vllm_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti vllm-dev bash
source all_conc_var.sh

# 4.
# Submit all results for ISL=8192, OSL=1024 (auto-run CONC=4,32,128, only supported case)
./kimi_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ========== Optional: Test without submitting, use perf mode ========== 

# Test ISL=8192, OSL=1024 (no submit, only supported case)
./kimi_benchmark perf -isl 8192 -osl 1024
```

**Results auto-submit to the leaderboard for the corresponding CONC** (one leaderboard per concurrency):
- CONC 4 → [Leaderboard](https://daniehua-kimik25-fp4-isl8192-osl1024-conc4.hf.space)
- CONC 32 → [Leaderboard](https://daniehua-kimik25-fp4-isl8192-osl1024-conc32.hf.space)
- CONC 128 → [Leaderboard](https://daniehua-kimik25-fp4-isl8192-osl1024-conc128.hf.space)

**Submission Content**: Each CONC configuration submits independently, including:
- Team name + CONC value
- **MI355X vs baseline Direct Comparison**: E2E, throughput, performance ratios
- Accuracy metric: `gsm8k_metric`, see [this reference](https://github.com/ROCm/ATOM/blob/main/.github/workflows/atom-test.yaml#L141)

**CONC values**: 4, 32, 128 only (only ISL=8192, OSL=1024 supported)

## Test Mode Comparison

| Mode | Command Example | What Runs | Time (Single Config)| Use Case |
|------|---------|---------|-------------|---------|
| **submit** ⭐ | `./kimi_benchmark submit "Team"` | Accuracy + Performance + Submit | ~15-20 mins | **Recommended: All-in-one, view ranking real-time** |
| **submit -isl -osl** ⭐ | `./kimi_benchmark submit "Team" -isl 8192 -osl 1024` | Auto-test CONC=4,32,128 + Submit | ~1–1.5 h | **Recommended: Batch test and submit** |
| **acc** | `./kimi_benchmark acc` | Accuracy test only | ~5-10 mins | Optional: Quick accuracy validation |
| **perf** | `./kimi_benchmark perf` | Accuracy + Performance (no submit) | ~15-20 mins | Optional: Test performance without submitting |
| **perf -isl -osl** | `./kimi_benchmark perf -isl 8192 -osl 1024` | Auto-test CONC=4,32,128 (no submit) | ~1–1.5 h | Optional: Batch test without submitting |

## Two Testing Approaches Comparison

| Approach | Recommended Command | Configs | Time Estimate | Recommended Scenario |
|------|---------|-------|---------|---------|
| **Approach 1: Single Config** ⭐ | `./kimi_benchmark submit "Team"` | 1 | ~15-20 mins | **Development phase rapid iteration** |
| **Approach 2: Multi-Concurrency** ⭐ | `./kimi_benchmark submit "Team" -isl 8192 -osl 1024` | 3 (CONC=4,32,128) | ~1–1.5 h | **Batch test all CONC** |

**Recommended Workflow** 🎯:
1. **Development Phase**: Use **Approach 1** (single config + submit) for rapid iteration, view Leaderboard real-time
2. **Batch Submission**: Use **Approach 2** (multi-conc + submit) to test CONC=4,32,128 for 8192/1024 and submit

**Why use submit directly?**
- ✅ submit = accuracy test + performance test + auto-submit (all-in-one)
- ✅ View Leaderboard ranking real-time, immediately know optimization effects
- ✅ Save time, no need to run perf then submit

## Evaluation Criteria

### Performance Metrics (Primary)

- **Throughput per GPU** (`tput_per_gpu`) - Single GPU = `total_token_throughput / 8`. Must meet or exceed the **minimum** in the baseline table below for each CONC.
- **Interactivity** (token/s/user) = `1000.0 / median_tpot_ms`. Must meet or exceed the **minimum** in the baseline table for each CONC.
- **E2E Latency (median)** (s) = `median_e2el_ms / 1000`. Must be **≤** the **maximum** in the baseline table for each CONC.

### KimiK2.5 FP4 Baseline (Minimum Requirements to Meet)

**Interactivity** vs **total token throughput per GPU**:

| CONC | Interactivity (min) | Throughput per GPU (min) |
|------|----------------------|---------------------------|
| 128  | ≥ 35 token/s/user   | ≥ 5300 token/s/GPU       |
| 32   | ≥ 65 token/s/user   | ≥ 4500 token/s/GPU       |
| 4    | ≥ 150 token/s/user  | ≥ 1350 token/s/GPU       |

**E2E Latency (median)** vs **total token throughput per GPU**:

| CONC | E2E latency (max) | Throughput per GPU (min) |
|------|--------------------|---------------------------|
| 128  | ≤ 24.5 s           | ≥ 5300 token/s/GPU       |
| 32   | ≤ 14 s             | ≥ 4500 token/s/GPU       |
| 4    | ≤ 6 s              | ≥ 1350 token/s/GPU       |

**Interpretation**: For each CONC, your result must satisfy **both** the interactivity and throughput thresholds, and **both** the E2E and throughput thresholds (E2E ≤ max, throughput ≥ min).

### Accuracy Requirements (Must Meet)

Accuracy requirement:
- `gsm8k_metric > 0.9325`

❌ Exceeding range will immediately terminate testing, performance benchmark will not run

### Baseline Comparison in Result JSON 📊

Each result JSON includes baseline thresholds and ratios:
- `tput_per_gpu_ratio_vs_baseline ≥ 1.0` = throughput meets or exceeds baseline ✅
- `median_e2e_ratio_vs_baseline ≤ 1.0` = E2E latency meets or is better than baseline ✅
- `interactivity_ratio_vs_baseline ≥ 1.0` = interactivity meets or exceeds baseline ✅

See `baseline` field in result JSON for the threshold values used.

## Development Tips

### View Logs

```bash
# View server logs in real-time
tail -f /tmp/vllm-server-*.log

# Filter errors
tail -f /tmp/vllm-server-*.log | grep -i error
```

### Multi-Concurrency Batch Testing (Recommended) ⭐

```bash
# 1. Load environment variables
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x
source all_conc_var.sh

# 2. Launch vLLM server
bash launch_vllm_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti vllm-dev bash
source specific_conc_var.sh

# Submit all results for ISL=8192, OSL=1024 (auto-test CONC=4,32,128, only supported case)
./kimi_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**Each command will automatically**:
- ✅ Test CONC values: 4, 32, 128 (only 8192/1024 supported)
- ✅ Run accuracy + performance tests
- ✅ Auto-submit to corresponding ISL-OSL Leaderboard
- ✅ Save all results to independent directory
- ✅ Generate summary report

**Leaderboard (per CONC)**:
- CONC 4: https://daniehua-kimik25-fp4-isl8192-osl1024-conc4.hf.space
- CONC 32: https://daniehua-kimik25-fp4-isl8192-osl1024-conc32.hf.space
- CONC 128: https://daniehua-kimik25-fp4-isl8192-osl1024-conc128.hf.space

**Result Output Example**:
```
============================================
Multi-Concurrency Testing Mode
============================================
ISL: 8192
OSL: 1024
Mode: submit
CONC values: 4, 32, 128
Team: YourTeam
Leaderboard: per-CONC (CONC=4, 32, 128 each submit to own leaderboard)
============================================

Results directory: batch_isl8192_osl1024_20251127_150000

============================================
Testing CONC=4
============================================
... (running tests) ...
✓ CONC=4: PASSED (180s)

============================================
Testing CONC=32
============================================
... (continue testing CONC=128) ...

============================================
Multi-Concurrency Test Complete!
============================================
Total tests: 3
Passed: 3
Failed: 0

Results saved in: batch_isl8192_osl1024_20251127_150000/
============================================
```

**Development Phase Quick Validation**:
```bash
# Recommended: Test and submit directly (all-in-one) ⭐
./kimi_benchmark submit "YourTeam" -isl 8192 -osl 1024

# Optional: Accuracy test only (quick validation)
./kimi_benchmark acc -isl 8192 -osl 1024

# Optional: Full test without submitting
./kimi_benchmark perf -isl 8192 -osl 1024
```

## FAQ

### Q: How to launch server only without running tests?

```bash
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x

# Load environment variables
source all_conc_var.sh

# Launch server
bash launch_vllm_server.sh
```

### Q: C++ code modifications not taking effect?

Need to recompile:

```bash
cd /workspace/vllm
rm -rf build/
pip uninstall -y vllm
VLLM_TARGET_DEVICE=rocm python3 setup.py develop
```

### Q: What if multi-concurrency test fails midway?

Test will continue with remaining CONC configurations, generating complete report at the end. Failed configurations will be marked as "FAILED".

View failure reasons:
```bash
# View summary
cat batch_isl*_osl*/summary.txt

# View server logs
tail -f /tmp/vllm-server-*.log
```

### Q: How to test specific CONC value only?

Use single configuration mode:

```bash
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x

# 1. Edit specific_conc_var.sh to modify CONC value
vim specific_conc_var.sh  # Modify CONC (4, 32, or 128)

# 2. Load environment variables
source specific_conc_var.sh

# 3. Recommended: Test and submit directly ⭐
./kimi_benchmark submit "YourTeam"
```

Or set manually:
```bash
cd /workspace/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x
source specific_conc_var.sh
export CONC=32  # Override default; use 4, 32, or 128
export NUM_PROMPTS=160 

# Recommended: Submit directly
./kimi_benchmark submit "YourTeam"

# Optional: Test without submitting
./kimi_benchmark perf
```

### Q: How long do tests take?

**Single Configuration Test**:
- **submit mode**: ~15-20 mins ⭐ **Recommended: All-in-one**
- **acc mode**: ~5-10 mins (Optional: Accuracy validation only)
- **perf mode**: ~15-20 mins (Optional: Test without submitting)

**Multi-Concurrency Test (per ISL-OSL combination)**:
**Multi-Concurrency Test** (only ISL=8192, OSL=1024; CONC=4,32,128):
- **CONC=4,32,128**: ~15-20 mins/CONC × 3 = **~1–1.5 hours** ⭐

**All 3 ISL-OSL Combinations** (18 configurations):
- **submit mode**: ~2 hours × 3 = **~6 hours** ⭐

**Recommended Workflow** 🎯:
1. **Development Phase**: Single config `submit "YourTeam"` rapid iteration (~15-20 mins/time)
   - See Leaderboard ranking immediately, quickly validate optimization effects
2. **Batch Submission**: `submit "YourTeam" -isl 8192 -osl 1024` to test CONC=4,32,128 and submit (~1–1.5 h)
   - Complete testing and submission at once, can run overnight

💡 **Why use submit directly?**
- ✅ All-in-one, no need to run perf then submit
- ✅ View ranking real-time, immediately know optimization effects
- ✅ Save time, avoid redundant runs

## Resource Links

- 🔧 [vLLM GitHub](https://github.com/vllm-project/vllm) - Inference framework
- 🔧 [AITER GitHub](https://github.com/ROCm/aiter) - AMD GPU operator library
- 📊 Leaderboards (one per CONC):
  - [CONC 4](https://daniehua-kimik25-fp4-isl8192-osl1024-conc4.hf.space) · [CONC 32](https://daniehua-kimik25-fp4-isl8192-osl1024-conc32.hf.space) · [CONC 128](https://daniehua-kimik25-fp4-isl8192-osl1024-conc128.hf.space)


**Good luck! 🚀**

Remember:
- **Performance matters, accuracy matters more!** All optimizations must pass accuracy validation
- **Rapid iteration**: Submit immediately after each optimization, see effects instantly

