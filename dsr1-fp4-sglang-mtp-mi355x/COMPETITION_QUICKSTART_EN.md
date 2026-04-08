# 🏆 Competition Quick Start Guide

## 📑 Table of Contents

- [Objective](#objective)
- [📌 Important Notice](#-important-notice)
- [Core Files](#core-files)
- [Quick Start (5 Steps)](#quick-start-5-steps)
  - [1️⃣ Prepare Working Directory (on Host Machine)](#1️⃣-prepare-working-directory-on-host-machine)
  - [2️⃣ Launch Development Container](#2️⃣-launch-development-container)
  - [3️⃣ Install Editable SGLang in Container](#3️⃣-install-editable-sglang-in-container)
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

Optimize inference performance of dsr1-fp4 model **with MTP (Multi-Token Prediction)** using SGLang on single node AMD MI355X GPUs with TP8 EP1, **surpass baseline on this model**, while maintaining model accuracy >= 0.93. The server is launched with MTP (EAGLE speculative decoding) enabled via `launch_sglang_server.sh`. **Only the ISL=8192, OSL=1024 (8k/1k long-context) case is benchmarked**, with CONC = 4, 32, 128.

## 📌 Important Notice

- **Single benchmark case**: Only **ISL=8192, OSL=1024 (8k/1k)** is used. Multi-conc mode only accepts `-isl 8192 -osl 1024`.
- **CONC values**: **4, 32, 128** only (set in `specific_conc_var.sh` for single runs; batch runs test all three).
- **MTP**: The server is started with MTP (EAGLE speculative decoding) via `launch_sglang_server.sh`; no extra steps required.
- **something you cannot change**:
  - model: amd/DeepSeek-R1-0528-mtp-mxfp4
  - TP: 8
  - ISL: 8192
  - OSL: 1024
  - CONC: 4,32,128
  - NUM_PROMPTS: CONC * 10
  - basline data in dsr1_benchmark.cpp
  - others, if you're not sure, let's actively talk on discord
- **unallowed optimizaiton methods**: optimization that will benefit all chips, for example, MTP core algorithms and scheduler and so on, optimization that obviously won't be accepted by SGLang, for example, new enhanced coupling AMD-specific dependencies, with no downgrade options. And pls check SGLang contribution guide for basic infos.let's actively talk on discord when you're not sure. 

## Core Files

| File | Purpose |
|------|------|
| `amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/launch_sglang_server.sh` | Launch SGLang server |
| `amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/dsr1_benchmark` | Run tests and submit results (binary file)|
| `amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/all_conc_var.sh` | Multi-concurrency test environment variables |
| `amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/specific_conc_var.sh` | Single configuration test environment variables |

## Quick Start (5 Steps)

### 1️⃣ Prepare Working Directory (on Host Machine)

```bash
# Create working directory on host machine
mkdir -p ~/competition
cd ~/competition

# Clone SGLang (you will optimize based on this)
git clone https://github.com/sgl-project/sglang.git

# Clone AITER (AMD GPU operator library)
git clone --recursive https://github.com/ROCm/aiter.git

# Clone scripts repository
git clone https://github.com/danielhua23/amdgpu_bounty_optimization.git
```

### 2️⃣ Launch Development Container

**Note**: Replace `HF_TOKEN` with your Hugging Face Token.

```bash
docker run -it \
  --name sglang-dev \
  --ipc=host --shm-size=16g --network=host \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /nfsdata/hf_hub_cache-1/:/root/.cache/huggingface \
  -v ~/competition:/workspace \
  -v ~/competition/aiter:/workspace/aiter \
  -v ~/competition/sglang:/workspace/sglang \
  -e HF_TOKEN=your_huggingface_token_here \
  lmsysorg/sglang:v0.5.8-rocm700-mi35x \
  /bin/bash
```

**Mount Instructions**:
- Host `~/competition/*` → Container `/workspace/*`
- Code modifications on host machine take effect immediately in container (and vice versa)
- Test scripts are located in `/workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/` directory

### 3️⃣ Install Editable SGLang in Container

> refer to https://docs.sglang.io/platforms/amd_gpu.html

```bash
# Uninstall existing sglang-related libraries in container
pip uninstall -y amd-aiter
pip uninstall -y sglang
pip uninstall -y sgl-kernel
# Enter aiter directory
cd /workspace/aiter
python3 setup.py develop
```
verify if newest aiter is installed
```bash
root@mi355:/workspace# pip list | grep aiter
amd-aiter                             0.1.10.post4.dev9+g1a9f7eaf0        /workspace/aiter
```
lets continue install sgl-kernel
```bash
# Enter SGLang directory
cd /workspace/sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install
```
verify if newest sgl-kernel is installed
```bash
root@mi355:/workspace# pip list | grep sgl-kernel
sgl-kernel                        0.3.18
```
lets continue install sglang python pkg
```bash
# Install sglang python package
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"

# verify
python -c "import sglang; print(sglang.__file__)"
# expect ouput: /workspace/sglang/python/sglang/__init__.py
```

### 4️⃣ Example: How to Recompile After Code Modifications

```bash
# Edit in container or host machine (VS Code)
# Example: Optimize scheduler
cd /workspace/sglang
vim python/sglang/srt/managers/scheduler.py

# If you modified C++/CUDA/HIP code, recompile:
cd sgl-kernel
rm -rf build/
pip uninstall sgl-kernel
python setup_rocm.py install
```

### 5️⃣ Test Optimization Results

#### Recommended Workflow ⭐

```
Development Phase (Rapid Iteration)
  ↓
1. Single Config Test & Submit (Approach 1)
   - Use submit mode to test single CONC (~20 mins), ISL=8192, OSL=1024
   - Auto-submit to Leaderboard, view ranking in real-time
  ↓
2. Multi-Concurrency Batch Test & Submit (Approach 2)
   - Use submit mode: `submit "Team" -isl 8192 -osl 1024` to test CONC=4,32,128 (~1–1.5 hours)
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

**Use Case**: Quickly validate single CONC config performance during development

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x

# 1. Load environment variables (no manual export needed)
source specific_conc_var.sh

# 2. Launch SGLang server (first launch needs 20+ mins for JIT compilation)
bash launch_sglang_server.sh

# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new windown and reload env var
docker exec -ti sglang-dev bash
source specific_conc_var.sh
# 4. Recommended: Test and submit directly (~20-30 mins) ⭐
./dsr1_benchmark submit "YourTeam"

# Optional: Quick accuracy validation only (~5-10 mins)
./dsr1_benchmark acc

# Optional: Test performance without submitting (~20-30 mins)
./dsr1_benchmark perf
```

**Environment Variables**: `specific_conc_var.sh` sets:
- `MODEL`, `PORT`, `TP` (server configuration)
- `ISL`, `OSL`, `CONC` (test configuration)
- `RANDOM_RANGE_RATIO`, `NUM_PROMPTS`, `RESULT_FILENAME` (test parameters)

**Tip**: All `.sh` scripts are located in `/workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x/` directory

---

#### Approach 2: Multi-Concurrency Batch Testing (Test All CONC with One Command) 🚀

**Use Case**: Batch test CONC = 4, 32, 128 for the **only supported case ISL=8192, OSL=1024 (8k/1k)** and submit to Leaderboard

**One command to auto-test all 3 CONC values and submit! ⭐**

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x

# 1. Load environment variables (no manual export needed)
source all_conc_var.sh

# 2. Launch SGLang server with MTP (first launch needs 20+ mins for JIT compilation)
bash launch_sglang_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti sglang-dev bash
source all_conc_var.sh

# 4. Submit all results for ISL=8192, OSL=1024 (auto-run CONC=4,32,128, ~45mins)
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024

# ========== Optional: Test without submitting, use perf mode ========== 
./dsr1_benchmark perf -isl 8192 -osl 1024
```

**Results auto-submit to the leaderboard for the corresponding CONC** (one leaderboard per concurrency):
- CONC 4 → [Leaderboard](https://daniehua-dsr1-fp4-isl8192-osl1024-conc4.hf.space)
- CONC 32 → [Leaderboard](https://daniehua-dsr1-fp4-isl8192-osl1024-conc32.hf.space)
- CONC 128 → [Leaderboard](https://daniehua-dsr1-fp4-isl8192-osl1024-conc128.hf.space)

**Submission Content**: Each CONC configuration submits independently, including:
- Team name + CONC value
- **MI355X vs baseline Direct Comparison**: E2E, throughput, performance ratios
- Accuracy metrics: gsm8k_metric


## Test Mode Comparison

| Mode | Command Example | What Runs | Time (Single Config) | Use Case |
|------|---------|---------|-------------|---------|
| **submit** ⭐ | `./dsr1_benchmark submit "Team"` | Accuracy + Performance + Submit (ISL=8192, OSL=1024) | ~20-30 mins | **Recommended: All-in-one, view ranking real-time** |
| **submit -isl -osl** ⭐ | `./dsr1_benchmark submit "Team" -isl 8192 -osl 1024` | Auto-test CONC=4,32,128 + Submit | ~50mins | **Recommended: Batch test and submit** |
| **acc** | `./dsr1_benchmark acc` | Accuracy test only | ~5-10 mins | Optional: Quick accuracy validation |
| **perf** | `./dsr1_benchmark perf` | Accuracy + Performance (no submit) | ~20-30 mins | Optional: Test performance without submitting |
| **perf -isl -osl** | `./dsr1_benchmark perf -isl 8192 -osl 1024` | Auto-test CONC=4,32,128 (no submit) | ~45mins | Optional: Batch test without submitting |

## Two Testing Approaches Comparison

| Approach | Recommended Command | Configs | Time Estimate | Recommended Scenario |
|------|---------|-------|---------|---------|
| **Approach 1: Single Config** ⭐ | `./dsr1_benchmark submit "Team"` | 1 (8k/1k, CONC from env) | ~20 mins | **Development phase rapid iteration** |
| **Approach 2: Multi-Concurrency** ⭐ | `./dsr1_benchmark submit "Team" -isl 8192 -osl 1024` | 3 (CONC=4,32,128) | ~45mins | **Batch test all CONC** |

**Recommended Workflow** 🎯:
1. **Development Phase**: Use **Approach 1** (single config + submit) for rapid iteration, view Leaderboard real-time
2. **Batch Submission**: Use **Approach 2** (multi-conc + submit) to test and submit all configurations at once

**Why use submit directly?**
- ✅ submit = accuracy test + performance test + auto-submit (all-in-one)
- ✅ View Leaderboard ranking real-time, immediately know optimization effects
- ✅ Save time, no need to run perf then submit

## Evaluation Criteria

### Performance Metrics (Primary)

- **Throughput per GPU** (`tput_per_gpu`) - Highest weight 🏅
  - Single GPU normalized throughput = `total_token_throughput / 8`
  - Direct comparison with baseline
- **E2E (median)** (ms) - End-to-end latency median
  - Direct comparison with baseline

### Accuracy Requirements (Must Meet)

All metrics must be:
- gsm8k_metric >= 0.93

❌ Exceeding range will immediately terminate testing, performance benchmark will not run

### Baseline Comparison 📊

**Auto-comparison feature**: Each result JSON automatically includes baseline data and performance ratios!

**Performance comparison**: Compare your result with the baseline thresholds in the result JSON.

See `baseline` field in result JSON for threshold values.

## Optimization Directions

### 1. Kernel Optimization ⚡
- Attention kernel
- MoE (Mixture of Experts) kernel  
- Quantization kernel (FP4/FP8)

### 2. Scheduling Optimization 📊
- Batch scheduler
- Prefill/decode switching strategy
- KV cache management

### 3. Memory Optimization 💾
- Memory allocation strategy
- Reduce memory fragmentation
- Paged attention

### 4. ROCm-Specific Optimization 🔧
- Leverage AMD GPU features
- HIP/ROCm API optimization
- AITER async iterator

## Development Tips

### View Logs

```bash
# View server logs in real-time
tail -f /tmp/sglang-server-*.log

# Filter errors
tail -f /tmp/sglang-server-*.log | grep -i error
```

### Multi-Concurrency Batch Testing (Recommended) ⭐

```bash
# 1. Load environment variables
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x
source all_conc_var.sh

# 2. Launch SGLang server with MTP
bash launch_sglang_server.sh

# ========== Recommended: Test and submit directly (all-in-one) ========== 
# Wait for server ready (see "application startup..."）, then open a new window to run tests
# 3. go into new window and reload env var
docker exec -ti sglang-dev bash
source all_conc_var.sh
# 4. Only supported case: ISL=8192, OSL=1024 (CONC=4,32,128)
./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024
```

**Each command will automatically**:
- ✅ Test 3 CONC values (4, 32, 128)
- ✅ Run accuracy + performance tests
- ✅ Auto-submit to Leaderboard
- ✅ Save all results to independent directory
- ✅ Generate summary report

**Leaderboard (per CONC)**:
- CONC 4: https://daniehua-dsr1-fp4-isl8192-osl1024-conc4.hf.space
- CONC 32: https://daniehua-dsr1-fp4-isl8192-osl1024-conc32.hf.space
- CONC 128: https://daniehua-dsr1-fp4-isl8192-osl1024-conc128.hf.space

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

Results directory: batch_isl8192_osl1024_20251125_150000

============================================
Testing CONC=4
============================================
... (running tests) ...
✓ CONC=4: PASSED (180s)
...
============================================
Multi-Concurrency Test Complete!
============================================
Total tests: 3
Passed: 3
Failed: 0
============================================
```

**Development Phase Quick Validation**:
```bash
# Recommended: Test and submit directly (all-in-one) ⭐
./dsr1_benchmark submit "YourTeam"

# Optional: Accuracy test only (quick validation)
./dsr1_benchmark acc

# Optional: Full test without submitting
./dsr1_benchmark perf
```

## FAQ

**Solution**: Your optimization affected model quality, need to adjust algorithm or parameters

### Q: How to launch server only without running tests?

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x

# Load environment variables
source all_conc_var.sh

# Launch server
bash launch_sglang_server.sh
```

Server will run in foreground, logs output directly to terminal.

### Q: C++ code modifications not taking effect?

Need to recompile:

```bash
cd /workspace/sglang/sgl-kernel
rm -rf build/
pip uninstall sgl-kernel
python setup_rocm.py install
```

### Q: What if multi-concurrency test fails midway?

Test will continue with remaining CONC configurations, generating complete report at the end. Failed configurations will be marked as "FAILED".

View failure reasons:
```bash
# View summary
cat batch_isl*_osl*/summary.txt

# View server logs
tail -f /tmp/sglang-server-*.log
```

### Q: How to test specific CONC value only?

Use single configuration mode (CONC must be 4, 32, or 128 for 8k/1k):

```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x

# 1. Edit specific_conc_var.sh to modify CONC value (4, 32, or 128)
vim specific_conc_var.sh  # e.g. CONC=32

# 2. Load environment variables
source specific_conc_var.sh

# 3. Recommended: Test and submit directly ⭐
./dsr1_benchmark submit "YourTeam"
```

Or set manually:
```bash
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-sglang-mi355x
source specific_conc_var.sh
export CONC=32  # Override default; use 4, 32, or 128
export NUM_PROMPTS=$((CONC * 10))

# Recommended: Submit directly
./dsr1_benchmark submit "YourTeam"

# Optional: Test without submitting
./dsr1_benchmark perf
```

### Q: How long do tests take?

**Single Configuration Test** (ISL=8192, OSL=1024):
- **submit mode**: ~20-30 mins ⭐ **Recommended: All-in-one**
- **acc mode**: ~5-10 mins (Optional: Accuracy validation only)
- **perf mode**: ~20-30 mins (Optional: Test without submitting)

**Multi-Concurrency Test** (CONC=4,32,128 only):
- **submit -isl 8192 -osl 1024**: ~20-30 mins/CONC × 3 = **~1–1.5 hours** ⭐ **Recommended**
- **perf -isl 8192 -osl 1024**: **~45mins** (Optional)

**Recommended Workflow** 🎯:
1. **Development Phase**: Single config `submit "YourTeam"` rapid iteration (~20 mins/time)
   - See Leaderboard ranking immediately, quickly validate optimization effects
2. **Batch Submission**: `submit "YourTeam" -isl 8192 -osl 1024` to test CONC=4,32,128 and submit (~50mins)

💡 **Why use submit directly?**
- ✅ All-in-one, no need to run perf then submit
- ✅ View ranking real-time, immediately know optimization effects
- ✅ Save time, avoid redundant runs



## Recommended Workflow

```
Round 1: Familiarize with Baseline
  ├─ Run baseline test: ./dsr1_benchmark submit "YourTeam"
  ├─ Understand SGLang architecture
  └─ View Leaderboard baseline performance

Round 3: AMD GPU Kernel Optimization
  ├─ Profile to find bottlenecks
  ├─ Optimize critical kernels
  └─ Real-time comparison: ./dsr1_benchmark submit "YourTeam", view Leaderboard

Round 4: System Optimization (note: please be careful if that will benefit all the chips or if that will be accepted by SGLang, see important notice on top)
  ├─ Memory management
  ├─ Communication optimization
  └─ End-to-end tuning, submit for validation after each optimization

Round 5: Batch Submission
  └─ ./dsr1_benchmark submit "YourTeam" -isl 8192 -osl 1024   # CONC=4,32,128
```

**Key Advantage**: Submit directly after each optimization, view Leaderboard ranking real-time, rapid iteration!

## Resource Links

- 🔧 [SGLang GitHub](https://github.com/sgl-project/sglang) - Inference framework
- 📊 Leaderboards (one per CONC): [CONC 4](https://daniehua-dsr1-fp4-isl8192-osl1024-conc4.hf.space) · [CONC 32](https://daniehua-dsr1-fp4-isl8192-osl1024-conc32.hf.space) · [CONC 128](https://daniehua-dsr1-fp4-isl8192-osl1024-conc128.hf.space)


**Good luck! 🚀**

Remember:
- **Use submit mode directly**: All-in-one, view ranking real-time ⭐
- **Performance matters, accuracy matters more!** All optimizations must pass accuracy validation
- **Rapid iteration**: Submit immediately after each optimization, see effects instantly


