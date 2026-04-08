# AMD Instinct MI355X Performance Optimization Bounty Program

## 🎯 Overview

Welcome to the **AMD Instinct MI355X Performance Optimization Bounty Program**! Stage 2! This competition challenges developers to optimize LLM inference performance on AMD's latest Instinct MI355X GPUs, targeting to beat baseline performance

## 🏆 Stage 2 Competition Rules
### Important Note

We have noticed some people used AI agents to submit codes constantly in the stage 1, which is banned in the stage 2, you can only benchmark your codes in your allowed time range, since two teams share one node. Moreover, we'll use huggingface leaderboard to show the rank of phase 2, huggingface leaderboard will show all submissions you submitted instead of the best one, so **FLOODING IS PROHIBITED!**

### Tracks Overview
#### Track 1: DeepSeek-R1-0528 FP4 + MTP
- ISL/OSL: 8k/1k
- Concurrency: 4, 32, 128
- TP: no restriction, but less or equal to 8, single node
- EP: no restriction, but less or equal to 8, single node

#### Track 2: KimiK2.5 FP4
- ISL/OSL: 8k/1k
- Concurrency: 4, 32, 128
- TP: no restriction, but less or equal to 8, single node
- EP: no restriction, but less or equal to 8, single node

### Rules

see https://docs.google.com/document/d/1J3yNnznmE-KKYzFFo8mdGJZnLjlITUOwFBD99Md7oGQ/edit?tab=t.0#heading=h.fpbfvfydozsa starting from page3

## 📊 Benchmarked Configurations & Leaderboard link
 
### Models & Backends
**Important Note: Backend is not limited, SGLang and vLLM is just for example, you can choose any framework you familiar with. It will be accepted only if the model performance surpass baseline and your codes are mergable, This bounty benchmarks only one case: ISL=8192, OSL=1024 (8k/1k long-context), with CONC = 4, 32, 128. You MUST submit your result to the Leaderboards below!**

| Model | Backend | Directory | Leaderboard (per CONC) |
|-------|---------|-----------|-------------------------|
| **DeepSeek-R1 MTP FP4** | SGLang | `dsr1-fp4-sglang-mtp-mi355x/` | [CONC 4](https://daniehua-dsr1-fp4-isl8192-osl1024-conc4.hf.space) · [CONC 32](https://daniehua-dsr1-fp4-isl8192-osl1024-conc32.hf.space) · [CONC 128](https://daniehua-dsr1-fp4-isl8192-osl1024-conc128.hf.space) |
| **DeepSeek-R1 MTP FP4** | AMD ATOM | `dsr1-fp4-atom-mtp-mi355x/` | [CONC 4](https://daniehua-dsr1-fp4-isl8192-osl1024-conc4.hf.space) · [CONC 32](https://daniehua-dsr1-fp4-isl8192-osl1024-conc32.hf.space) · [CONC 128](https://daniehua-dsr1-fp4-isl8192-osl1024-conc128.hf.space) |
| **KimiK2.5 FP4** | vLLM | `kimik25-fp4-vllm-mi355x/` | [CONC 4](https://daniehua-kimik25-fp4-isl8192-osl1024-conc4.hf.space) · [CONC 32](https://daniehua-kimik25-fp4-isl8192-osl1024-conc32.hf.space) · [CONC 128](https://daniehua-kimik25-fp4-isl8192-osl1024-conc128.hf.space) |


### Test Configurations

**Only the following configuration is benchmarked**:

| ISL | OSL | Description | CONC |
|-----|-----|-------------|------|
| 8192 | 1024 | Long context (8k/1k) | **4, 32, 128** |

**CONC** = Maximum concurrent requests. You must run and submit results for all three CONC values (4, 32, 128) for the single supported case ISL=8192, OSL=1024.


## 🚀 Quick Start Guide

> note: the guide is just for reference, you don't have to follow it, e.g. you can build native ROCm yourself or change the docker images to newest,etc..

**Benchmark case: ISL=8192, OSL=1024 (8k/1k) only; CONC = 4, 32, 128.**

### KimiK2.5 FP4 (vLLM)
→ [kimik25-fp4-vllm-mi355x KIMIK25_COMPETITION_QUICKSTART_EN.md](kimik25-fp4-vllm-mi355x/KIMIK25_COMPETITION_QUICKSTART_EN.md)

### DeepSeek-R1 MTP (SGLang)

> Note: it's not mandatory that you must use SGLang to optimize DeepSeek-R1 MTP; AMD ATOM is also the choice and it might be the better choice given that it's taken fully controlled by AMD and might be with better performance.

→ [dsr1-fp4-sglang-mtp-mi355x COMPETITION_QUICKSTART_EN.md](dsr1-fp4-sglang-mtp-mi355x/COMPETITION_QUICKSTART_EN.md)

### DeepSeek-R1 MTP (AMD ATOM)
→ [dsr1-fp4-atom-mtp-mi355x COMPETITION_QUICKSTART_EN.md](dsr1-fp4-atom-mtp-mi355x/COMPETITION_QUICKSTART_EN.md)

### Hardware
- AMD Instinct MI355X GPU (8 GPUs)

### 🔗 Important Links

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **AMD ATOM GitHub**: https://github.com/ROCm/ATOM/
- **AMD ROCm**: https://rocm.docs.amd.com/

## Acknowledgments

- **vLLM Team** for the vLLM inference engine
- **SGLang Team** for the SGLang inference framework
- **AMD ATOM Team** for the ATOM inference framework
- **AMD** for the Instinct MI355X GPUs and ROCm platform

