<div align="center">

# DeepEP-Ascend

**High-performance Expert Parallelism for MoE on Ascend NPUs**

[![Platform](https://img.shields.io/badge/Platform-A2%20%7C%20A3%20%7C%20A5-blue)]()
[![CANN](https://img.shields.io/badge/CANN-8.5%2B%20%7C%209.0-green)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)]()

English | [简体中文](README_CN.md)

[Normal API](doc/NORMAL_API.md) · [Low-Latency API](doc/LOW_LATENCY_API.md) · [Fused MoE](doc/FUSED_DEEP_MOE.md) · [Platform Guide](doc/PLATFORM_GUIDE.md)

</div>

---

## Why DeepEP-Ascend?

DeepEP-Ascend is the Ascend NPU port of [DeepEP](https://github.com/deepseek-ai/DeepEP), providing optimized Expert Parallelism (EP) communication for Mixture-of-Experts models on Atlas hardware.

**Three operation modes, one library:**

- 🚀 **Normal** — High-throughput dispatch/combine for training & prefill
- ⚡ **Low-Latency** — Sub-150us decode-phase inference
- 🔥 **Fused MoE** — Dispatch + FFN + combine in a single call (A3 only)

**Key features:**

- Strategy-based architecture (`default`, `ops`, `alltoall`) via `DEEP_USE_MODE`
- Multi-quantization: BF16, INT8 per-token, MXFP8 per-block (A5)
- Support for A2, A3, A5 (C310) platforms

## Platform Capabilities

| | A2 Single | A2 Dual | A3 | A5 (C310) |
|:---|:---:|:---:|:---:|:---:|
| Normal dispatch/combine | ✅ | ✅ (no quant) | ✅ | ✅ |
| Low-latency dispatch/combine | ✅ | ✅ | ✅ | ✅ |
| Fused MoE | ❌ | ❌ | ✅ INT8 | 🔜 planned |
| INT8 per-token quant | ✅ | ✅ (LL only) | ✅ | ✅ |
| MXFP8 per-block quant | ❌ | ❌ | ❌ | ✅ |
| Normal max BS | 8000 | 4096 | 8192 (32k long-seq) | — |
| Low-latency max BS | 512 | 512 | 512 | 512 |

> [!NOTE]
> Quantization details: [Normal API](doc/NORMAL_API.md) and [Low-Latency API](doc/LOW_LATENCY_API.md). Platform setup: [Platform Guide](doc/PLATFORM_GUIDE.md).

## Quick Start

### Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# A5 (C310)
bash build.sh -a deepep Ascend950

# A3
bash build.sh -a deepep

# A2
bash build.sh -a deepep2
```

### Install & Verify

```bash
pip install output/deep_ep*.whl

# Link deep_ep_cpp shared library
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" \
  && ln -s deep_ep/deep_ep_cpp*.so && cd -

# Verify installation
python -c "import deep_ep; print(deep_ep.__path__)"
```

### Requirements

| | A2 | A3 | A5 |
|:---|:---|:---|:---|
| CANN | 8.5+ | 8.5+ / 9.0+ | 9.0 only |
| Driver | HDK 25.1.RC1.1+ | HDK 25.1.RC1.1+ | HDK 25.1.RC1.1+ |
| Python | ≥ 3.9 (3.11 recommended) | ≥ 3.9 (3.11 recommended) | ≥ 3.9 (3.11 recommended) |
| PyTorch | ≥ 2.8.0, torch-npu ≥ 2.8.0 | ≥ 2.8.0, torch-npu ≥ 2.8.0 | ≥ 2.8.0, torch-npu ≥ 2.8.0 |

## Architecture

Strategy selection is controlled by the **single** environment variable `DEEP_USE_MODE`. The chosen value determines both Normal and Low-Latency strategies simultaneously:

| `DEEP_USE_MODE` | Normal Strategy | Low-Latency Strategy |
|:---|:---|:---|
| `default` | DefaultNormalCommStrategy (custom ops) | DefaultLowLatencyCommStrategy (custom ops) |
| `ops` | DefaultNormalCommStrategy (custom ops) | OpsLowLatencyCommStrategy (torch_npu ops) |
| `alltoall` | AlltoAllNormalCommStrategy (torch.distributed) | AllToAllLowLatencyCommStrategy (torch.distributed) |

> [!WARNING]
> Invalid `DEEP_USE_MODE` values raise `ValueError`. The `normal_strategy` and `low_latency_strategy` parameters in `Buffer.__init__` are currently overridden by `DEEP_USE_MODE` — passing custom values has no effect.

The `ops` strategy for low-latency supports `comm_alg` options: see [Low-Latency API](doc/LOW_LATENCY_API.md).

## API Overview

| API | Mode | Description | Doc |
|:---|:---|:---|:---|
| `get_dispatch_layout` | Normal | Calculate dispatch layout | [API](doc/GET_DISPATCH_LAYOUT_API.md) |
| `dispatch` / `combine` | Normal | High-throughput token dispatch & reduce | [API](doc/NORMAL_API.md) |
| `low_latency_dispatch` / `low_latency_combine` | Low-Latency | Low-latency token dispatch & reduce | [API](doc/LOW_LATENCY_API.md) |
| `fused_deep_moe` | Fused | Dispatch + FFN + combine (A3 only) | [API](doc/FUSED_DEEP_MOE.md) |

## Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `DEEP_USE_MODE` | `default` | Strategy selection: `default`, `ops`, or `alltoall`. Controls both Normal and Low-Latency simultaneously. |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | ⛔ **Deprecated but still functional.** Enable INT8 per-token quantization in normal dispatch. Recommend using tuple input instead. |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | `1` = per-expert token count, `0` = prefix sum. |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | Shared expert rank count (ops strategy). |

<details>
<summary><b>HCCL Variables (A2 only)</b></summary>

| Variable | Default | Description |
|:---|:---|:---|
| `HCCL_BUFFSIZE` | `200` (MB) | HCCL buffer size (MB). **Must set on A2** (e.g., `1024`). Default 200MB insufficient. |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | Set `1` for A2 dual-node hierarchical. |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | Set `0` for A2 dual-node hierarchical. |
| `HCCL_OP_EXPANSION_MODE` | — | **Must unset/disable on A2.** |

</details>

## Test

```bash
# Normal + Low-Latency (all platforms)
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py

# Fused MoE (A3 only)
python3 tests/python/deepep/test_fused_deep_moe.py

# A2 single-node (8 processes)
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8

# A2 dual-node (set IP in run_test_internode.sh first)
bash tests/python/deepep/run_test_internode.sh
```

## FAQ

1. **Import failure** — Check installation: `pip show deep-ep`
2. **`deep_ep_cpp` not found** — Create symlink: `cd site-packages && ln -s deep_ep/deep_ep_cpp*.so`
3. **A2 errors** — Always set `HCCL_BUFFSIZE` before running DeepEP
4. **Strategy not working** — Check `DEEP_USE_MODE` is set to a valid value (`default`, `ops`, `alltoall`). Invalid values raise `ValueError`.
