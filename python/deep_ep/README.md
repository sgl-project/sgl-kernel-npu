<h2 align="left">
DeepEP-Ascend
</h2>

<p align="left">
<a><b>English</b></a> | <a href="README_CN.md"><b>中文</b></a>
</p>


## Introduction

**DeepEP-Ascend** is the Ascend NPU implementation of [DeepEP](https://github.com/deepseek-ai/DeepEP), providing highly optimized Expert Parallelism (EP) communication kernels for Mixture-of-Experts (MoE) models on Ascend hardware. It supports two communication modes:

- **Normal Mode**: High-throughput dispatch and combine operations for training and prefill phases.
- **Low-Latency Mode**: Optimized for production inference with small batch sizes, achieving sub-150us latency.

DeepEP-Ascend uses a **strategy-based architecture** that allows flexible selection of communication implementations via environment variables, supporting various hardware topologies (A2, A3, A5) and communication backends (HCCS, RDMA, AlltoAll).


## Software and Hardware

Supported Hardware Models: Atlas A2, A3 (support CANN 8.5 and CANN 9.0), and Atlas A5 (only supports CANN 9.0).
Platform: aarch64/x86

Supporting Software:
- Driver Ascend HDK 25.1.RC1.1, CANN Community Edition 8.5.0 and later versions (refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) to install the CANN development kit package, as well as the supporting firmware and drivers)
- Before installing CANN software, you need to install the relevant [dependency list](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0045.html)
- Python >= 3.9, Recommendation: Python 3.11
- PyTorch >= 2.8.0, torch-npu >= 2.8.0


## Quick Start

DeepEP-Ascend supports A2, A3 and A5 and needs to generate packages separately on each platform.

### Compile and Build

1. Prepare the CANN environment variables (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Build the project
- **A5**
    ```bash
    bash build.sh -a deepep Ascend950
    ```
- **A3**
    ```bash
    bash build.sh -a deepep
    ```
- **A2**
    ```bash
    bash build.sh -a deepep2
    ```

### Installation

1. Pip install the `.whl` file into your Python environment
```bash
pip install output/deep_ep*.whl

# Link to the deep_ep_cpp.*.so file
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -

# (Optional) Confirm whether the import can be successful
python -c "import deep_ep; print(deep_ep.__path__)"
```

2. Execute the environment variables for CANN (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. In the Python project, import `deep_ep`
```python
import deep_ep
```


## Architecture

DeepEP-Ascend employs a **strategy-based architecture** where communication implementations are abstracted into interchangeable strategies, selected via environment variables.

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Buffer** | `buffer.py` | Main entry point. Initializes communication buffer and delegates to strategy objects. |
| **NormalStrategy** | `ep_strategy.py` / `strategies/normal_strategy.py` | Normal mode dispatch/combine strategies (default, alltoall). |
| **LowLatencyStrategy** | `ep_strategy.py` / `strategies/low_latency_strategy.py` | Low-latency mode dispatch/combine strategies (default, ops, alltoall). |
| **EventOverlap** | `utils.py` | Event synchronization utility for async operations. |
| **FuseMode** | `buffer.py` | Enum for fused MoE computation modes. |

### Strategy Selection

Strategies are configured via environment variables at Buffer initialization:

| Environment Variable | Value | Normal Strategy | Low-Latency Strategy |
|---------------------|-------|-----------------|---------------------|
| `DEEP_USE_MODE=default` | default | `DefaultNormalCommStrategy` (deep_ep_cpp custom ops) | `DefaultLowLatencyCommStrategy` (deep_ep_cpp custom ops) |
| `DEEP_USE_MODE=alltoall` | alltoall | `AlltoAllNormalCommStrategy` (torch.distributed alltoallv) | `AllToAllLowLatencyCommStrategy` (torch.distributed alltoall) |
| `DEEP_USE_MODE=default` | ops | `DefaultNormalCommStrategy` (deep_ep_cpp custom ops) | `OpsLowLatencyCommStrategy` (torch_npu ops) |

> **Note**: Invalid env (e.g., `DEEP_USE_MODE=error`) will raise a `ValueError`.


## API Overview

The `Buffer` class is the primary interface. Below is a summary of the core APIs:

| API | Mode | Description |
|-----|------|-------------|
| `Buffer(group, num_nvl_bytes, num_rdma_bytes, ...)` | — | Initialize communication buffer with strategy selection. |
| `get_dispatch_layout(topk_idx, num_experts, ...)` | Normal | Calculate layout for subsequent dispatch. Returns `num_tokens_per_rank`, `num_tokens_per_rdma_rank`, `num_tokens_per_expert`, `is_token_in_rank`. |
| `dispatch(x, topk_idx, topk_weights, ...)` | Normal | Dispatch tokens to expert ranks. Returns received tokens, topk info, and a handle for combine. |
| `combine(x, handle, ...)` | Normal | Combine (reduce) tokens from dispatch. Must use the handle returned by `dispatch`. |
| `low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, ...)` | Low-Latency | Low-latency token dispatch for decode phase. |
| `low_latency_combine(x, topk_idx, topk_weights, handle, ...)` | Low-Latency | Low-latency token combine for decode phase. |
| `fused_deep_moe(x, topk_idx, topk_weights, ...)` | Fused | Fused dispatch + FFN + combine in a single call. |
| `get_dispatch_config(num_ranks)` | — | Get recommended Config for normal dispatch. |
| `get_combine_config(num_ranks)` | — | Get recommended Config for normal combine. |
| `clean_low_latency_buffer(...)` | — | Clean buffer before switching from normal to low-latency mode. |

For detailed API documentation, see:
- [Normal Mode API (dispatch/combine)](doc/NORMAL_API.md)
- [Low-Latency Mode API](doc/LOW_LATENCY_API.md)
- [get_dispatch_layout API](doc/GET_DISPATCH_LAYOUT_API.md)
- [Fused Deep MoE API (English)](doc/FUSED_DEEP_MOE_EN.md) | [中文](doc/FUSED_DEEP_MOE_CN.md)


## Communication Modes

### Normal Mode (Prefill / Training)

High-throughput dispatch and combine for training and prefill phases:
- **A3**: Pure HCCS intranode communication, full-mesh HCCS internode communication. No hierarchical implementation needed.
- **A2 Intranode**: Pure HCCS communication, supports up to `bs=8000` for normal dispatch/combine.
- **A2 Internode**: Hierarchical (HCCS intranode + RDMA internode) or non-hierarchical (pure RDMA) implementation. Supports up to `bs=4096`.

### Low-Latency Mode (Decode)

Optimized for inference with small batch sizes (128 tokens/batch):
- **A3**: Supports `default`, `ops`, and `alltoall` strategies. `ops` strategy supports `comm_alg` options: `hierarchy`, `fullmesh_v1`, `fullmesh_v2`, `ccu`.
- **A2 Intranode**: Supports up to `bs=512` for low_latency dispatch/combine.
- **A2 Internode**: Hierarchical (HCCS + RDMA) or non-hierarchical (pure RDMA) implementation. Supports up to `bs=512`.


## Fused MoE

The `fused_deep_moe` API fuses dispatch + expert FFN computation + combine into a single operator call, significantly reducing communication overhead and end-to-end latency.

Two fuse modes are available via the `FuseMode` enum:
- `FuseMode.FUSED_DEEP_MOE` (default): Full fusion of dispatch + FFN + combine.
- `FuseMode.DISPATCH_FFN_COMBINE`: Dispatch + FFN + combine with separate dispatch handling.

Quantization modes (`quant_mode`):
- `1`: INT8 quantization (default)
- FP8 will be supported in A5 release.

See [Fused Deep MoE API](doc/FUSED_DEEP_MOE_EN.md) for details.


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_USE_MODE` | `default` | Normal mode strategy and Low-latency mode strategy: `default`, `ops`, or `alltoall`. |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | Enable INT8 quantization in normal dispatch. A2 internode does NOT support quantization in normal mode. |
| `SGLANG_DEEPEP_BF16_DISPATCH` | `0` | Disable quantization in low_latency_dispatch (BF16 dispatch). Set to `1` to disable; only effective in decode phase. |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | Dispatch return type for `num_recv_tokens_per_expert_list`: `1` = per-expert token count, `0` = prefix sum. |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | Number of shared expert ranks (used by ops strategy). |
| `HCCL_BUFFSIZE` | `200` (MB) | HCCL buffer size in MB. **Must be set** when using DeepEP on A2. |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | Set to `1` for A2 dual-node hierarchical communication. |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | Set to `0` for A2 dual-node hierarchical communication. |
| `HCCL_OP_EXPANSION_MODE` | — | **Must be disabled** on A2 when using DeepEP (remove or unset this variable). |
| `DEBUG_MODE` | `OFF` | Set to `ON` to enable DEBUG logging for parameter tracing. |


## Platform-Specific Notes

### A2 Single Node

- Applicable when P/D node ranks = 8 (supports PD separation or mixed deployment).
- Not recommended when ranks < 8 (insufficient parallelism for EP benefits).
- **Performance limits**: normal up to `bs=8000`, low_latency up to `bs=512`.
- **Must set** `HCCL_BUFFSIZE` (e.g., `export HCCL_BUFFSIZE=1024`).
- **Must disable** `HCCL_OP_EXPANSION_MODE`.

For detailed A2 usage, see [A2_DEEPEP_CN.md](A2_DEEPEP_CN.md).

### A2 Dual Node

- Applicable when P/D node ranks > 8 (cross-node communication).
- **Normal mode does NOT support quantization** (`DEEP_NORMAL_MODE_USE_INT8_QUANT=0`).
- **Must set** `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0` for hierarchical communication.
- **Performance limits**: normal up to `bs=4096`, low_latency up to `bs=512`.

### A3

- Pure HCCS communication for both intranode and internode. No hierarchical implementation needed.
- Supports `ops` strategy with multiple `comm_alg` options for low-latency mode.

### A5

- Only supports CANN 9.0.
- Build with: `bash build.sh -a deepep Ascend950`.


## Test

Execute DeepEP-related test scripts:

```bash
python3 tests/python/deepep/test_fused_deep_moe.py
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py

# A2 single-node tests
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8

# A2 dual-node internode test (set primary node IP in run_test_internode.sh first)
bash tests/python/deepep/run_test_internode.sh
```


## FAQ

1. If installing the `.whl` file results in the inability to import `deep_ep` in the project, check whether it is correctly installed in the `site-packages` directory of the current Python environment:
```
pip show deep-ep
```

2. If after installing the `.whl`, you encounter an issue where `deep_ep_cpp` is not found, you need to create a symbolic link of the `deep_ep_cpp*.so` files from the `site-packages/deep_ep` directory to the `site-packages` directory. Execute the following command in the `site-packages` directory:
```
ln -s deep_ep/deep_ep_cpp*.so
```

3. If you get a `ValueError` about unsupported mode combination, check that `DEEP_NORMAL_MODE` and `DEEP_LOW_LATENCY_MODE` are a valid pair. See the [Strategy Selection](#strategy-selection) table for valid combinations.

4. On A2, always set `HCCL_BUFFSIZE` before running DeepEP. Missing this will cause dispatch/combine operators to fail.
