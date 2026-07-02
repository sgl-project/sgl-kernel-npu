# DeepEP-Ascend

<div align="center">

[![Platform](https://img.shields.io/badge/Platform-A2%20%7C%20A3%20%7C%20A5-blue)]()
[![CANN](https://img.shields.io/badge/CANN-8.5%2B%20%7C%209.0-green)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)]()

English | [中文](#中文)

[Normal API](doc/NORMAL_API.md) · [Low-Latency API](doc/LOW_LATENCY_API.md) · [Fused MoE](doc/FUSED_DEEP_MOE.md) · [A2 Guide](doc/A2_DEEPEP.md)

</div>

---

## English

### Introduction

**DeepEP-Ascend** is the Ascend NPU implementation of [DeepEP](https://github.com/deepseek-ai/DeepEP), providing highly optimized Expert Parallelism (EP) communication kernels for Mixture-of-Experts (MoE) models on Ascend hardware. It supports two communication modes:

- **Normal Mode**: High-throughput dispatch and combine operations for training and prefill phases.
- **Low-Latency Mode**: Optimized for production inference with small batch sizes, achieving sub-150us latency.

DeepEP-Ascend uses a **strategy-based architecture** that allows flexible selection of communication implementations via environment variables, supporting various hardware topologies (A2, A3, A5) and communication backends (HCCS, RDMA, AlltoAll).

### Software and Hardware

Supported Hardware Models: Atlas A2, A3 (support CANN 8.5 and CANN 9.0), and Atlas A5 (only supports CANN 9.0).

Platform: aarch64/x86

Supporting Software:
- Driver Ascend HDK 25.1.RC1.1, CANN Community Edition 8.5.0 and later versions (refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) to install the CANN development kit package, as well as the supporting firmware and drivers)
- Before installing CANN software, you need to install the relevant [dependency list](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0045.html)
- Python >= 3.9, Recommendation: Python 3.11
- PyTorch >= 2.8.0, torch-npu >= 2.8.0

### Quick Start

DeepEP-Ascend supports A2, A3 and A5 and needs to generate packages separately on each platform.

#### Compile and Build

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

#### Installation

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

### Architecture

DeepEP-Ascend employs a **strategy-based architecture** where communication implementations are abstracted into interchangeable strategies, selected via environment variables.

#### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Buffer** | `buffer.py` | Main entry point. Initializes communication buffer and delegates to strategy objects. |
| **NormalStrategy** | `ep_strategy.py` / `strategies/normal_strategy.py` | Normal mode dispatch/combine strategies (default, alltoall). |
| **LowLatencyStrategy** | `ep_strategy.py` / `strategies/low_latency_strategy.py` | Low-latency mode dispatch/combine strategies (default, ops, alltoall). |
| **EventOverlap** | `utils.py` | Event synchronization utility for async operations. |
| **FuseMode** | `buffer.py` | Enum for fused MoE computation modes. |

#### Strategy Selection

Strategies are configured via environment variables at Buffer initialization:

| Environment Variable | Value | Normal Strategy | Low-Latency Strategy |
|---------------------|-------|-----------------|---------------------|
| `DEEP_USE_MODE=default` | default | `DefaultNormalCommStrategy` (deep_ep_cpp custom ops) | `DefaultLowLatencyCommStrategy` (deep_ep_cpp custom ops) |
| `DEEP_USE_MODE=alltoall` | alltoall | `AlltoAllNormalCommStrategy` (torch.distributed alltoallv) | `AllToAllLowLatencyCommStrategy` (torch.distributed alltoall) |
| `DEEP_USE_MODE=ops` | ops | `DefaultNormalCommStrategy` (deep_ep_cpp custom ops) | `OpsLowLatencyCommStrategy` (torch_npu ops) |

> **Note**: Invalid env (e.g., `DEEP_USE_MODE=error`) will raise a `ValueError`.

### API Overview

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
| `clean_low_latency_buffer(...)` | — | Compatibility no-op in the current backend; kept for callers that invoke it before switching to low-latency mode. |

For detailed API documentation, see:
- [Normal Mode API (dispatch/combine)](doc/NORMAL_API.md)
- [Low-Latency Mode API](doc/LOW_LATENCY_API.md)
- [get_dispatch_layout API](doc/GET_DISPATCH_LAYOUT_API.md)
- [Fused Deep MoE API](doc/FUSED_DEEP_MOE.md)

### Communication Modes

#### Normal Mode (Prefill / Training)

High-throughput dispatch and combine for training and prefill phases:
- **A3**: Pure HCCS intranode communication, full-mesh HCCS internode communication. No hierarchical implementation needed.
- **A2 Intranode**: Pure HCCS communication, supports up to `bs=8000` for normal dispatch/combine.
- **A2 Internode**: Hierarchical (HCCS intranode + RDMA internode) or non-hierarchical (pure RDMA) implementation. Supports up to `bs=4096`.

#### Low-Latency Mode (Decode)

Optimized for inference with small batch sizes (128 tokens/batch):
- **A3**: Supports `default`, `ops`, and `alltoall` strategies. `ops` strategy supports `comm_alg` options: `hierarchy`, `fullmesh_v1`, `fullmesh_v2`, `ccu`.
- **A5**: Supports `default` and `ops` strategies with MXFP8 per-block quantization (`use_ue8m0=True`, quant_mode=3). Data format: `float8_e4m3fn` + `float8_e8m0fnu` scales.
- **A2 Intranode**: Supports up to `bs=512` for low_latency dispatch/combine.
- **A2 Internode**: Hierarchical (HCCS + RDMA) or non-hierarchical (pure RDMA) implementation. Supports up to `bs=512`.

Quantization modes in low_latency_dispatch:
- **BF16**: `use_fp8=False` — no quantization, bfloat16 communication.
- **MXFP8 per-block**: `use_fp8=True, use_ue8m0=True` — per-block MXFP8 quantization (quant_mode=3), `float8_e4m3fn` data + `float8_e8m0fnu` scales (A5 only).

### Fused MoE

The `fused_deep_moe` API fuses dispatch + expert FFN computation + combine into a single operator call, significantly reducing communication overhead and end-to-end latency.

Two fuse modes are available via the `FuseMode` enum:
- `FuseMode.FUSED_DEEP_MOE` (default): Full fusion of dispatch + FFN + combine.
- `FuseMode.DISPATCH_FFN_COMBINE`: Dispatch + FFN + combine with separate dispatch handling.

Quantization modes (`quant_mode`):
- `1`: INT8 quantization (default)
- FP8 will be supported in A5 release.

See [Fused Deep MoE API](doc/FUSED_DEEP_MOE.md) for details.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_USE_MODE` | `default` | Normal mode strategy and Low-latency mode strategy: `default`, `ops`, or `alltoall`. |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | **Deprecated.** Enable INT8 quantization in normal dispatch. A2 internode does NOT support quantization in normal mode. MXFP8 per-block quantization (A5 only) is triggered by passing a tuple `(float8_e4m3fn_tensor, float8_e8m0fnu_tensor)` as `x`. |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | Dispatch return type for `num_recv_tokens_per_expert_list`: `1` = per-expert token count, `0` = prefix sum. |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | Number of shared expert ranks (used by ops strategy). |
| `HCCL_BUFFSIZE` | `200` (MB) | HCCL buffer size in MB. **Must be set** when using DeepEP on A2. |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | Set to `1` for A2 dual-node hierarchical communication. |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | Set to `0` for A2 dual-node hierarchical communication. |
| `HCCL_OP_EXPANSION_MODE` | — | **Must be disabled** on A2 when using DeepEP (remove or unset this variable). |

### Platform-Specific Notes

#### A2 Single Node

- Applicable when P/D node ranks = 8 (supports PD separation or mixed deployment).
- Not recommended when ranks < 8 (insufficient parallelism for EP benefits).
- **Performance limits**: normal up to `bs=8000`, low_latency up to `bs=512`.
- **Must set** `HCCL_BUFFSIZE` (e.g., `export HCCL_BUFFSIZE=1024`).
- **Must disable** `HCCL_OP_EXPANSION_MODE`.

For detailed A2 usage, see [A2_DEEPEP](doc/A2_DEEPEP.md).

#### A2 Dual Node

- Applicable when P/D node ranks > 8 (cross-node communication).
- **Normal mode does NOT support quantization** (`DEEP_NORMAL_MODE_USE_INT8_QUANT=0`).
- **Must set** `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0` for hierarchical communication.
- **Performance limits**: normal up to `bs=4096`, low_latency up to `bs=512`.

#### A3

- Pure HCCS communication for both intranode and internode. No hierarchical implementation needed.
- Supports `ops` strategy with multiple `comm_alg` options for low-latency mode.

#### A5

- Only supports CANN 9.0.
- Build with: `bash build.sh -a deepep Ascend950`.

### Test

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

### FAQ

1. If installing the `.whl` file results in the inability to import `deep_ep` in the project, check whether it is correctly installed in the `site-packages` directory of the current Python environment:
```
pip show deep-ep
```

2. If after installing the `.whl`, you encounter an issue where `deep_ep_cpp` is not found, you need to create a symbolic link of the `deep_ep_cpp*.so` files from the `site-packages/deep_ep` directory to the `site-packages` directory. Execute the following command in the `site-packages` directory:
```
ln -s deep_ep/deep_ep_cpp*.so
```

3. If you get a `ValueError` about unsupported mode combination, check that `DEEP_USE_MODE` is set to a valid value (`default`, `ops`, `alltoall`). See the [Strategy Selection](#strategy-selection) table for valid combinations.

4. On A2, always set `HCCL_BUFFSIZE` before running DeepEP. Missing this will cause dispatch/combine operators to fail.

---

<a id="中文"></a>

## 中文

### 介绍

**DeepEP-Ascend** 是 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的 Ascend NPU 实现，为 MoE（混合专家）模型提供高度优化的专家并行（EP）通信内核。它支持两种通信模式：

- **Normal 模式**：高吞吐的 dispatch 和 combine 操作，适用于训练和 Prefill 阶段。
- **Low-Latency 模式**：针对小 batch 生产推理优化，延迟低于 150us。

DeepEP-Ascend 采用**策略式架构**，通过环境变量灵活选择通信实现方式，支持多种硬件拓扑（A2、A3、A5）和通信后端（HCCS、RDMA、AlltoAll）。

### 软硬件配套说明

硬件型号支持：Atlas A2、A3 系列产品能适配 CANN 8.5 和 CANN 9.0，Atlas A5 只能适配 CANN 9.0。

平台：aarch64/x86

配套软件：
- 驱动 Ascend HDK 25.1.RC1.1、CANN社区版 8.5.0 及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装 CANN 开发套件包以及配套固件和驱动）
- 安装 CANN 软件前需安装相关[依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0045.html)
- Python >= 3.9，推荐 Python 3.11
- PyTorch >= 2.8.0, torch-npu >= 2.8.0

### 快速上手

DeepEP-Ascend 支持 A2、A3 和 A5，需要在各平台上分别生成包。

#### 编译构建

1、准备 CANN 的环境变量（根据安装路径修改）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、构建项目
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

#### 安装

1、执行 pip 安装命令，将 `.whl` 安装到你的 Python 环境下
```bash
pip install output/deep_ep*.whl

# 设置 deep_ep_cpp*.so 的软链接
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -

# （可选）确认是否可以成功导入
python -c "import deep_ep; print(deep_ep.__path__)"
```

2、执行 CANN 的环境变量（根据安装路径修改）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3、在 Python 工程中导入 `deep_ep`
```python
import deep_ep
```

### 架构

DeepEP-Ascend 采用**策略式架构**，通信实现被抽象为可互换的策略，通过环境变量进行选择。

#### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **Buffer** | `buffer.py` | 主入口。初始化通信缓冲区并委托给策略对象。 |
| **NormalStrategy** | `ep_strategy.py` / `strategies/normal_strategy.py` | Normal 模式 dispatch/combine 策略（default、alltoall）。 |
| **LowLatencyStrategy** | `ep_strategy.py` / `strategies/low_latency_strategy.py` | Low-latency 模式 dispatch/combine 策略（default、ops、alltoall）。 |
| **EventOverlap** | `utils.py` | 异步操作的事件同步工具。 |
| **FuseMode** | `buffer.py` | 融合 MoE 计算模式的枚举。 |

#### 策略选择

策略在 Buffer 初始化时通过环境变量配置：

| 环境变量组合 | Normal 策略 | Low-Latency 策略 |
|-------------|------------|-----------------|
| `DEEP_USE_MODE=default` | `DefaultNormalCommStrategy`（deep_ep_cpp 自定义算子） | `DefaultLowLatencyCommStrategy`（deep_ep_cpp 自定义算子） |
| `DEEP_USE_MODE=alltoall` | `AlltoAllNormalCommStrategy`（torch.distributed alltoallv） | `AllToAllLowLatencyCommStrategy`（torch.distributed alltoall） |
| `DEEP_USE_MODE=ops` | `DefaultNormalCommStrategy`（deep_ep_cpp 自定义算子） | `OpsLowLatencyCommStrategy`（torch_npu 算子） |

> **注意**：无效配置（如 `DEEP_USE_MODE=error`）会抛出 `ValueError`。

### API 总览

`Buffer` 类是主要接口，核心 API 概览如下：

| API | 模式 | 说明 |
|-----|------|------|
| `Buffer(group, num_nvl_bytes, num_rdma_bytes, ...)` | — | 初始化通信缓冲区并选择策略。 |
| `get_dispatch_layout(topk_idx, num_experts, ...)` | Normal | 计算后续 dispatch 所需的布局信息。返回 `num_tokens_per_rank`、`num_tokens_per_rdma_rank`、`num_tokens_per_expert`、`is_token_in_rank`。 |
| `dispatch(x, topk_idx, topk_weights, ...)` | Normal | 将 token 分发到专家 rank。返回接收的 token、topk 信息及 combine 所需的 handle。 |
| `combine(x, handle, ...)` | Normal | 归约 dispatch 返回的 token。必须使用 `dispatch` 返回的 handle。 |
| `low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, ...)` | Low-Latency | 低时延 token 分发，用于 Decode 阶段。 |
| `low_latency_combine(x, topk_idx, topk_weights, handle, ...)` | Low-Latency | 低时延 token 归约，用于 Decode 阶段。 |
| `fused_deep_moe(x, topk_idx, topk_weights, ...)` | 融合 | 一次调用完成 dispatch + FFN + combine。 |
| `get_dispatch_config(num_ranks)` | — | 获取推荐 Normal dispatch 配置。 |
| `get_combine_config(num_ranks)` | — | 获取推荐 Normal combine 配置。 |
| `clean_low_latency_buffer(...)` | — | 当前后端实现为空操作，用于兼容从 Normal 模式切换到 Low-Latency 模式前调用该接口的代码。 |

详细 API 文档请参考：
- [Normal 模式 API（dispatch/combine）](doc/NORMAL_API.md)
- [Low-Latency 模式 API](doc/LOW_LATENCY_API.md)
- [get_dispatch_layout API](doc/GET_DISPATCH_LAYOUT_API.md)
- [融合 Deep MoE API](doc/FUSED_DEEP_MOE.md)

### 通信模式

#### Normal 模式（Prefill / 训练）

高吞吐的 dispatch 和 combine，适用于训练和 Prefill 阶段：
- **A3**：纯 HCCS 节点内通信，全互联 HCCS 节点间通信。无需分层实现。
- **A2 单机**：纯 HCCS 通信，normal dispatch/combine 最大支持 `bs=8000`。
- **A2 双机**：分层（节点内 HCCS + 节点间 RDMA）或不分层（纯 RDMA）实现。最大支持 `bs=4096`。

#### Low-Latency 模式（Decode）

针对小 batch 推理优化（128 tokens/batch）：
- **A3**：支持 `default`、`ops`、`alltoall` 策略。`ops` 策略支持 `comm_alg` 选项：`hierarchy`、`fullmesh_v1`、`fullmesh_v2`、`ccu`。
- **A5**：支持 `default` 和 `ops` 策略，支持 MXFP8 per-block 量化（`use_ue8m0=True`，quant_mode=3）。数据格式：`float8_e4m3fn` + `float8_e8m0fnu` 缩放因子。
- **A2 单机**：low_latency dispatch/combine 最大支持 `bs=512`。
- **A2 双机**：分层（HCCS + RDMA）或不分层（纯 RDMA）实现。最大支持 `bs=512`。

low_latency_dispatch 量化模式：
- **BF16**：`use_fp8=False` — 不量化，bfloat16 通信。
- **MXFP8 per-block**：`use_fp8=True, use_ue8m0=True` — per-block MXFP8 量化（quant_mode=3），`float8_e4m3fn` 数据 + `float8_e8m0fnu` 缩放因子（仅 A5）。

### 融合 MoE

`fused_deep_moe` API 将 dispatch + 专家 FFN 计算 + combine 融合为单次算子调用，显著降低通信开销和端到端延迟。

通过 `FuseMode` 枚举提供两种融合模式：
- `FuseMode.FUSED_DEEP_MOE`（默认）：dispatch + FFN + combine 完整融合。
- `FuseMode.DISPATCH_FFN_COMBINE`：dispatch + FFN + combine，dispatch 分离处理。

量化模式（`quant_mode`）：
- `1`：INT8 量化（默认）
- FP8 将在 A5 版本中支持。

详见 [融合 Deep MoE API](doc/FUSED_DEEP_MOE.md)。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEP_USE_MODE` | `default` | Normal 模式策略 and Low-latency 模式策略：`default`、`ops` 或 `alltoall`。 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | **已弃用。** 在 Normal dispatch 中启用 INT8 量化。A2 双机 Normal 模式**不支持**量化。MXFP8 per-block 量化（仅 A5）通过传入 tuple `(float8_e4m3fn_tensor, float8_e8m0fnu_tensor)` 作为 `x` 参数触发。 |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | dispatch 返回的 `num_recv_tokens_per_expert_list` 类型：`1` = 各专家 token 数，`0` = 前缀和。 |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | 共享专家 rank 数（ops 策略使用）。 |
| `HCCL_BUFFSIZE` | `200`（MB） | HCCL 缓冲区大小（MB）。A2 使用 DeepEP 时**必须设置**。 |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | A2 双机分层通信时设为 `1`。 |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | A2 双机分层通信时设为 `0`。 |
| `HCCL_OP_EXPANSION_MODE` | — | A2 使用 DeepEP 时**必须禁用**（移除或取消设置此变量）。 |

### 平台特定说明

#### A2 单机

- 适用条件：P/D 节点 ranks = 8（支持 PD 分离或混部）。
- ranks < 8时不推荐开启 DeepEP（并行度不足，EP优化收益有限）。
- **性能上限**：normal 最大 `bs=8000`，low_latency 最大 `bs=512`。
- **必须设置** `HCCL_BUFFSIZE`（如 `export HCCL_BUFFSIZE=1024`）。
- **必须禁用** `HCCL_OP_EXPANSION_MODE`。

详细 A2 使用说明请参考 [A2_DEEPEP](doc/A2_DEEPEP.md)。

#### A2 双机

- 适用条件：P/D 节点 ranks > 8（跨节点通信）。
- **Normal 模式不支持量化**（`DEEP_NORMAL_MODE_USE_INT8_QUANT=0`）。
- **必须设置** `HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0` 启用分层通信。
- **性能上限**：normal 最大 `bs=4096`，low_latency 最大 `bs=512`。

#### A3

- 纯 HCCS 通信（节点内和节点间）。无需分层实现。
- Low-latency 模式支持 `ops` 策略及多种 `comm_alg` 选项。

#### A5

- 仅支持 CANN 9.0。
- 构建命令：`bash build.sh -a deepep Ascend950`。

### 测试

执行 DeepEP 相关测试脚本：

```bash
python3 tests/python/deepep/test_fused_deep_moe.py
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py

# A2 单机测试
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8

# A2 双机跨节点测试（需先设置 run_test_internode.sh 中的主节点 IP）
bash tests/python/deepep/run_test_internode.sh
```

### 常见问题

1、如果安装 `.whl` 后，在工程中 `import deep_ep` 出现找不到 `deep_ep` 库，则检查是否正确安装到当前 Python 环境的 `site-packages` 目录下；
查看安装路径：
```
pip show deep-ep
```

2、如果安装 `.whl` 后，出现找不到 `deep_ep_cpp`，则需要将 `site-packages/deep_ep` 目录下的 `deep_ep_cpp*.so` 文件软链接到 `site-packages` 目录下；
在 `site-packages` 目录下执行：
```
ln -s deep_ep/deep_ep_cpp*.so
```

3、如果遇到 `ValueError` 提示不支持的模式组合，请检查 `DEEP_USE_MODE` 是否为有效值（`default`、`ops`、`alltoall`）。参见[策略选择](#策略选择)表格。

4、在 A2 上运行 DeepEP 前，必须设置 `HCCL_BUFFSIZE`，否则 dispatch/combine 算子会报错。
