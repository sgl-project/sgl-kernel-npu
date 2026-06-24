# A2 DeepEP Guide

<div align="center">

[![Platform](https://img.shields.io/badge/Platform-A2%20only-red)]()

</div>

> [!IMPORTANT]
> **A2 only.** This document applies exclusively to Atlas A2 series.

English | [中文](#中文)

---

## English

### Software and Hardware

Supported Hardware: Atlas A2 series
Platform: aarch64/x86
Supporting Software:
- Driver Ascend HDK ≥ 25.3.RC1, CANN ≥ 8.5.0

### Build

Execute the project build script `build.sh`:
```bash
# Building Project, deepep2 for A2 package
bash build.sh -a deepep2
```
After building, a `deep_ep` whl package will be generated in the `output` directory.

### Install

1. Pip install the `.whl` file into your Python environment:
```bash
pip install output/deep_ep*.whl

# Link to deep_ep_cpp*.so
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so && cd -

# (Optional) Confirm successful import
python -c "import deep_ep; print(deep_ep.__path__)"
```
> [!TIP]
> If you skip the symlink step, you will get a "deep_ep_cpp.so not found" error at runtime.

### Usage

DeepEP provides the following core APIs:

| API | Phase | Feature |
|-----|-------|---------|
| `dispatch` | Prefill | High throughput, also called normal_dispatch |
| `combine` | Prefill | High throughput, also called normal_combine, paired with `dispatch` |
| `low_latency_dispatch` | Decode | Low latency, optimized for Decode |
| `low_latency_combine` | Decode | Low latency, paired with `low_latency_dispatch` |

Framework configuration (SGLang):

| Node Type | Parameter | Recommended Value |
|-----------|-----------|-------------------|
| P node (Prefill) | `--deepep-mode` | `normal` |
| D node (Decode) | `--deepep-mode` | `low_latency` |
| Mixed PD node | `--deepep-mode` | `auto` |

**Note**: DeepEP A2 only supports HCCL communication domain. When DeepEP is enabled, `HCCL_BUFFSIZE` **must** be set, otherwise dispatch & combine operators will fail:
```bash
# Adjust size based on your model scenario
export HCCL_BUFFSIZE=1024
```

On A2, you must **disable** `HCCL_OP_EXPANSION_MODE`, otherwise unknown operator errors will occur:
```bash
# Remove this environment variable on A2
# export HCCL_OP_EXPANSION_MODE=AIV
```

#### A2 Single Node

**Applicable when**: P/D node ranks = 8 (supports PD separation or mixed deployment).

**Not recommended when**: ranks < 8 — insufficient parallelism for EP optimization benefits.

**Performance limits**:
- normal dispatch & combine: up to `bs=8000`
- low_latency dispatch & combine: up to `bs=512`

(Optional) Enable quantization in Prefill phase:
```bash
# Quantization in dispatch phase: bfloat16 → int8
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
```

(Optional) Set the return type of `num_recv_tokens_per_expert_list` from dispatch:
```bash
# Unset or set to 1: per-expert token count; set to 0: prefix sum
export MOE_EXPERT_TOKEN_NUMS_TYPE=0
```

#### Single-Operator Test

```bash
# Normal single-operator test
python3 tests/python/deepep/test_intranode.py --num-processes=8

# Low-latency single-operator test
python3 tests/python/deepep/test_low_latency.py --num-processes=8

# Normal + low_latency single-operator test
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8
```

#### A2 Dual Node

**Applicable when**: P/D node ranks > 8 (cross-node communication).

**Restriction**: Prefill phase does **not** support quantization. Disable it:
```bash
# Ensure this variable is unset or set to 0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=0
```

**Performance limits**:
- normal dispatch & combine: up to `bs=4096`
- low_latency dispatch & combine: up to `bs=512`

**(Required)** For hierarchical communication in dispatch & combine, set the following environment variables on **both** P and D nodes:
```bash
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
```

(Optional) Set the return type of `num_recv_tokens_per_expert_list` from dispatch:
```bash
export MOE_EXPERT_TOKEN_NUMS_TYPE=0
```

#### Dual-Node Test

For A2 dual-node cross-node communication tests (set the primary node IP in `run_test_internode.sh` first). Line 22 can be replaced with the desired test case (`test_internode.py`, `test_low_latency.py`):
```bash
cd tests/python/deepep/run_test_internode.sh

# Set the primary node IP in run_test_internode.sh first
bash run_test_internode.sh
```

---

<a id="中文"></a>

## 中文

### 软硬件配套说明

硬件型号支持：Atlas A2 系列产品
平台：aarch64/x86
配套软件
- 驾动 Ascend HDK ≥ 25.3.RC1、CANN ≥ 8.5.0

### 构建DeepEp包

执行工程构建脚本 build.sh
```bash
# Building Project, deepep2 for a2 package
bash build.sh -a deepep2
```
构建完成后将在`output`目录下生成deep_ep的whl包。

### 安装

1、执行pip安装命令，将`.whl`安装到你的python环境下
```bash
pip install output/deep_ep*.whl

# 设置deep_ep_cpp*.so的软链接
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so && cd -

# （可选）确认是否可以成功导入
python -c "import deep_ep; print(deep_ep.__path__)"
```
> [!TIP]
> 若未执行软链接，运行时将报错"找不到 deep_ep_cpp.so"。

### 使用

DeepEp 向上层提供以下核心接口：

| 接口名 | 适用阶段 | 特性 |
|--------|----------|------|
| `dispatch` | Prefill | 高吞吐，也作 normal_dispatch |
| `combine` | Prefill | 高吞吐，也作 normal_combine，与 `dispatch` 配套使用 |
| `low_latency_dispatch` | Decode | 低时延，专为 Decode 优化 |
| `low_latency_combine` | Decode | 低时延，与 `low_latency_dispatch` 配套使用 |

框架配置建议（SGLang）：

| 节点类型 | 参数 | 建议值 |
|----------|------|--------|
| P 节点（Prefill） | `--deepep-mode` | `normal` |
| D 节点（Decode） | `--deepep-mode` | `low_latency` |
| 混部节点（PD） | `--deepep-mode` | `auto` |

**注意**：当前deepep A2仅支持HCCL通信域通信，开启deepep后，必须设置的`HCCL_BUFFSIZE`大小，否则dispatch&combine算子会报错。
```bash
# 根据实际模型场景灵活调整大小
export HCCL_BUFFSIZE=1024
```

A2场景下叠加deepep，需**禁用**环境变量`HCCL_OP_EXPANSION_MODE`，否则会出现未知算子错误。
```bash
# A2下需要去除该环境变量
# export HCCL_OP_EXPANSION_MODE=AIV
```

#### A2单机

**适用条件**：P/D 节点 ranks = 8（支持 PD 分离或混部）

**不推荐启用场景**：当 ranks < 8 时不推荐开启 DeepEp，缺乏足够并行度，难以体现EP的优化收益。

**性能上限**：
- normal dispatch&combine：最大支持 `bs=8000`
- low_latency dispatch&combine：最大支持 `bs=512`

（可选）支持在Prefill阶段**开启**量化，设置环境变量：
```bash
# 在dispatch阶段会进行量化，bfloat16 --> int8
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
```

（可选）支持设置dispatch接口返回出参`num_recv_tokens_per_expert_list`类型，设置环境变量：
```bash
# 不设置或设置为1返回本卡各专家接收token数，设置为0返回前缀和
export MOE_EXPERT_TOKEN_NUMS_TYPE=0
```

#### 单算子测试

执行deepep相关测试脚本
```bash
# normal单算子测试
python3 tests/python/deepep/test_intranode.py --num-processes=8

# low_latency 单算子测试
python3 tests/python/deepep/test_low_latency.py --num-processes=8

# normal+low_latency 单算子测试
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8
```

#### A2双机

**适用条件**：P/D 节点 ranks > 8（跨节点通信）

**禁用限制**：Prefill 阶段 **不支持开启量化**，需禁用：
```bash
# 确保该变量未设置或设为 0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=0
```

**性能上限**：
- normal dispatch&combine：最大支持 `bs=4096`
- low_latency dispatch&combine：最大支持 `bs=512`

（必须）dispatch&combine算子使用分层通信，P/D都需要设置以下环境变量：
```bash
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
```

（可选）支持设置dispatch接口返回出参`num_recv_tokens_per_expert_list`类型，设置环境变量：
```bash
# 不设置或设置为1返回本卡各专家接收token数，设置为0返回前缀和
export MOE_EXPERT_TOKEN_NUMS_TYPE=0
```

#### 双机跨节点测试

在A2双机下执行，测试跨节点通信 (需要先设置run_test_internode.sh中的主节点IP)。
`line:22` 可以替换为需要执行的测试用例 (test_internode.py、test_low_latency.py)
```bash
cd tests/python/deepep/run_test_internode.sh

# 需要先设置run_test_internode.sh中的主节点IP
bash run_test_internode.sh
```
