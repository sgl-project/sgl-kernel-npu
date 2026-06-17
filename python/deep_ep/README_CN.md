<h2 align="left">
DeepEP-Ascend
</h2>

<p align="left">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>


## 介绍

**DeepEP-Ascend** 是 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的 Ascend NPU 实现，为 MoE（混合专家）模型提供高度优化的专家并行（EP）通信内核。它支持两种通信模式：

- **Normal 模式**：高吞吐的 dispatch 和 combine 操作，适用于训练和 Prefill 阶段。
- **Low-Latency 模式**：针对小 batch 生产推理优化，延迟低于 150us。

DeepEP-Ascend 采用**策略式架构**，通过环境变量灵活选择通信实现方式，支持多种硬件拓扑（A2、A3、A5）和通信后端（HCCS、RDMA、AlltoAll）。


## 软硬件配套说明

硬件型号支持：Atlas A2、A3 系列产品能适配 CANN 8.5 和 CANN 9.0，Atlas A5 只能适配 CANN 9.0。

平台：aarch64/x86

配套软件：
- 驱动 Ascend HDK 25.1.RC1.1、CANN社区版 8.5.0 及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装 CANN 开发套件包以及配套固件和驱动）
- 安装 CANN 软件前需安装相关[依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0045.html)
- Python >= 3.9，推荐 Python 3.11
- PyTorch >= 2.8.0，torch-npu >= 2.8.0


## 快速上手

DeepEP-Ascend 支持 A2、A3 和 A5，需要在各平台上分别生成包。

### 编译构建

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

### 安装

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


## 架构

DeepEP-Ascend 采用**策略式架构**，通信实现被抽象为可互换的策略，通过环境变量进行选择。

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **Buffer** | `buffer.py` | 主入口。初始化通信缓冲区并委托给策略对象。 |
| **NormalStrategy** | `ep_strategy.py` / `strategies/normal_strategy.py` | Normal 模式 dispatch/combine 策略（default、alltoall）。 |
| **LowLatencyStrategy** | `ep_strategy.py` / `strategies/low_latency_strategy.py` | Low-latency 模式 dispatch/combine 策略（default、ops、alltoall）。 |
| **EventOverlap** | `utils.py` | 异步操作的事件同步工具。 |
| **FuseMode** | `buffer.py` | 融合 MoE 计算模式的枚举。 |

### 策略选择

策略在 Buffer 初始化时通过环境变量配置：

| 环境变量组合 | Normal 策略 | Low-Latency 策略 |
|-------------|------------|-----------------|
| `DEEP_NORMAL_MODE=default` | `DefaultNormalCommStrategy`（deep_ep_cpp 自定义算子） | `DefaultLowLatencyCommStrategy`（deep_ep_cpp 自定义算子） |
| `DEEP_NORMAL_MODE=alltoall` | `AlltoAllNormalCommStrategy`（torch.distributed alltoallv） | `AllToAllLowLatencyCommStrategy`（torch.distributed alltoall） |
| `DEEP_NORMAL_MODE=default` | `DefaultNormalCommStrategy`（deep_ep_cpp 自定义算子） | `OpsLowLatencyCommStrategy`（torch_npu 算子） |

> **注意**：无效配置（如 `DEEP_NORMAL_MODE=error`）会抛出 `ValueError`。


## API 总览

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
- [融合 Deep MoE API（中文）](doc/FUSED_DEEP_MOE_CN.md) | [English](doc/FUSED_DEEP_MOE_EN.md)


## 通信模式

### Normal 模式（Prefill / 训练）

高吞吐的 dispatch 和 combine，适用于训练和 Prefill 阶段：
- **A3**：纯 HCCS 节点内通信，全互联 HCCS 节点间通信。无需分层实现。
- **A2 单机**：纯 HCCS 通信，normal dispatch/combine 最大支持 `bs=8000`。
- **A2 双机**：分层（节点内 HCCS + 节点间 RDMA）或不分层（纯 RDMA）实现。最大支持 `bs=4096`。

### Low-Latency 模式（Decode）

针对小 batch 推理优化（128 tokens/batch）：
- **A3**：支持 `default`、`ops`、`alltoall` 策略。`ops` 策略支持 `comm_alg` 选项：`hierarchy`、`fullmesh_v1`、`fullmesh_v2`、`ccu`。
- **A2 单机**：low_latency dispatch/combine 最大支持 `bs=512`。
- **A2 双机**：分层（HCCS + RDMA）或不分层（纯 RDMA）实现。最大支持 `bs=512`。


## 融合 MoE

`fused_deep_moe` API 将 dispatch + 专家 FFN 计算 + combine 融合为单次算子调用，显著降低通信开销和端到端延迟。

通过 `FuseMode` 枚举提供两种融合模式：
- `FuseMode.FUSED_DEEP_MOE`（默认）：dispatch + FFN + combine 完整融合。
- `FuseMode.DISPATCH_FFN_COMBINE`：dispatch + FFN + combine，dispatch 分离处理。

量化模式（`quant_mode`）：
- `1`：INT8 量化（默认）
- FP8 将在 A5 版本中支持。

详见 [融合 Deep MoE API](doc/FUSED_DEEP_MOE_CN.md)。


## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEP_USE_MODE` | `default` | Normal 模式策略 and Low-latency 模式策略：`default`、`ops` 或 `alltoall`。 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | 在 Normal dispatch 中启用 INT8 量化。A2 双机 Normal 模式**不支持**量化。 |
| `SGLANG_DEEPEP_BF16_DISPATCH` | `0` | 在 low_latency_dispatch 中关闭量化（BF16 dispatch）。设为 `1` 关闭量化；仅在 Decode 阶段生效。 |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | dispatch 返回的 `num_recv_tokens_per_expert_list` 类型：`1` = 各专家 token 数，`0` = 前缀和。 |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | 共享专家 rank 数（ops 策略使用）。 |
| `HCCL_BUFFSIZE` | `200`（MB） | HCCL 缓冲区大小（MB）。A2 使用 DeepEP 时**必须设置**。 |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | A2 双机分层通信时设为 `1`。 |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | A2 双机分层通信时设为 `0`。 |
| `HCCL_OP_EXPANSION_MODE` | — | A2 使用 DeepEP 时**必须禁用**（移除或取消设置此变量）。 |
| `DEBUG_MODE` | `OFF` | 设为 `ON` 启用 DEBUG 日志用于参数追踪。 |


## 平台特定说明

### A2 单机

- 适用条件：P/D 节点 ranks = 8（支持 PD 分离或混部）。
- ranks < 8时不推荐开启 DeepEP（并行度不足，EP优化收益有限）。
- **性能上限**：normal 最大 `bs=8000`，low_latency 最大 `bs=512`。
- **必须设置** `HCCL_BUFFSIZE`（如 `export HCCL_BUFFSIZE=1024`）。
- **必须禁用** `HCCL_OP_EXPANSION_MODE`。

详细 A2 使用说明请参考 [A2_DEEPEP_CN.md](A2_DEEPEP_CN.md)。

### A2 双机

- 适用条件：P/D 节点 ranks > 8（跨节点通信）。
- **Normal 模式不支持量化**（`DEEP_NORMAL_MODE_USE_INT8_QUANT=0`）。
- **必须设置** `HCCL_INTRA_PCIE_ENABLE=1` 和 `HCCL_INTRA_ROCE_ENABLE=0` 启用分层通信。
- **性能上限**：normal 最大 `bs=4096`，low_latency 最大 `bs=512`。

### A3

- 纯 HCCS 通信（节点内和节点间）。无需分层实现。
- Low-latency 模式支持 `ops` 策略及多种 `comm_alg` 选项。

### A5

- 仅支持 CANN 9.0。
- 构建命令：`bash build.sh -a deepep Ascend950`。


## 测试

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


## 常见问题

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

3、如果遇到 `ValueError` 提示不支持的模式组合，请检查 `DEEP_NORMAL_MODE` 和 `DEEP_LOW_LATENCY_MODE` 是否为有效组合。参见[策略选择](#策略选择)表格。
