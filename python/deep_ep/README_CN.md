<div align="center">

# DeepEP-Ascend

**Ascend NPU 上高性能 MoE 专家并行通信库**

[![平台](https://img.shields.io/badge/平台-A2%20%7C%20A3%20%7C%20A5-blue)]()
[![CANN](https://img.shields.io/badge/CANN-8.5%2B%20%7C%209.0-green)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)]()

[English](README.md) | 简体中文

[Normal API](doc/NORMAL_API.md) · [Low-Latency API](doc/LOW_LATENCY_API.md) · [Fused MoE](doc/FUSED_DEEP_MOE.md) · [平台指南](doc/PLATFORM_GUIDE.md)

</div>

---

## 为什么选择 DeepEP-Ascend？

DeepEP-Ascend 是 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的 Ascend NPU 实现，为 Atlas 硬件上的 MoE 模型提供优化的专家并行（EP）通信。

**三种运行模式，一个库：**

- 🚀 **Normal** — 高吞吐 dispatch/combine，适用于训练和 Prefill
- ⚡ **Low-Latency** — 低时延（<150us）decode 推理
- 🔥 **Fused MoE** — dispatch + FFN + combine 融合单算子调用（仅 A3）

**核心特性：**

- 策略式架构（`default`、`ops`、`alltoall`），通过 `DEEP_USE_MODE` 选择
- 多量化支持：BF16、INT8 per-token、MXFP8 per-block（A5）
- 支持 A2、A3、A5（C310）平台

## 平台能力

| | A2 单机 | A2 双机 | A3 | A5 (C310) |
|:---|:---:|:---:|:---:|:---:|
| Normal dispatch/combine | ✅ | ✅（不支持量化） | ✅ | ✅ |
| Low-latency dispatch/combine | ✅ | ✅ | ✅ | ✅ |
| Fused MoE | ❌ | ❌ | ✅ INT8 | 🔜 计划中 |
| INT8 per-token 量化 | ✅ | ✅（仅 LL） | ✅ | ✅ |
| MXFP8 per-block 量化 | ❌ | ❌ | ❌ | ✅ |
| Normal 最大 BS | 8000 | 4096 | 8192（长序列 32k） | — |
| Low-latency 最大 BS | 512 | 512 | 512 | 512 |

> [!NOTE]
> 量化详情见 [Normal API](doc/NORMAL_API.md) 和 [Low-Latency API](doc/LOW_LATENCY_API.md)。平台部署见 [平台指南](doc/PLATFORM_GUIDE.md)。

## 快速上手

### 编译构建

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# A5 (C310)
bash build.sh -a deepep Ascend950

# A3
bash build.sh -a deepep

# A2
bash build.sh -a deepep2
```

### 安装与验证

```bash
pip install output/deep_ep*.whl

# 设置 deep_ep_cpp 软链接
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" \
  && ln -s deep_ep/deep_ep_cpp*.so && cd -

# 验证安装
python -c "import deep_ep; print(deep_ep.__path__)"
```

### 环境要求

| | A2 | A3 | A5 |
|:---|:---|:---|:---|
| CANN | 8.5+ | 8.5+ / 9.0+ | 9.0 |
| 驱动 | HDK 25.1.RC1.1+ | HDK 25.1.RC1.1+ | HDK 25.1.RC1.1+ |
| Python | ≥ 3.9（推荐 3.11） | ≥ 3.9（推荐 3.11） | ≥ 3.9（推荐 3.11） |
| PyTorch | ≥ 2.8.0, torch-npu ≥ 2.8.0 | ≥ 2.8.0, torch-npu ≥ 2.8.0 | ≥ 2.8.0, torch-npu ≥ 2.8.0 |

## 架构

策略选择由**单一**环境变量 `DEEP_USE_MODE` 控制，其值同时决定 Normal 和 Low-Latency 策略：

| `DEEP_USE_MODE` | Normal 策略 | Low-Latency 策略 |
|:---|:---|:---|
| `default` | DefaultNormalCommStrategy（自定义算子） | DefaultLowLatencyCommStrategy（自定义算子） |
| `ops` | DefaultNormalCommStrategy（自定义算子） | OpsLowLatencyCommStrategy（torch_npu 算子） |
| `alltoall` | AlltoAllNormalCommStrategy（torch.distributed） | AllToAllLowLatencyCommStrategy（torch.distributed） |

> [!WARNING]
> 无效 `DEEP_USE_MODE` 值会抛出 `ValueError`。`Buffer.__init__` 的 `normal_strategy`/`low_latency_strategy` 参数目前被 `DEEP_USE_MODE` 覆盖——传入自定义值无效。

`ops` 策略支持 `comm_alg` 选项，详见 [Low-Latency API](doc/LOW_LATENCY_API.md)。

## API 总览

| API | 模式 | 说明 | 文档 |
|:---|:---|:---|:---|
| `get_dispatch_layout` | Normal | 计算 dispatch 布局 | [API](doc/GET_DISPATCH_LAYOUT_API.md) |
| `dispatch` / `combine` | Normal | 高吞吐 token 分发与归约 | [API](doc/NORMAL_API.md) |
| `low_latency_dispatch` / `low_latency_combine` | Low-Latency | 低时延 token 分发与归约 | [API](doc/LOW_LATENCY_API.md) |
| `fused_deep_moe` | Fused | dispatch + FFN + combine（仅 A3） | [API](doc/FUSED_DEEP_MOE.md) |

## 环境变量

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `DEEP_USE_MODE` | `default` | 策略选择：`default`、`ops` 或 `alltoall`，同时控制 Normal 和 Low-Latency。 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `0` | ⛔ **已弃用但仍生效。** 启用 Normal dispatch INT8 per-token 量化。建议使用 tuple 输入替代。 |
| `MOE_EXPERT_TOKEN_NUMS_TYPE` | `1` | `1` = 各专家 token 数，`0` = 前缀和。 |
| `MOE_SHARED_EXPERT_RANK_NUM` | `0` | 共享专家 rank 数（ops 策略使用）。 |

<details>
<summary><b>HCCL 变量（仅 A2）</b></summary>

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HCCL_BUFFSIZE` | `200`（MB） | HCCL 缓冲区大小（MB）。A2 **必须设置**（如 `1024`），默认 200MB 不够。 |
| `HCCL_INTRA_PCIE_ENABLE` | `0` | A2 双机分层通信时设 `1`。 |
| `HCCL_INTRA_ROCE_ENABLE` | `1` | A2 双机分层通信时设 `0`。 |
| `HCCL_OP_EXPANSION_MODE` | — | A2 **必须禁用**（移除或 unset）。 |

</details>

## 测试

```bash
# Normal + Low-Latency（全平台）
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_low_latency.py

# Fused MoE（仅 A3）
python3 tests/python/deepep/test_fused_deep_moe.py

# A2 单机（8 进程）
python3 tests/python/deepep/test_intranode.py --num-processes=8
python3 tests/python/deepep/test_low_latency.py --num-processes=8
python3 tests/python/deepep/test_normal_and_low_latency.py --num-processes=8

# A2 双机（先设置 run_test_internode.sh 中的主节点 IP）
bash tests/python/deepep/run_test_internode.sh
```

## 常见问题

1. **导入失败** — 检查安装：`pip show deep-ep`
2. **`deep_ep_cpp` 找不到** — 创建软链接：`cd site-packages && ln -s deep_ep/deep_ep_cpp*.so`
3. **A2 运行错误** — 必须先设置 `HCCL_BUFFSIZE`
4. **策略不生效** — 检查 `DEEP_USE_MODE` 是否为有效值（`default`、`ops`、`alltoall`）。无效值会抛 `ValueError`。
