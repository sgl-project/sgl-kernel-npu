# Fused Deep MoE API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Fused-purple)]()
[![Platform](https://img.shields.io/badge/Platform-A3%20only-red)]()
[![Quant](https://img.shields.io/badge/Quantization-INT8-yellow)]()

English | [中文](#中文)

</div>

> [!IMPORTANT]
> **A3 only.** This API is NOT available on A2 or A5. A5 FP8/MXFP8 support is planned for future release.

---

## English

### Introduction

In Mixture of Experts (MoE) models, the `fused_deep_moe` operator implements the hyper-fusion of Dispatch + Experts FFN (2×GMM) + Combine functionalities.
This operator completes token distribution, expert computation (matrix multiplication, activation, quantization/dequantization), and result aggregation in a single call. Compared with traditional multi-operator implementations, it significantly reduces communication overhead and end-to-end latency.
The communication latency (Batch size = 32 / 155μs, Dispatch = 80μs, Combine = 75μs) is reduced to less than 85μs, with a 70μs reduction in single-layer communication latency and a 4ms reduction in inference end-to-end latency.

Two fuse modes are available via the `FuseMode` enum:

| FuseMode | Value | CANN Operator | Description |
|----------|-------|---------------|-------------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | Full fusion: InitRouting + AllToAll + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Unpermute/Combine in a single AscendC kernel. |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | Separate dispatch handling: InitRouting + AllToAll dispatch + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Combine in a single AscendC kernel, using a different internal fusion strategy. |

> [!NOTE]
> `FuseMode` is **not** exported from the package's top-level `__init__.py`. Import it explicitly:
> ```python
> from deep_ep.buffer import FuseMode
> ```
> Or use integer values directly: `fuse_mode=1` (FUSED_DEEP_MOE) or `fuse_mode=2` (DISPATCH_FFN_COMBINE).

#### Key Differences Between Fuse Modes

| Aspect | `FUSED_DEEP_MOE` | `DISPATCH_FFN_COMBINE` |
|--------|-------------------|------------------------|
| **Weight scale dtype** | `float32` (auto-converted to `float` internally) | `int64` (float32 values reinterpreted as int64 bit patterns; **not auto-converted** — caller must perform the conversion manually) |
| **Weight layout (GMM1)** | Permuted via tile-N permutation (`reshape_fusion_gmm_weight` in test code) | Standard NZ format, no additional permutation needed |
| **`num_max_dispatch_tokens_per_rank`** semantics | Max tokens to dispatch per rank | Max tokens received in dispatch (i.e., `max_bs × num_ranks × topk`) |
| **Second return value** | `ep_recv_count`, shape `[num_local_experts × num_ranks]` | `expert_token_nums`, shape `[num_local_experts]` (per-rank only) |
| **Shared expert support** | Supported | **Not supported** |
| **BF16 weight support** | Available | **Not available** — only INT8 weights are supported currently |

### Python API

```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: torch.Tensor,
    gmm1_permuted_weight_scale: torch.Tensor,
    gmm2_weight: torch.Tensor,
    gmm2_weight_scale: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Parameter Description

| Parameter | Type | Shape | Description                                                                                                                                                                                                                                                                                                      |
|-----------|------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token (commonly `bfloat16`). **bs** range **[1, 256]**. **hidden** range **[512, 7168]**, must be divisible by **32**.                                                                                                                     |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token, `int32` type. A value of `-1` indicates the token is not dispatched.                                                                                                                                                                                                              |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`).                                                                                                                                                                                                                                               |
| **gmm1_permuted_weight** | `torch.Tensor` | e.g., `[G, 7168, 4096]` | First-stage (up-projection) expert weights. For `FUSED_DEEP_MOE`, tile-N permuted layout to fit Grouped MatMul (reference implementation `reshape_fusion_gmm_weight` in test code); for `DISPATCH_FFN_COMBINE`, standard NZ format without permutation.                                                          |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | e.g., `[G, 4096]` | Quantization scale for first-stage weights. For `FUSED_DEEP_MOE`, `float32` dtype (auto-converted internally); for `DISPATCH_FFN_COMBINE`, **`int64` dtype** (float32 scale values reinterpreted as int64 bit patterns — **not auto-converted** by the Python API, caller must perform the conversion manually). |
| **gmm2_weight** | `torch.Tensor` | e.g., `[G, 7168, 2048]` | Second-stage (down-projection) expert weights.                                                                                                                                                                                                                                                                   |
| **gmm2_weight_scale** | `torch.Tensor` | e.g., `[G, 7168]` | Quantization scale for second-stage weights. Same dtype rules as `gmm1_permuted_weight_scale`.                                                                                                                                                                                                                   |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | For `FUSED_DEEP_MOE`: maximum number of tokens to dispatch per rank, used for buffer/memory allocation. For `DISPATCH_FFN_COMBINE`: maximum number of tokens received in dispatch (typically `max_bs × num_ranks × topk`). All ranks must hold the same value.                                                   |
| **num_experts** | `int` | Scalar, range **(0, 512]** | Total number of global experts. Must be divisible by `(num_ranks - shared_expert_rank_num)`; otherwise the tiling check will reject it.                                                                                                                                                                          |
| **quant_mode** | `int` | Scalar, default `1` | Quantization mode: `0` = no quantization (BF16 weights), `1` = INT8 (default). FP8 will be supported in A5 release.                                                                                                                                                                                              |
| **fuse_mode** | `FuseMode` | Scalar, default `FuseMode.FUSED_DEEP_MOE` | Fuse mode selection. `FUSED_DEEP_MOE` (1): full fusion via `aclnnFusedDeepMoe`. `DISPATCH_FFN_COMBINE` (2): separate dispatch handling via `aclnnDispatchFFNCombine`.                                                                                                                                            |

### Constraints

- **bs** (batch size): range **[0, 256]**.
- **hidden**: range **[512, 7168]**, must be divisible by **32**.
- **num_topk** (topk): range **(0, 12]** — the kernel tiling enforces `topk ≤ 12`.
- **num_experts**: range **(0, 512]** — the kernel tiling enforces `moeExpertNum ≤ 512`. `num_experts` must be divisible by `(num_ranks - shared_expert_rank_num)`.

### Return Values

#### For `fuse_mode=FUSED_DEEP_MOE` (default)

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **output** | `torch.Tensor` | `[bs, hidden]` | Fused expert outputs. |
| **ep_recv_count** | `torch.Tensor` | `[num_local_experts × num_ranks]` | Number of tokens received by each expert across all ranks in the EP communication domain, used for subsequent communication synchronization or load balancing statistics. |

#### For `fuse_mode=DISPATCH_FFN_COMBINE`

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **output** | `torch.Tensor` | `[bs, hidden]` | Fused expert outputs. |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | Number of tokens received by each local expert on this rank, used for subsequent communication synchronization or load balancing statistics. |

---

<a id="中文"></a>

## 中文

### 介绍

在 MoE（Mixture of Experts，混合专家模型）中，`fused_deep_moe` 算子实现 Dispatch + Experts FFN (2×GMM) + Combine 的超融合功能。

该算子在一次调用中完成 token 分发、专家计算（矩阵乘、激活、量化/反量化）以及结果合并操作，相比传统多算子实现显著降低通信开销和端到端时延。

通信时长（Batch size = 32 / 155μs，Dispatch = 80μs，Combine = 75μs）降低到85μs以内，单层通信时长降低70μs，推理端到端时延降低4ms。

通过 `FuseMode` 枚举提供两种融合模式：

| FuseMode | 值 | CANN 算子 | 说明 |
|----------|---|-----------|------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | 完整融合：InitRouting + AllToAll + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Unpermute/Combine 在单个 AscendC kernel 中完成。 |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | 分离 dispatch 处理：InitRouting + AllToAll 分发 + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Combine 在单个 AscendC kernel 中完成，采用不同的内部融合策略。 |

> [!NOTE]
> `FuseMode` **未**从包顶层 `__init__.py` 导出，需显式导入：
> ```python
> from deep_ep.buffer import FuseMode
> ```
> 或直接使用整数值：`fuse_mode=1`（FUSED_DEEP_MOE）或 `fuse_mode=2`（DISPATCH_FFN_COMBINE）。

#### 两种融合模式的关键差异

| 方面 | `FUSED_DEEP_MOE` | `DISPATCH_FFN_COMBINE` |
|------|-------------------|------------------------|
| **权重 scale 数据类型** | `float32`（内部自动转换为 `float`） | `int64`（float32 scale 值重新解释为 int64 位模式；**不会自动转换** — 调用者需手动执行转换） |
| **GMM1 权重布局** | 经 tile-N 重排（参考实现 `reshape_fusion_gmm_weight` 在测试代码中） | 标准 NZ 格式，无需额外重排 |
| **`num_max_dispatch_tokens_per_rank`** 语义 | 每个 rank 最多分发的 token 数 | dispatch 中最多接收的 token 数（即 `max_bs × num_ranks × topk`） |
| **第二返回值** | `ep_recv_count`，形状 `[num_local_experts × num_ranks]` | `expert_token_nums`，形状 `[num_local_experts]`（仅本 rank） |
| **共享专家支持** | 支持 | **不支持** |

### Python API

```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: torch.Tensor,
    gmm1_permuted_weight_scale: torch.Tensor,
    gmm2_weight: torch.Tensor,
    gmm2_weight_scale: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数说明

| 参数 | 类型 | 形状 | 说明                                                                                                                                                                  |
|------|------|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | 输入 token 表示，每行一个 token 的隐藏向量（常用 `bfloat16`）。**bs** 取值范围 **[1, 256]**。**hidden** 取值范围 **[512, 7168]**，且必须能被 **32** 整除。                                               |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | 每个 token 的专家索引，`int32` 类型。若值为 `-1` 表示该 token 不分发。                                                                                                                   |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | 合并专家输出的加权系数（`float32`）。                                                                                                                                             |
| **gmm1_permuted_weight** | `torch.Tensor` | 例如 `[G, 7168, 4096]` | 第一阶段（上投）专家权重。`FUSED_DEEP_MOE` 模式下需做 tile-N permute 重排（参考实现 `reshape_fusion_gmm_weight` 在测试代码中）；`DISPATCH_FFN_COMBINE` 模式下使用标准 NZ 格式，无需额外重排。                         |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | 例如 `[G, 4096]` | 第一阶段权重量化 scale。`FUSED_DEEP_MOE` 模式下为 `float32` dtype（内部自动转换）；`DISPATCH_FFN_COMBINE` 模式下为 **`int64` dtype**（float32 scale 值重新解释为 int64 位模式 — **不会自动转换**，调用者需手动执行转换）。 |
| **gmm2_weight** | `torch.Tensor` | 例如 `[G, 7168, 2048]` | 第二阶段（下投）专家权重 （融合算子需要转置）。                                                                                                                                            |
| **gmm2_weight_scale** | `torch.Tensor` | 例如 `[G, 7168]` | 第二阶段权重量化 scale。数据类型规则同 `gmm1_permuted_weight_scale`。                                                                                                                |
| **num_max_dispatch_tokens_per_rank** | `int` | 标量 | `FUSED_DEEP_MOE` 模式：每个 rank 最多分发的 token 数，用于 buffer/内存分配。`DISPATCH_FFN_COMBINE` 模式：dispatch 中最多接收的 token 数（通常为 `max_bs × num_ranks × topk`）。所有 rank 必须持有相同值。        |
| **num_experts** | `int` | 标量，范围 **(0, 512]** | 全局专家总数。必须能被 `(num_ranks - shared_expert_rank_num)` 整除，否则 tiling 校验会拒绝。                                                                                              |
| **quant_mode** | `int` | 标量，默认 `1` | 量化模式：`0` = 无量化（BF16 权重），`1` = INT8（默认）；后续 A5 支持 FP8。                                                                                                                |
| **fuse_mode** | `FuseMode` | 标量，默认 `FuseMode.FUSED_DEEP_MOE` | 融合模式选择。`FUSED_DEEP_MOE`（1）：通过 `aclnnFusedDeepMoe` 完整融合。`DISPATCH_FFN_COMBINE`（2）：通过 `aclnnDispatchFFNCombine` 分离 dispatch 处理。                                       |

### 约束说明

- **bs**（batch size）：取值范围 **[1, 256]**。
- **hidden**：取值范围 **[512, 7168]**，且必须能被 **32** 整除。
- **num_topk**（topk）：取值范围 **(0, 12]** — kernel tiling 强制 `topk ≤ 12`。
- **num_experts**：取值范围 **(0, 512]** — kernel tiling 强制 `moeExpertNum ≤ 512`。`num_experts` 必须能被 `(num_ranks - shared_expert_rank_num)` 整除。

### 返回值

#### `fuse_mode=FUSED_DEEP_MOE`（默认）

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **output** | `torch.Tensor` | `[bs, hidden]` | 融合专家输出。 |
| **ep_recv_count** | `torch.Tensor` | `[num_local_experts × num_ranks]` | EP 通信域各专家在各 rank 收到的 token 数量，用于后续通信同步或负载均衡统计。 |

#### `fuse_mode=DISPATCH_FFN_COMBINE`

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **output** | `torch.Tensor` | `[bs, hidden]` | 融合专家输出。 |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | 本 rank 各本地专家收到的 token 数量，用于后续通信同步或负载均衡统计。 |
