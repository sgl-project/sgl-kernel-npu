# Fused Deep MoE API

<div align="center">

[![Mode](https://img.shields.io/badge/Mode-Fused-purple)]()
[![Platform](https://img.shields.io/badge/Platform-A3%20%7C%20A5-blue)]()
[![Quant](https://img.shields.io/badge/Quantization-INT8%20%7C%20FP8%20%7C%20FP4-yellow)]()

English | [中文](#中文)

</div>

> [!IMPORTANT]
> This API is available on both A3 and A5 in the current codebase, but the A5 path is not identical to the A3 path. In particular, A5 has different weight dtype/layout support, capacity handling, and second-return-value semantics.

---

## English

### Introduction

In Mixture of Experts (MoE) models, the `fused_deep_moe` operator implements the hyper-fusion of Dispatch + Experts FFN (2×GMM) + Combine functionalities.
This operator completes token distribution, expert computation (matrix multiplication, activation, quantization/dequantization), and result aggregation in a single call. Compared with traditional multi-operator implementations, it significantly reduces communication overhead and end-to-end latency.

Two fuse modes are available via the `FuseMode` enum:

| FuseMode | Value | CANN Operator | Description |
|----------|-------|---------------|-------------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | Full fusion: Dispatch + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Unpermute/Combine in a single AscendC kernel. |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | Integrated routing (`MoeInitRoutingQuantV2`) + AllToAll + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Combine in a single AscendC kernel. |

> [!NOTE]
> `FuseMode` is **not** exported from the package's top-level `__init__.py`. Import it explicitly:
> ```python
> from deep_ep.buffer import FuseMode
> ```
> Or use integer values directly: `fuse_mode=1` (FUSED_DEEP_MOE) or `fuse_mode=2` (DISPATCH_FFN_COMBINE).

#### Key Differences Between Fuse Modes

| Aspect | `FUSED_DEEP_MOE` (mode=1) | `DISPATCH_FFN_COMBINE` (mode=2) |
|--------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 path uses `float32` scales in runtime call; A5 path follows A5 fused host-op contract | Separate dispatch+FFN+combine path with different contract |
| **Weight layout (GMM1)** | A3 legacy path uses the existing permuted-weight contract; A5 also supports `ND` / `FRACTAL_NZ` in current fused host op | Standard path for `aclnnDispatchFFNCombine` |
| **Shared expert** | A3 path supports shared-expert attributes; A5 public fused path currently does not wire shared-expert tensors through the Python entry | Not supported |
| **Second return value** | A3: `ep_recv_count`, shape `[num_local_experts × num_ranks]`; A5: `expert_token_nums`, shape `[num_local_experts]` | `expert_token_nums`, shape `[num_local_experts]` |

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
    profile_enable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Parameter Description

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token. On A3 this is typically `bfloat16`; on A5 the fused host op supports both `bfloat16` and `float16`. **bs** range **[1, 256]**. **hidden** range **[512, 7168]**. |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token. Python converts it to `int32` before launch. A value of `-1` indicates the token is not dispatched. |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`). |
| **gmm1_permuted_weight** | `torch.Tensor` | e.g., `[G, 7168, 4096]` | First-stage (up-projection) expert weights. A3 keeps the existing fused-path weight contract. On A5, the current fused host op additionally supports quantized weights in `ND` and `FRACTAL_NZ` format. |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | e.g., `[G, 4096]` | Quantization scale for first-stage weights. A3 runtime converts scales to `float32` before launch. On A5 fused path, the host-op contract is different and follows the current A5 quantized-weight definition. |
| **gmm2_weight** | `torch.Tensor` | e.g., `[G, 7168, 2048]` | Second-stage (down-projection) expert weights. Same A3/A5 difference as `gmm1_permuted_weight`. |
| **gmm2_weight_scale** | `torch.Tensor` | e.g., `[G, 7168]` | Quantization scale for second-stage weights. Same A3/A5 difference as `gmm1_permuted_weight_scale`. |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | For A3, used in the existing fused-path buffer sizing logic. For A5, this value is also used as per-rank **capacity**, and must be **greater than or equal to local bs**. |
| **num_experts** | `int` | Scalar, range **(0, 512]** | Total number of global experts. On A5 fused path, current tiling requires `num_experts` to be divisible by EP rank size. |
| **quant_mode** | `int` | Scalar, default `1` | Quantization mode attribute passed to the fused operator. A3 follows the legacy fused-path semantics. On A5, this parameter is currently not effective in the public fused path: activation quantization follows the weight quantization type, so the practical supported combinations are `w8a8` and `w4a4`. `w4a8` is not supported, and non-quantized model weights are not supported in the current A5 fused path. |
| **fuse_mode** | `FuseMode` | Scalar, default `FuseMode.FUSED_DEEP_MOE` | Fuse mode selection. |
| **profile_enable** | `bool` | Scalar, default `False` | Whether to enable fused-kernel profiling for the current launch. It only takes effect when profiling has been started in advance (begin_profile). |

### Constraints

#### For `fuse_mode=FUSED_DEEP_MOE` (mode=1)

- **bs** (batch size): range **[1, 256]**.
- **hidden**: range **[512, 7168]**.
- **gmm1_hidden**: range **[1024, 6144]**.
- **gmm1_hidden** must be divisible by **1024** on A5 fused path.
- **num_topk** (topk): range **(0, 12]**.
- **num_experts**: range **(0, 512]**.
- On **A5**, `num_experts` must be divisible by EP rank size.
- On **A5**, `num_max_dispatch_tokens_per_rank >= bs`.
- On **A5 MXFP4** paths, `hidden` and `gmm1_hidden` must be even.
- On **A5 MXFP4** paths, quantized weights in `FRACTAL_NZ` format are not supported currently.

#### For `fuse_mode=DISPATCH_FFN_COMBINE` (mode=2)

- Constraints follow the `aclnnDispatchFFNCombine` path and differ from `FUSED_DEEP_MOE`.
- Shared expert is not supported.

### Return Values

#### For `fuse_mode=FUSED_DEEP_MOE` (default)

| Platform | Parameter | Type | Shape | Description |
|----------|-----------|------|-------|-------------|
| A3 | **output** | `torch.Tensor` | `[bs, hidden]` | Fused expert outputs. |
| A3 | **ep_recv_count** | `torch.Tensor` | `[num_local_experts × num_ranks]` | Number of tokens received by each expert across all ranks in the EP communication domain. |
| A5 | **output** | `torch.Tensor` | `[bs, hidden]` | Fused expert outputs. When capacity padding is used internally, the returned tensor is narrowed back to the original local `bs`. |
| A5 | **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | Number of tokens received by each local expert on this rank. |

#### For `fuse_mode=DISPATCH_FFN_COMBINE`

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **output** | `torch.Tensor` | `[bs, hidden]` | Fused expert outputs. |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | Number of tokens received by each local expert on this rank. |

---

<a id="中文"></a>

## 中文

### 介绍

在 MoE（Mixture of Experts，混合专家模型）中，`fused_deep_moe` 算子实现了 Dispatch + Experts FFN（2×GMM）+ Combine 的融合功能。
该算子在一次调用中完成 token 分发、专家计算（矩阵乘、激活、量化/反量化）以及结果聚合，相比传统多算子实现可以显著减少通信开销和端到端时延。

通过 `FuseMode` 枚举提供两种融合模式：

| FuseMode | 值 | CANN 算子 | 说明 |
|----------|----|-----------|------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | 完整融合的 dispatch + FFN + combine 路径。 |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | 另一条 dispatch + FFN + combine 融合路径。 |

> [!NOTE]
> `FuseMode` **没有**从包顶层 `__init__.py` 导出，需要显式导入：
> ```python
> from deep_ep.buffer import FuseMode
> ```
> 或者直接使用整数：`fuse_mode=1`（FUSED_DEEP_MOE）或 `fuse_mode=2`（DISPATCH_FFN_COMBINE）。

#### 两种融合模式的关键差异

| 维度 | `FUSED_DEEP_MOE`（mode=1） | `DISPATCH_FFN_COMBINE`（mode=2） |
|------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 路径在 runtime 调用前会转成 `float32`；A5 路径遵循当前 A5 fused host-op 契约 | 走独立的 dispatch+FFN+combine 契约 |
| **GMM1 权重布局** | A3 保持现有 fused 权重契约；A5 当前 fused host op 额外支持 `ND` / `FRACTAL_NZ` | 遵循 `aclnnDispatchFFNCombine` 自身契约 |
| **Shared expert** | A3 路径支持 shared-expert 属性；A5 公共 fused Python 路径当前未真正透传 shared-expert tensor | 不支持 |
| **第二返回值** | A3：`ep_recv_count`，shape `[num_local_experts × num_ranks]`；A5：`expert_token_nums`，shape `[num_local_experts]` | `expert_token_nums`，shape `[num_local_experts]` |

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
    profile_enable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数说明

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **x** | `torch.Tensor` | `[bs, hidden]` | 输入 token 表示。A3 上通常使用 `bfloat16`；A5 fused host op 支持 `bfloat16` 和 `float16`。**bs** 范围 **[1, 256]**，**hidden** 范围 **[512, 7168]**。 |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | 每个 token 的 expert 索引。Python 层会在下发前转成 `int32`。`-1` 表示该 token 不分发。 |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | expert 输出聚合权重（`float32`）。 |
| **gmm1_permuted_weight** | `torch.Tensor` | 例如 `[G, 7168, 4096]` | 第一层（up-projection）expert 权重。A3 保持原有 fused 权重契约；A5 当前 fused host op 额外支持量化权重的 `ND` / `FRACTAL_NZ` 格式。 |
| **gmm1_permuted_weight_scale** | `torch.Tensor` | 例如 `[G, 4096]` | 第一层权重 scale。A3 runtime 会先转成 `float32`；A5 fused 路径遵循当前 A5 quantized-weight host-op 契约。 |
| **gmm2_weight** | `torch.Tensor` | 例如 `[G, 7168, 2048]` | 第二层（down-projection）expert 权重。A3/A5 差异同 `gmm1_permuted_weight`。 |
| **gmm2_weight_scale** | `torch.Tensor` | 例如 `[G, 7168]` | 第二层权重 scale。A3/A5 差异同 `gmm1_permuted_weight_scale`。 |
| **num_max_dispatch_tokens_per_rank** | `int` | 标量 | A3 中沿用现有 fused buffer 逻辑；A5 中这个值还直接作为每个 rank 的 **capacity**，必须满足 **大于等于本地 bs**。 |
| **num_experts** | `int` | 标量，范围 **(0, 512]** | 全局 expert 总数。A5 fused 当前 tiling 要求 `num_experts` 能被 EP rank size 整除。 |
| **quant_mode** | `int` | 标量，默认 `1` | 下发给 fused 算子的量化模式属性。A3 保持 legacy fused 语义；A5 公共 fused 路径上该参数当前实际上不生效，激活量化方式会跟随权重量化方式，因此当前实际只支持 `w8a8` 和 `w4a4`。`w4a8` 暂不支持，非量化模型权重在当前 A5 fused 路径上也不支持。 |
| **fuse_mode** | `FuseMode` | 标量，默认 `FuseMode.FUSED_DEEP_MOE` | 融合模式选择。 |
| **profile_enable** | `bool` | 标量，默认值为 `False` | 是否为当前运行启用kernel性能分析。仅在预先启动了性能分析时（begin_profile）才生效。 |

### A5 变化点说明

- 当前 A5 fused 路径走 `__DAV_C310__` runtime 分支。
- A5 中 `global_bs = num_max_dispatch_tokens_per_rank * num_ranks`。
- 当 `num_max_dispatch_tokens_per_rank > bs` 时，A5 会对 `x`、`expert_ids` 和 expert scale 做 padding，并走内部 active-mask 路径。
- A5 公共 fused 路径当前返回的是本地 `expert_token_nums`，而不是 A3 的 `ep_recv_count`。
- A5 fused host op 当前支持量化 GMM 权重：
  - weight dtype：FP8 E4M3 / FP8 E5M2 / FP4 E2M1 / FP4 E1M2
  - weight format：`ND` / `FRACTAL_NZ`
- 虽然 A5 host op 定义了 shared-expert 可选输入，但当前公共 fused Python 入口并没有把 shared-expert tensor 真正传进这条路径。

### 约束说明

#### 对于 `fuse_mode=FUSED_DEEP_MOE`（mode=1）

- **bs**（batch size）：范围 **[1, 256]**。
- **hidden**：范围 **[512, 7168]**。
- **gmm1_hidden**：范围 **[1024, 6144]**。
- 在 **A5 fused** 路径上，**gmm1_hidden** 必须能被 **1024** 整除。
- **num_topk**（topk）：范围 **(0, 12]**。
- **num_experts**：范围 **(0, 512]**。
- 在 **A5** 上，`num_experts` 必须能被 EP rank size 整除。
- 在 **A5** 上，`num_max_dispatch_tokens_per_rank >= bs`。
- 在 **A5 MXFP4** 路径上，`hidden` 和 `gmm1_hidden` 还必须为偶数。
- 在 **A5 MXFP4** 路径上，量化权重当前暂不支持 `FRACTAL_NZ` 格式。

#### 对于 `fuse_mode=DISPATCH_FFN_COMBINE`（mode=2）

- 约束遵循 `aclnnDispatchFFNCombine` 路径，与 `FUSED_DEEP_MOE` 不同。
- 不支持 shared expert。

### 返回值

#### 对于 `fuse_mode=FUSED_DEEP_MOE`（默认）

| 平台 | 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|------|
| A3 | **output** | `torch.Tensor` | `[bs, hidden]` | 融合后的 expert 输出。 |
| A3 | **ep_recv_count** | `torch.Tensor` | `[num_local_experts × num_ranks]` | EP 通信域内按 rank 展开的 expert 接收 token 计数。 |
| A5 | **output** | `torch.Tensor` | `[bs, hidden]` | 融合后的 expert 输出。若内部使用了 capacity padding，返回前会裁回原始本地 `bs`。 |
| A5 | **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | 当前 rank 上每个本地 expert 接收到的 token 数。 |

#### 对于 `fuse_mode=DISPATCH_FFN_COMBINE`

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **output** | `torch.Tensor` | `[bs, hidden]` | 融合后的 expert 输出。 |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | 当前 rank 上每个本地 expert 接收到的 token 数。 |
