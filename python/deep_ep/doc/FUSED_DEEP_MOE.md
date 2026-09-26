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

Three fuse modes are available via the `FuseMode` enum:

| FuseMode | Value | CANN Operator | Description |
|----------|-------|---------------|-------------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | Full fusion: Dispatch + GMM1 + activation/quantization + GMM2 + Dequant + Unpermute/Combine in a single AscendC kernel. The A5 path supports SwiGLU and SiTU. |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | Integrated routing (`MoeInitRoutingQuantV2`) + AllToAll + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Combine in a single AscendC kernel. |
| `FuseMode.MEGA_MOE` | `3` | `cann_ops_transformer.ops.mega_moe` | Atlas A3 MegaMoe path using one weight tensor per local expert. |

> [!NOTE]
> `FuseMode` is **not** exported from the package's top-level `__init__.py`. Import it explicitly:
> ```python
> from deep_ep.buffer import FuseMode
> ```
> Or use integer values directly: `fuse_mode=1` (FUSED_DEEP_MOE), `fuse_mode=2` (DISPATCH_FFN_COMBINE), or `fuse_mode=3` (MEGA_MOE).

#### Key Differences Between DeepEP Fuse Modes

| Aspect | `FUSED_DEEP_MOE` (mode=1) | `DISPATCH_FFN_COMBINE` (mode=2) |
|--------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 path uses `float32` scales in runtime call; A5 path follows A5 fused host-op contract | Separate dispatch+FFN+combine path with different contract |
| **Weight layout (GMM1)** | A3 legacy path uses the existing permuted-weight contract; A5 also supports `ND` / `FRACTAL_NZ` in current fused host op | Standard path for `aclnnDispatchFFNCombine` |
| **Shared expert** | A3 path supports shared-expert attributes; A5 public fused path currently does not wire shared-expert tensors through the Python entry | Not supported |
| **Second return value** | A3: `ep_recv_count`, shape `[num_local_experts × num_ranks]`; A5: `expert_token_nums`, shape `[num_local_experts]` | `expert_token_nums`, shape `[num_local_experts]` |

`MEGA_MOE` is a separate Atlas A3 path. It requires one weight tensor per local expert and returns local `expert_token_nums`.

### Python API

```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: TensorOrTensors,
    gmm1_permuted_weight_scale: Optional[TensorOrTensors],
    gmm2_weight: TensorOrTensors,
    gmm2_weight_scale: Optional[TensorOrTensors],
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    activation: Optional[str] = "swiglu",
    beta: Optional[float] = 4.0,
    linear_beta: Optional[float] = 25.0,
    profile_enable: bool = False,
    *,
    l1_bias: Optional[TensorOrTensors] = None,
    l2_bias: Optional[TensorOrTensors] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Parameter Description

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token. On A3 this is typically `bfloat16`; on A5 the fused host op supports both `bfloat16` and `float16`. **bs** range **[1, 256]**. **hidden** range **[512, 7168]**. |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token. Python converts it to `int32` before launch. A value of `-1` indicates the token is not dispatched. |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`). |
| **gmm1_permuted_weight** | `TensorOrTensors` | FuseMode-dependent | First-stage expert weights. The DeepEP modes keep their existing tensor contract. `MEGA_MOE` requires one tensor per local expert. |
| **gmm1_permuted_weight_scale** | `Optional[TensorOrTensors]` | FuseMode-dependent | The DeepEP modes keep their existing scale contract. For `MEGA_MOE`, pass one scale tensor per local expert; it is optional for A16W16 and required for quantized execution. |
| **gmm2_weight** | `TensorOrTensors` | FuseMode-dependent | Second-stage expert weights. The DeepEP modes keep their existing tensor contract. `MEGA_MOE` requires one tensor per local expert. |
| **gmm2_weight_scale** | `Optional[TensorOrTensors]` | FuseMode-dependent | Follows the same mode rules as `gmm1_permuted_weight_scale`. |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | Keeps the existing meaning for the DeepEP modes. For `MEGA_MOE`, it is the padded per-rank token capacity, must be at least the local `bs`, and is limited to 4096 on Atlas A3. |
| **num_experts** | `int` | Scalar | Total number of global experts. The existing A5 fused path and `MEGA_MOE` require it to be divisible by the process-group size. |
| **quant_mode** | `int` | Scalar, default `1` | The DeepEP modes keep their existing semantics. `MEGA_MOE` maps `0` to non-quantized dispatch and `1` to INT8 dispatch; weight dtype and optional biases distinguish W8 from W4. The A5 fused path continues to follow its current weight-quantization contract. |
| **fuse_mode** | `FuseMode` | Scalar, default `FuseMode.FUSED_DEEP_MOE` | Selects `FUSED_DEEP_MOE`, `DISPATCH_FFN_COMBINE`, or `MEGA_MOE`. |
| **activation** | `Optional[str]` | Scalar, default `"swiglu"` | The DeepEP modes support `"swiglu"` and `"situ"`; SiTU requires `FUSED_DEEP_MOE`. `MEGA_MOE` supports `"swiglu"`, `"swiglu_gpt_oss"`, and `"situ"`. |
| **beta** | `Optional[float]` | Scalar, default `4.0` | Soft-saturation bound for the SiTU gate branch. `None` uses the internal default `4.0`. It must be greater than zero when SiTU is selected. |
| **linear_beta** | `Optional[float]` | Scalar, default `25.0` | Optional soft-saturation bound for the SiTU up branch. A positive value enables the transformation; `None` leaves the up branch unchanged. |
| **profile_enable** | `bool` | Scalar, default `False` | Whether to enable fused-kernel profiling for the current launch. It only takes effect when profiling has been started in advance (begin_profile). |
| **l1_bias** | `Optional[TensorOrTensors]` | FuseMode-dependent | Optional first-stage A8W4 compensation biases for `MEGA_MOE`. |
| **l2_bias** | `Optional[TensorOrTensors]` | FuseMode-dependent | Optional second-stage A8W4 compensation biases for `MEGA_MOE`. |

### Activation Selection

GMM1 produces two equally sized branches, `gate` and `up`. The default SwiGLU formula is:

```text
silu(x) = x * sigmoid(x)
swiglu(gate, up) = silu(gate) * up
```

SiTU soft-clamps the gate branch and optionally the up branch:

```text
activated_gate = beta * tanh(gate / beta) * sigmoid(gate)
activated_up = linear_beta * tanh(up / linear_beta)  # linear_beta is provided
activated_up = up                                    # linear_beta is None
situ(gate, up) = activated_gate * activated_up
```

Example:

```python
output, expert_token_nums = buffer.fused_deep_moe(
    x,
    topk_idx,
    topk_weights,
    gmm1_permuted_weight,
    gmm1_permuted_weight_scale,
    gmm2_weight,
    gmm2_weight_scale,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    activation="situ",
    beta=4.0,
    linear_beta=25.0,
)
```

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
- SiTU is supported by `FUSED_DEEP_MOE` on both A3 and A5.
- For SiTU, `beta` must be greater than zero. When `linear_beta` is provided, it must also be greater than zero.

#### For `fuse_mode=DISPATCH_FFN_COMBINE` (mode=2)

- Constraints follow the `aclnnDispatchFFNCombine` path and differ from `FUSED_DEEP_MOE`.
- Shared expert is not supported.
- Only SwiGLU is supported. Selecting `activation="situ"` raises `NotImplementedError`.

#### For `fuse_mode=MEGA_MOE` (mode=3)

- Requires `cann_ops_transformer`; the operator is imported lazily only when this mode is selected.
- Requires one weight tensor per local expert.
- `num_experts` must be divisible by the process-group size.
- `num_max_dispatch_tokens_per_rank` must be at least the local token count and cannot exceed 4096 on Atlas A3.
- Supports `"swiglu"`, `"swiglu_gpt_oss"`, and `"situ"`.
- A16W16 uses `quant_mode=0` without scales or biases.
- A8W8 uses `quant_mode=1` with scales and without biases.
- A8W4 uses `quant_mode=1` with scales and compensation biases.

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

#### For `fuse_mode=MEGA_MOE`

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **output** | `torch.Tensor` | `[bs, hidden]` | MegaMoe output, narrowed back to the original local `bs` after internal capacity padding. |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | Number of tokens received by each local expert on this rank. |

---

<a id="中文"></a>

## 中文

### 介绍

在 MoE（Mixture of Experts，混合专家模型）中，`fused_deep_moe` 算子实现了 Dispatch + Experts FFN（2×GMM）+ Combine 的融合功能。
该算子在一次调用中完成 token 分发、专家计算（矩阵乘、激活、量化/反量化）以及结果聚合，相比传统多算子实现可以显著减少通信开销和端到端时延。

通过 `FuseMode` 枚举提供三种融合模式：

| FuseMode | 值 | CANN 算子 | 说明 |
|----------|----|-----------|------|
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | Dispatch + GMM1 + 激活/量化 + GMM2 + 反量化 + Unpermute/Combine 的完整融合路径；A5 路径支持 SwiGLU 和 SiTU。 |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | 另一条 dispatch + FFN + combine 融合路径。 |
| `FuseMode.MEGA_MOE` | `3` | `cann_ops_transformer.ops.mega_moe` | Atlas A3 MegaMoe 路径，每个本地 expert 对应一个权重 Tensor。 |

> [!NOTE]
> `FuseMode` **没有**从包顶层 `__init__.py` 导出，需要显式导入：
> ```python
> from deep_ep.buffer import FuseMode
> ```
> 或者直接使用整数：`fuse_mode=1`（FUSED_DEEP_MOE）、`fuse_mode=2`（DISPATCH_FFN_COMBINE）或 `fuse_mode=3`（MEGA_MOE）。

#### 两种 DeepEP 融合模式的关键差异

| 维度 | `FUSED_DEEP_MOE`（mode=1） | `DISPATCH_FFN_COMBINE`（mode=2） |
|------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 路径在 runtime 调用前会转成 `float32`；A5 路径遵循当前 A5 fused host-op 契约 | 走独立的 dispatch+FFN+combine 契约 |
| **GMM1 权重布局** | A3 保持现有 fused 权重契约；A5 当前 fused host op 额外支持 `ND` / `FRACTAL_NZ` | 遵循 `aclnnDispatchFFNCombine` 自身契约 |
| **Shared expert** | A3 路径支持 shared-expert 属性；A5 公共 fused Python 路径当前未真正透传 shared-expert tensor | 不支持 |
| **第二返回值** | A3：`ep_recv_count`，shape `[num_local_experts × num_ranks]`；A5：`expert_token_nums`，shape `[num_local_experts]` | `expert_token_nums`，shape `[num_local_experts]` |

`MEGA_MOE` 是独立的 Atlas A3 路径，要求每个本地 expert 对应一个权重 Tensor，并返回本地 `expert_token_nums`。

### Python API

```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: TensorOrTensors,
    gmm1_permuted_weight_scale: Optional[TensorOrTensors],
    gmm2_weight: TensorOrTensors,
    gmm2_weight_scale: Optional[TensorOrTensors],
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    activation: Optional[str] = "swiglu",
    beta: Optional[float] = 4.0,
    linear_beta: Optional[float] = 25.0,
    profile_enable: bool = False,
    *,
    l1_bias: Optional[TensorOrTensors] = None,
    l2_bias: Optional[TensorOrTensors] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数说明

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **x** | `torch.Tensor` | `[bs, hidden]` | 输入 token 表示。A3 上通常使用 `bfloat16`；A5 fused host op 支持 `bfloat16` 和 `float16`。**bs** 范围 **[1, 256]**，**hidden** 范围 **[512, 7168]**。 |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | 每个 token 的 expert 索引。Python 层会在下发前转成 `int32`。`-1` 表示该 token 不分发。 |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | expert 输出聚合权重（`float32`）。 |
| **gmm1_permuted_weight** | `TensorOrTensors` | 取决于 FuseMode | 第一层 expert 权重。两个 DeepEP 模式保持原有 Tensor 契约；`MEGA_MOE` 要求每个本地 expert 对应一个 Tensor。 |
| **gmm1_permuted_weight_scale** | `Optional[TensorOrTensors]` | 取决于 FuseMode | 两个 DeepEP 模式保持原有 scale 契约；`MEGA_MOE` 按本地 expert 传入 scale 列表，A16W16 可传 `None`，量化模式必须提供。 |
| **gmm2_weight** | `TensorOrTensors` | 取决于 FuseMode | 第二层 expert 权重。两个 DeepEP 模式保持原有 Tensor 契约；`MEGA_MOE` 要求每个本地 expert 对应一个 Tensor。 |
| **gmm2_weight_scale** | `Optional[TensorOrTensors]` | 取决于 FuseMode | 模式规则与 `gmm1_permuted_weight_scale` 相同。 |
| **num_max_dispatch_tokens_per_rank** | `int` | 标量 | 两个 DeepEP 模式保持原有含义；在 `MEGA_MOE` 中表示每个 rank padding 后的 token 容量，必须不小于本地 `bs`，且 Atlas A3 上不能超过 4096。 |
| **num_experts** | `int` | 标量 | 全局 expert 总数。现有 A5 fused 路径和 `MEGA_MOE` 都要求它能被进程组大小整除。 |
| **quant_mode** | `int` | 标量，默认 `1` | 两个 DeepEP 模式保持原有语义。`MEGA_MOE` 将 `0` 映射为非量化 dispatch，将 `1` 映射为 INT8 dispatch；权重 dtype 和可选 bias 用于区分 W8 与 W4。A5 现有 fused 路径继续遵循自身的权重量化契约。 |
| **fuse_mode** | `FuseMode` | 标量，默认 `FuseMode.FUSED_DEEP_MOE` | 选择 `FUSED_DEEP_MOE`、`DISPATCH_FFN_COMBINE` 或 `MEGA_MOE`。 |
| **activation** | `Optional[str]` | 标量，默认 `"swiglu"` | DeepEP 模式支持 `"swiglu"` 和 `"situ"`，SiTU 要求使用 `FUSED_DEEP_MOE`；`MEGA_MOE` 支持 `"swiglu"`、`"swiglu_gpt_oss"` 和 `"situ"`。 |
| **beta** | `Optional[float]` | 标量，默认 `4.0` | SiTU gate 分支的软饱和边界。`None` 使用内部默认值 `4.0`。选择 SiTU 时该值必须大于零。 |
| **linear_beta** | `Optional[float]` | 标量，默认 `25.0` | SiTU up 分支可选的软饱和边界。正数表示启用该变换；`None` 表示 up 分支保持不变。 |
| **profile_enable** | `bool` | 标量，默认值为 `False` | 是否为当前运行启用kernel性能分析。仅在预先启动了性能分析时（begin_profile）才生效。 |
| **l1_bias** | `Optional[TensorOrTensors]` | 取决于 FuseMode | `MEGA_MOE` 第一层可选的 A8W4 补偿 bias。 |
| **l2_bias** | `Optional[TensorOrTensors]` | 取决于 FuseMode | `MEGA_MOE` 第二层可选的 A8W4 补偿 bias。 |

### 激活选择

GMM1 输出会被平均拆分成 `gate` 和 `up` 两个分支。默认的 SwiGLU 公式如下：

```text
silu(x) = x * sigmoid(x)
swiglu(gate, up) = silu(gate) * up
```

SiTU 对 gate 分支进行软限幅，并可选地对 up 分支进行软限幅：

```text
activated_gate = beta * tanh(gate / beta) * sigmoid(gate)
activated_up = linear_beta * tanh(up / linear_beta)  # 提供 linear_beta
activated_up = up                                    # linear_beta 为 None
situ(gate, up) = activated_gate * activated_up
```

调用示例：

```python
output, expert_token_nums = buffer.fused_deep_moe(
    x,
    topk_idx,
    topk_weights,
    gmm1_permuted_weight,
    gmm1_permuted_weight_scale,
    gmm2_weight,
    gmm2_weight_scale,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    activation="situ",
    beta=4.0,
    linear_beta=25.0,
)
```

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
- A3 和 A5 的 `FUSED_DEEP_MOE` 都支持 SiTU。
- 使用 SiTU 时，`beta` 必须大于零；提供 `linear_beta` 时，该值也必须大于零。

#### 对于 `fuse_mode=DISPATCH_FFN_COMBINE`（mode=2）

- 约束遵循 `aclnnDispatchFFNCombine` 路径，与 `FUSED_DEEP_MOE` 不同。
- 不支持 shared expert。
- 只支持 SwiGLU；选择 `activation="situ"` 会抛出 `NotImplementedError`。

#### 对于 `fuse_mode=MEGA_MOE`（mode=3）

- 依赖 `cann_ops_transformer`，仅在选择该模式时延迟导入算子。
- 要求每个本地 expert 对应一个权重 Tensor。
- `num_experts` 必须能被进程组大小整除。
- `num_max_dispatch_tokens_per_rank` 必须不小于本地 token 数，且 Atlas A3 上不能超过 4096。
- 支持 `"swiglu"`、`"swiglu_gpt_oss"` 和 `"situ"`。
- A16W16 使用 `quant_mode=0`，不传 scale 和 bias。
- A8W8 使用 `quant_mode=1`，需要 scale，不传 bias。
- A8W4 使用 `quant_mode=1`，需要 scale 和补偿 bias。

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

#### 对于 `fuse_mode=MEGA_MOE`

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **output** | `torch.Tensor` | `[bs, hidden]` | MegaMoe 输出；内部 capacity padding 后，返回前会裁回原始本地 `bs`。 |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | 当前 rank 上每个本地 expert 接收到的 token 数。 |
