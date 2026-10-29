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
| `FuseMode.MEGA_MOE` | `3` | `cann_ops_transformer.ops.mega_moe` | A3 and A5 MegaMoe path with platform-specific weight layouts. |

> [!NOTE]
> `FuseMode` is **not** exported from the package's top-level `__init__.py`. Import it explicitly:
> ```python
> from deep_ep.buffer import FuseMode
> ```
> Or use integer values directly: `fuse_mode=1` (FUSED_DEEP_MOE), `fuse_mode=2` (DISPATCH_FFN_COMBINE), or `fuse_mode=3` (MEGA_MOE).

#### Key Differences Between Fuse Modes

| Aspect | `FUSED_DEEP_MOE` (mode=1) | `DISPATCH_FFN_COMBINE` (mode=2) |
|--------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 path uses `float32` scales in runtime call; A5 path follows A5 fused host-op contract | Separate dispatch+FFN+combine path with different contract |
| **Weight layout (GMM1)** | A3 legacy path uses the existing permuted-weight contract; A5 also supports `ND` / `FRACTAL_NZ` in current fused host op | Standard path for `aclnnDispatchFFNCombine` |
| **Shared expert** | A3 path supports shared-expert attributes; A5 public fused path currently does not wire shared-expert tensors through the Python entry | Not supported |
| **Second return value** | A3: `ep_recv_count`, shape `[num_local_experts × num_ranks]`; A5: `expert_token_nums`, shape `[num_local_experts]` | `expert_token_nums`, shape `[num_local_experts]` |

`MEGA_MOE` runs on A3 and A5 and returns local `expert_token_nums`. Its layout, quantization and capacity requirements are described below.

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
    quant_mode: Union[int, str] = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    activation: Optional[str] = "swiglu",
    beta: Optional[float] = 4.0,
    linear_beta: Optional[float] = 25.0,
    l1_bias: Optional[TensorOrTensors] = None,
    l2_bias: Optional[TensorOrTensors] = None,
    profile_enable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Parameter Description

The x ranges below describe `FUSED_DEEP_MOE`; see the MegaMoe section for its separate A5 constraints.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token. On A3 this is typically `bfloat16`; on A5 the fused host op supports both `bfloat16` and `float16`. **bs** range **[1, 256]**. **hidden** range **[512, 7168]**. |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token. Python converts it to `int32` before launch. Only the DeepEP modes support `-1`; MegaMoe requires valid, non-duplicate expert IDs per token. |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`; A5 MegaMoe also accepts BF16). |
| **gmm1_permuted_weight** | `TensorOrTensors` | FuseMode-dependent | First-stage expert weights. The DeepEP modes keep their existing tensor contract. `MEGA_MOE` requires one tensor per local expert on A3; A5 also supports a single stacked Tensor. |
| **gmm1_permuted_weight_scale** | `Optional[TensorOrTensors]` | FuseMode-dependent | The DeepEP modes keep their existing scale contract. A3 MegaMoe scales are optional for A16W16 and required for INT execution. A5 requires matching ND E8M0 scale lists for all MXFP modes. |
| **gmm2_weight** | `TensorOrTensors` | FuseMode-dependent | Second-stage expert weights. The DeepEP modes keep their existing tensor contract. `MEGA_MOE` requires one tensor per local expert on A3; A5 also supports a single stacked Tensor. |
| **gmm2_weight_scale** | `Optional[TensorOrTensors]` | FuseMode-dependent | Follows the same mode rules as `gmm1_permuted_weight_scale`. |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | Keeps the existing meaning for the DeepEP modes. For MegaMoe it is a common capacity upper bound across ranks: A3 pad to it, A5 passes actual lengths. Atlas A3 is limited to 4096. |
| **num_experts** | `int` | Scalar | Total number of global experts. The existing A5 fused path and `MEGA_MOE` require it to be divisible by the process-group size. |
| **quant_mode** | `Union[int, str]` | Scalar, default `1` | DeepEP modes retain integer 0/1. A3 MegaMoe accepts 0/`"bf16"`, 1/`"int8"`; A5 requires `"mx_fp8_e4m3"`, `"mx_fp8_e5m2"`, `"mx_fp4_e2m1"` or `"mx_fp8_e4m3_w4"`. Strings are exclusive to MegaMoe. |
| **fuse_mode** | `FuseMode` | Scalar, default `FuseMode.FUSED_DEEP_MOE` | Selects `FUSED_DEEP_MOE`, `DISPATCH_FFN_COMBINE`, or `MEGA_MOE`. |
| **activation** | `Optional[str]` | Scalar, default `"swiglu"` | On A5, `FUSED_DEEP_MOE` supports `"situ"` and `"swiglu"`; `DISPATCH_FFN_COMBINE` supports only `"swiglu"`; on A3 and A5, `MEGA_MOE` supports `"situ"` and `"swiglu"`. |
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

Example: this SiTU call uses `FUSED_DEEP_MOE` on A5. On Atlas A3, select `fuse_mode=FuseMode.MEGA_MOE` for SiTU.

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
- SiTU is supported by `FUSED_DEEP_MOE` on **A5** and by `MEGA_MOE` on **A3 and A5**.
- For SiTU, `beta` must be greater than zero. When `linear_beta` is provided, it must also be greater than zero.

#### For `fuse_mode=DISPATCH_FFN_COMBINE` (mode=2)

- Constraints follow the `aclnnDispatchFFNCombine` path and differ from `FUSED_DEEP_MOE`.
- Shared expert is not supported.
- Only SwiGLU is supported. Selecting `activation="situ"` raises `NotImplementedError`.

#### For `fuse_mode=MEGA_MOE` (mode=3)

- Requires `cann_ops_transformer` (CANN 9.2.0 on all platforms), imported lazily only when this mode is called.
- The Python strategy does not query the device or cache its architecture. Quantization modes map directly to CANN parameters; hardware support and platform-specific constraints are checked by CANN.
- Supports `"swiglu"` and `"situ"`, returning local `expert_token_nums`.
- A3 keep per-expert TensorLists and capacity padding with an active mask.
- A3 A16W16 uses `0` / `"bf16"` without scales or biases; A8W8-INT uses `1` / `"int8"` with scales; A8W4-INT additionally supplies compensation biases.
- A5 requires `"mx_fp8_e4m3"`, `"mx_fp8_e5m2"`, `"mx_fp4_e2m1"` (A4W4-FP) or `"mx_fp8_e4m3_w4"` (A8W4-FP). Integer 0/1 and Per-token FP8 are not exposed on A5 MegaMoe.
- A5 uses BF16 `x[N,H]` at its actual length with `1 <= N <= capacity` and `x_active_mask=None`. Ranks may have different `N` but must configure the same capacity. No input padding or output slicing occurs.
- A5 requires `l1_bias/l2_bias=None`. These MegaMoe rules do not change `FUSED_DEEP_MOE`.

Let `E = num_experts / ep_world_size` and `I` be the logical intermediate dimension. Stacked lists contain one Tensor with the following shape; per-expert lists contain `E` Tensors without the leading `E`:

| Input | FP8 | Packed FP4 |
|---|---|---|
| GMM1 weight | `[E,2I,H]` | `[E,2I,H/2]` |
| GMM2 weight | `[E,H,I]` | `[E,H,I/2]` |
| GMM1 scale | `[E,2I,ceil(H/64),2]` | Same |
| GMM2 scale | `[E,H,ceil(I/64),2]` | Same |

All four lists must use the same layout, including when `E=1`. FP8 weights must match the requested E4M3FN/E5M2 dtype; packed FP4 uses `uint8`. All scales are required ND `torch.float8_e8m0fnu`, including FP4; weights/scales must be contiguous and on `x.device`.

| quant_mode | GMM1 format | GMM2 format |
|---|---|---|
| `"mx_fp8_e5m2"` | ND | ND |
| `"mx_fp8_e4m3"` | ND / FRACTAL_NZ | ND / FRACTAL_NZ |
| `"mx_fp4_e2m1"` | FRACTAL_NZ | FRACTAL_NZ_C0_32 |
| `"mx_fp8_e4m3_w4"` | FRACTAL_NZ_C0_32 | FRACTAL_NZ_C0_32 |

Prepare CANN weight layouts/formats before calling; the wrapper does not quantize, transpose or format-cast weights. FP4 types use `torch_npu.float4_e2m1fn_x2` (integer 296), not a torch dtype.

Inputs must satisfy CANN's platform-specific requirements. Callers must provide valid, non-duplicate expert IDs per token; the wrapper does not scan expert ID values.

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
| **output** | `torch.Tensor` | `[bs, hidden]` | MegaMoe output at local `bs`: A3 slice after padding; A5 does not pad or slice. |
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
| `FuseMode.MEGA_MOE` | `3` | `cann_ops_transformer.ops.mega_moe` | A3、A5 MegaMoe 路径，权重布局按平台区分。 |

> [!NOTE]
> `FuseMode` **没有**从包顶层 `__init__.py` 导出，需要显式导入：
> ```python
> from deep_ep.buffer import FuseMode
> ```
> 或者直接使用整数：`fuse_mode=1`（FUSED_DEEP_MOE）、`fuse_mode=2`（DISPATCH_FFN_COMBINE）或 `fuse_mode=3`（MEGA_MOE）。

#### 融合模式的关键差异

| 维度 | `FUSED_DEEP_MOE`（mode=1） | `DISPATCH_FFN_COMBINE`（mode=2） |
|------|---------------------------|---------------------------------|
| **Weight scale dtype** | A3 路径在 runtime 调用前会转成 `float32`；A5 路径遵循当前 A5 fused host-op 契约 | 走独立的 dispatch+FFN+combine 契约 |
| **GMM1 权重布局** | A3 保持现有 fused 权重契约；A5 当前 fused host op 额外支持 `ND` / `FRACTAL_NZ` | 遵循 `aclnnDispatchFFNCombine` 自身契约 |
| **Shared expert** | A3 路径支持 shared-expert 属性；A5 公共 fused Python 路径当前未真正透传 shared-expert tensor | 不支持 |
| **第二返回值** | A3：`ep_recv_count`，shape `[num_local_experts × num_ranks]`；A5：`expert_token_nums`，shape `[num_local_experts]` | `expert_token_nums`，shape `[num_local_experts]` |

`MEGA_MOE` 支持 A3、A5，返回本地 `expert_token_nums`；布局、量化和容量要求见下文。

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
    quant_mode: Union[int, str] = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    activation: Optional[str] = "swiglu",
    beta: Optional[float] = 4.0,
    linear_beta: Optional[float] = 25.0,
    l1_bias: Optional[TensorOrTensors] = None,
    l2_bias: Optional[TensorOrTensors] = None,
    profile_enable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 参数说明

下表 x 的范围描述 `FUSED_DEEP_MOE`；A5 MegaMoe 的独立约束见下文对应章节。

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| **x** | `torch.Tensor` | `[bs, hidden]` | 输入 token 表示。A3 上通常使用 `bfloat16`；A5 fused host op 支持 `bfloat16` 和 `float16`。**bs** 范围 **[1, 256]**，**hidden** 范围 **[512, 7168]**。 |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | 每个 token 的 expert 索引。Python 层转成 `int32`。仅 DeepEP 模式支持 `-1`；MegaMoe 要求有效专家编号，同一 token 不重复选择专家。 |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | expert 输出聚合权重（`float32`；A5 MegaMoe 也接受 BF16）。 |
| **gmm1_permuted_weight** | `TensorOrTensors` | 取决于 FuseMode | 第一层 expert 权重。两个 DeepEP 模式保持原有 Tensor 契约；A3 MegaMoe 使用逐专家列表，A5 也支持单 Tensor 堆叠列表。 |
| **gmm1_permuted_weight_scale** | `Optional[TensorOrTensors]` | 取决于 FuseMode | 两个 DeepEP 模式保持原有 scale 契约；A3 MegaMoe 的 A16W16 可不传 scale，INT 模式必须提供；A5 所有 MXFP 模式必须传匹配布局的 ND E8M0 scale 列表。 |
| **gmm2_weight** | `TensorOrTensors` | 取决于 FuseMode | 第二层 expert 权重。两个 DeepEP 模式保持原有 Tensor 契约；A3 MegaMoe 使用逐专家列表，A5 也支持单 Tensor 堆叠列表。 |
| **gmm2_weight_scale** | `Optional[TensorOrTensors]` | 取决于 FuseMode | 模式规则与 `gmm1_permuted_weight_scale` 相同。 |
| **num_max_dispatch_tokens_per_rank** | `int` | 标量 | 两个 DeepEP 模式保持原有含义；在 MegaMoe 中表示各 rank 一致的容量上界：A3 padding 到该值，A5 传实际长度。Atlas A3 上不能超过 4096。 |
| **num_experts** | `int` | 标量 | 全局 expert 总数。现有 A5 fused 路径和 `MEGA_MOE` 都要求它能被进程组大小整除。 |
| **quant_mode** | `Union[int, str]` | 标量，默认 `1` | DeepEP 模式保留整数 0/1。A3 MegaMoe 接受 0/`"bf16"`、1/`"int8"`；A5 必须选择 `"mx_fp8_e4m3"`、`"mx_fp8_e5m2"`、`"mx_fp4_e2m1"` 或 `"mx_fp8_e4m3_w4"`。字符串仅用于 MegaMoe。 |
| **fuse_mode** | `FuseMode` | 标量，默认 `FuseMode.FUSED_DEEP_MOE` | 选择 `FUSED_DEEP_MOE`、`DISPATCH_FFN_COMBINE` 或 `MEGA_MOE`。 |
| **activation** | `Optional[str]` | 标量，默认 `"swiglu"` | 在 A5 上，`FUSED_DEEP_MOE` 支持 `"situ"` 和 `"swiglu"`；`DISPATCH_FFN_COMBINE` 仅支持 `"swiglu"`；在 A3、A5 上，`MEGA_MOE` 支持 `"situ"` 和 `"swiglu"`。 |
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

调用示例：该 SiTU 调用在 A5 上使用 `FUSED_DEEP_MOE`；在 Atlas A3 上使用 SiTU 时，应选择 `fuse_mode=FuseMode.MEGA_MOE`。

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
- 在 **A5** 上，`FUSED_DEEP_MOE` 支持 SiTU；在 **A3、A5** 上，`MEGA_MOE` 支持 SiTU。
- 使用 SiTU 时，`beta` 必须大于零；提供 `linear_beta` 时，该值也必须大于零。

#### 对于 `fuse_mode=DISPATCH_FFN_COMBINE`（mode=2）

- 约束遵循 `aclnnDispatchFFNCombine` 路径，与 `FUSED_DEEP_MOE` 不同。
- 不支持 shared expert。
- 只支持 SwiGLU；选择 `activation="situ"` 会抛出 `NotImplementedError`。

#### 对于 `fuse_mode=MEGA_MOE`（mode=3）

- 依赖 `cann_ops_transformer`（各机型均要求 CANN 9.2.0），仅在调用该模式时惰性导入。
- Python 策略不查询设备或缓存架构，只将显式量化模式映射为 CANN 参数；平台支持范围和芯片相关约束由 CANN 检查。
- 支持 `"swiglu"` 和 `"situ"`，返回本地 `expert_token_nums`。
- A3 保留逐专家 TensorList、capacity padding 和 active mask。
- A3 的 A16W16 使用 `0` / `"bf16"`，不传 scale 和 bias；A8W8-INT 使用 `1` / `"int8"` 并提供 scale；A8W4-INT 还需补偿 bias。
- A5 必须选择 `"mx_fp8_e4m3"`、`"mx_fp8_e5m2"`、`"mx_fp4_e2m1"`（A4W4-FP）或 `"mx_fp8_e4m3_w4"`（A8W4-FP），不接受整数 0/1 和 Per-token FP8。
- A5 BF16 `x[N,H]` 按实际长度调用，满足 `1 <= N <= capacity`，`x_active_mask=None`。各 rank 的 `N` 可以不同，但容量上界必须一致，不 padding、不裁切输出。
- A5 要求 `l1_bias/l2_bias=None`。这些 MegaMoe 规则不改变 `FUSED_DEEP_MOE`。

设 `E = num_experts / ep_world_size`，`I` 为逻辑中间维度。堆叠列表长度为 1，形状如下；逐专家列表长度为 `E`，每个 Tensor 去掉首维 `E`：

| 输入 | FP8 | packed FP4 |
|---|---|---|
| GMM1 权重 | `[E,2I,H]` | `[E,2I,H/2]` |
| GMM2 权重 | `[E,H,I]` | `[E,H,I/2]` |
| GMM1 scale | `[E,2I,ceil(H/64),2]` | 相同 |
| GMM2 scale | `[E,H,ceil(I/64),2]` | 相同 |

四列表必须统一布局，`E=1` 时也按 Tensor 维数区分。FP8 权重 dtype 与模式一致，FP4 用 `uint8` 打包；所有模式的 scale 均为必填 ND `torch.float8_e8m0fnu`。权重和 scale 必须连续，并与 `x` 在同一设备上。

| quant_mode | GMM1 格式 | GMM2 格式 |
|---|---|---|
| `"mx_fp8_e5m2"` | ND | ND |
| `"mx_fp8_e4m3"` | ND / FRACTAL_NZ | ND / FRACTAL_NZ |
| `"mx_fp4_e2m1"` | FRACTAL_NZ | FRACTAL_NZ_C0_32 |
| `"mx_fp8_e4m3_w4"` | FRACTAL_NZ_C0_32 | FRACTAL_NZ_C0_32 |

调用方准备 CANN 所需布局和格式，封装不代做量化、转置或 format cast。FP4 类型通过 `torch_npu.float4_e2m1fn_x2`（整数 296）传入，不使用 torch dtype。

输入必须满足 CANN 对应平台的接口约束。调用方必须保证每个 token 的专家编号有效且不重复；封装不扫描专家编号的具体数值。

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
| **output** | `torch.Tensor` | `[bs, hidden]` | MegaMoe 输出：A3 padding 后裁回本地 `bs`；A5 不 padding、不裁切。 |
| **expert_token_nums** | `torch.Tensor` | `[num_local_experts]` | 当前 rank 上每个本地 expert 接收到的 token 数。 |
