# Fused Deep MoE API

`Buffer.fused_deep_moe(...)` is the unified fused MoE entrypoint in DeepEP-Ascend.
It supports three execution modes:


- `FuseMode.FUSED_DEEP_MOE`: DeepEP `aclnnFusedDeepMoe`
- `FuseMode.DISPATCH_FFN_COMBINE`: DeepEP `aclnnDispatchFFNCombine`
- `FuseMode.MEGA_MOE`: `cann_ops_transformer.ops.mega_moe`

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
| `FuseMode.FUSED_DEEP_MOE` | `1` | `aclnnFusedDeepMoe` | Full fusion: Dispatch + GMM1 + activation/quantization + GMM2 + Dequant + Unpermute/Combine in a single AscendC kernel. The A5 path supports SwiGLU and SiTU. |
| `FuseMode.DISPATCH_FFN_COMBINE` | `2` | `aclnnDispatchFFNCombine` | Integrated routing (`MoeInitRoutingQuantV2`) + AllToAll + GMM1 + DequantSwigluQuant + GMM2 + Dequant + Combine in a single AscendC kernel. |
| `FuseMode.MEGA_MOE` | `3` | `cann_ops_transformer.ops.mega_moe` | MegaMoe fused dispatch + expert FFN + combine. |

> [!NOTE]
> `FuseMode` is **not** exported from the package's top-level `__init__.py`. Import it explicitly:
> ```python
> from deep_ep.buffer import FuseMode
> ```
> Or use integer values directly: `1` (FUSED_DEEP_MOE), `2` (DISPATCH_FFN_COMBINE), or `3` (MEGA_MOE).

#### Key Differences Between DeepEP Fuse Modes

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
    gmm1_permuted_weight: Union[torch.Tensor, List[torch.Tensor]],
    gmm1_permuted_weight_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    gmm2_weight: Union[torch.Tensor, List[torch.Tensor]],
    gmm2_weight_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    activation: Optional[str] = "swiglu",
    beta: Optional[float] = 4.0,
    linear_beta: Optional[float] = 25.0,
    profile_enable: bool = False,
    *,
    l1_bias: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    l2_bias: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

## FuseMode Routing

| `fuse_mode` | Actual implementation |
|---|---|
| `FuseMode.FUSED_DEEP_MOE` | DeepEP `aclnnFusedDeepMoe` |
| `FuseMode.DISPATCH_FFN_COMBINE` | DeepEP `aclnnDispatchFFNCombine` |
| `FuseMode.MEGA_MOE` | `cann_ops_transformer.ops.mega_moe` |

`activation="swiglu_gpt_oss"` requires `FuseMode.MEGA_MOE`.

## Parameter Reference

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| **x** | `torch.Tensor` | `[bs, hidden]` | Input token representations, where each row is the hidden vector of a token. On A3 this is typically `bfloat16`; on A5 the fused host op supports both `bfloat16` and `float16`. **bs** range **[1, 256]**. **hidden** range **[512, 7168]**. |
| **topk_idx** | `torch.Tensor` | `[bs, num_topk]` | Expert indices for each token. Python converts it to `int32` before launch. A value of `-1` indicates the token is not dispatched. |
| **topk_weights** | `torch.Tensor` | `[bs, num_topk]` | Weighting coefficients for aggregating expert outputs (`float32`). |
| **gmm1_permuted_weight** | `torch.Tensor` or `list[torch.Tensor]` | FuseMode-dependent | First-stage weights. DeepEP modes require tensor form. `MEGA_MOE` accepts a leading local-expert dimension or one tensor per local expert. |
| **gmm1_permuted_weight_scale** | `torch.Tensor`, `list[torch.Tensor]`, or `None` | FuseMode-dependent | DeepEP modes require tensor form. It is optional for `MEGA_MOE` A16W16 and required for quantized MegaMoe execution. |
| **gmm2_weight** | `torch.Tensor` or `list[torch.Tensor]` | FuseMode-dependent | Second-stage weights. DeepEP modes require tensor form; `MEGA_MOE` also accepts one tensor per local expert. |
| **gmm2_weight_scale** | `torch.Tensor`, `list[torch.Tensor]`, or `None` | FuseMode-dependent | Follows the same mode rules as `gmm1_permuted_weight_scale`. |
| **num_max_dispatch_tokens_per_rank** | `int` | Scalar | For A3, used in the existing fused-path buffer sizing logic. For A5, this value is also used as per-rank **capacity**, and must be **greater than or equal to local bs**. |
| **num_experts** | `int` | Scalar, range **(0, 512]** | Total number of global experts. On A5 fused path, current tiling requires `num_experts` to be divisible by EP rank size. |
| **quant_mode** | `int` | Scalar, default `1` | Quantization mode attribute passed to the fused operator. A3 follows the legacy fused-path semantics. On A5, this parameter is currently not effective in the public fused path: activation quantization follows the weight quantization type, so the practical supported combinations are `w8a8` and `w4a4`. `w4a8` is not supported, and non-quantized model weights are not supported in the current A5 fused path. |
| **fuse_mode** | `FuseMode` | Scalar, default `FuseMode.FUSED_DEEP_MOE` | Selects the DeepEP fused kernel or MegaMoe implementation. |
| **activation** | `Optional[str]` | Scalar, default `"swiglu"` | DeepEP supports `"swiglu"` and `"situ"`; SiTU requires `FUSED_DEEP_MOE`. MegaMoe additionally supports `"swiglu_gpt_oss"`. |
| **beta** | `Optional[float]` | Scalar, default `4.0` | Soft-saturation bound for the SiTU gate branch. `None` uses the internal default `4.0`. It must be greater than zero when SiTU is selected. |
| **linear_beta** | `Optional[float]` | Scalar, default `25.0` | Optional soft-saturation bound for the SiTU up branch. A positive value enables the transformation; `None` leaves the up branch unchanged. |
| **profile_enable** | `bool` | Scalar, default `False` | Whether to enable fused-kernel profiling for the current launch. It only takes effect when profiling has been started in advance (begin_profile). |

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
    fuse_mode=FuseMode.MEGA_MOE,
    activation="situ",
    beta=4.0,
    linear_beta=25.0,
)
```

- `FuseMode.FUSED_DEEP_MOE`: use DeepEP
- `FuseMode.DISPATCH_FFN_COMBINE`: use DeepEP; SiTU is not supported
- `FuseMode.MEGA_MOE`: use MegaMoe

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
- DeepEP SiTU is supported by the A5 fused kernel; the A3 DeepEP runtime rejects it. MegaMoe provides the A3 SiTU path.
- For SiTU, `beta` must be greater than zero. When `linear_beta` is provided, it must also be greater than zero.

#### For `fuse_mode=DISPATCH_FFN_COMBINE` (mode=2)

- Constraints follow the `aclnnDispatchFFNCombine` path and differ from `FUSED_DEEP_MOE`.
- Shared expert is not supported.
- Only SwiGLU is supported. Selecting `activation="situ"` raises `NotImplementedError`.

### DeepEP modes

- Supports `FuseMode.FUSED_DEEP_MOE`
- Supports `FuseMode.DISPATCH_FFN_COMBINE`
- Supports `activation="swiglu"` and `activation="situ"`; SiTU requires `FUSED_DEEP_MOE`
- Uses legacy `quant_mode`
- Requires tensor-form legacy fused weights/scales
- Does not support `l1_bias` or `l2_bias`

### `FuseMode.MEGA_MOE`

- Selected with `FuseMode.MEGA_MOE`
- Requires `cann_ops_transformer`
- Supports `activation="swiglu"`, `activation="swiglu_gpt_oss"`, and `activation="situ"`
- Interprets legacy parameter names as mega_moe inputs:
  - `gmm1_permuted_weight -> l1_weights`
  - `gmm1_permuted_weight_scale -> l1_weights_sf`
  - `gmm2_weight -> l2_weights`
  - `gmm2_weight_scale -> l2_weights_sf`
- Accepts either:
  - a tensor whose leading dimension is the local expert count
  - or `list[Tensor]` with one tensor per local expert

## Parameter Notes

| Parameter | Notes |
|---|---|
| `x` | `[bs, hidden]` token tensor. |
| `topk_idx` | `[bs, num_topk]` routing indices. `-1` is allowed. |
| `topk_weights` | `[bs, num_topk]` combine weights. |
| `gmm1_permuted_weight` | Legacy name kept for compatibility. On `mega_moe`, it must contain first linear weights in layout `[hidden, 2 * intermediate_hidden]` per expert. |
| `gmm1_permuted_weight_scale` | Optional on `mega_moe` A16W16. Required on `mega_moe` A8W8-INT/A8W4-INT. |
| `gmm2_weight` | Legacy name kept for compatibility. On `mega_moe`, it must contain second linear weights in layout `[intermediate_hidden, hidden]` per expert. |
| `gmm2_weight_scale` | Optional on `mega_moe` A16W16. Required on `mega_moe` A8W8-INT/A8W4-INT. |
| `num_max_dispatch_tokens_per_rank` | EP dispatch capacity hint shared across ranks. |
| `num_experts` | Global expert count. On `mega_moe`, it must be divisible by the process-group size. |
| `quant_mode` | Public quantization selector. MegaMoe maps it to its internal dispatch quantization mode. |
| `fuse_mode` | `FUSED_DEEP_MOE` and `DISPATCH_FFN_COMBINE` select DeepEP; `MEGA_MOE` selects MegaMoe. |
| `activation` | `mega_moe` supports `"swiglu"`, `"swiglu_gpt_oss"`, and `"situ"`. `deep_ep` supports `"swiglu"` and `"situ"`; SiTU requires `FUSED_DEEP_MOE`. |
| `linear_beta` | SiTU up-branch soft-saturation bound. It is passed through MegaMoe activation parameters. |
| `l1_bias`, `l2_bias` | Optional A8W4-INT compensation biases for `mega_moe` only. |

## Mega MoE Quantized Scenes

### A16W16

- `quant_mode=0`
- `gmm1_permuted_weight_scale=None`
- `gmm2_weight_scale=None`
- `l1_bias=None`
- `l2_bias=None`

### A8W8-INT

- `quant_mode=1` (internally mapped to INT8 dispatch)
- `gmm1_permuted_weight_scale` required
- `gmm2_weight_scale` required
- `l1_bias=None`
- `l2_bias=None`

### A8W4-INT

- `quant_mode=1` (internally mapped to INT8 dispatch)
- `gmm1_permuted_weight_scale` required
- `gmm2_weight_scale` required
- `l1_bias` required
- `l2_bias` required

## `situ` Activation

`situ` is not implemented in `buffer.py` itself. It is forwarded to DeepEP with
`FuseMode.FUSED_DEEP_MOE`, or to MegaMoe with `FuseMode.MEGA_MOE`.

- Public API name: `linear_beta`
- MegaMoe activation parameter name: `linear_beta`
- DeepEP SiTU requires `linear_beta > 0` when it is provided
- MegaMoe receives `linear_beta` through its activation parameters

## Return Value

`fused_deep_moe(...)` returns `(output, aux)`:

- `output`: fused MoE output with shape `[bs, hidden]`
- `aux`:
  - `deep_ep + FuseMode.FUSED_DEEP_MOE`: `ep_recv_count`
  - `deep_ep + FuseMode.DISPATCH_FFN_COMBINE`: `expert_token_nums`
  - `mega_moe`: `expert_token_nums`

## Dependency Note

The `mega_moe` backend depends on:

```python
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
```

If that package is not available, the DeepEP modes are unchanged, but calls using
`FuseMode.MEGA_MOE` raise an import error with an explicit message.
