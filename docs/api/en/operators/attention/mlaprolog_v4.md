# mlaprolog_v4

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann` | Supported | Supported |
| `torch_native` | Supported | Supported |

**CANN backend requirement: CANN 9.2.0.**

Both implementations are registered for `Generation.ARCH22` and `Generation.ARCH35`, and `auto` is an alias of `cann`.

On arch22, the supported scope is BF16+C16 and dynamic INT8+C16, both with and without RoPE. The MXFP8+C16 and MXFP8+C8 cases remain limited to arch35. See the broader CANN V4 [upstream hardware contract](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md).

## Signature

```python
from typing import Optional, Tuple

import torch

from sgl_kernel_npu.operators.attention import mlaprolog_v4

def mlaprolog_v4(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

`impl="auto"` and `impl="cann"` call the public `cann_ops_transformer.mla_prolog` API, which wraps `aclnnMlaPrologV4WeightNz`. Other arguments preserve that API's names and defaults. This is the in-place cache API with **five outputs**, not its seven-output functional variant.

RoPE is enabled when `rope_sin` and `rope_cos` are both nonempty tensors. It is disabled when both are absent or empty. If only one is nonempty, the public entry point raises `ValueError` before backend dispatch.

## Description

Computes the MLA projection and normalization steps before attention, optionally applying RoPE to the R-dimensional query and key components. It returns the absorbed query (`query`) and the query component (`query_rope`), and updates the shared KV latent cache (`kv_cache`) and the R-dimensional key cache (`kr_cache`) **in place**. It does not compute attention scores, softmax, or the final value projection.

**Why a separate integration:** Kimi K3 MLA does not use RoPE. The [torch_npu V3 implementation](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp) inspected on 2026-10-28 calls `aclnnMlaPrologV3WeightNz`, whose entry point [always enables RoPE](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp). This API uses V4 through `cann_ops_transformer.mla_prolog` to expose both RoPE and native no-RoPE operation.

Dimensions used below are `H` for the input width, `Rq` for the query latent width, `C` for the KV latent width, `N` for the local query-head count, `D` for the query width before absorption, and `R` for the projected query/key component width.

With RoPE enabled, adjacent projection channels form rotation pairs. The rotated even channels occupy the first half of `query_rope` and `kr_cache`, followed by the rotated odd channels in the second half. The sine/cosine tensors follow this half ordering and are shared across heads for each token; pass ordinary sine values without negating the first half. Rotation uses FP32 arithmetic on the projection results (BF16 in mode 0, dequantized FP32 in mode 2, FP32 in mode 3), then converts the rotated results to BF16. Mode 2 does not round the dequantized projections to BF16 before RoPE. With RoPE disabled, the projected components retain their original channel order.

For a slot `s=cache_index[t]` in `PA_BSND`, the logical destination is `[s // P, s % P, 0, :]`, where `P` is the page size. `PA_NZ` writes the equivalent logical slot in tiled storage. Only the addressed cache slots are updated; callers own the lifetime and capacity of both cache tensors.

The independent reference covers the following BF16, dynamic INT8 and MXFP8 cases, each with and without RoPE. `C16` means both caches use BF16; `C8` means `kv_cache` uses FP8 and `kr_cache` uses BF16.

| Case | `weight_quant_mode` | `kv_cache_quant_mode` | `query_quant_mode` | `query` / `kv_cache` / `kr_cache` |
| --- | --- | --- | --- | --- |
| BF16 projections, C16 | `0` | `0` | `0` | BF16 / BF16 / BF16 |
| Dynamic INT8 projections, C16 | `2` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 projections, C16 | `3` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 projections, C8 | `3` | `1` | `1` | FP8 E4M3FN / FP8 E4M3FN / BF16 |

Mode 2 takes INT8 activations and three INT8 projection weights, with FP32 per-token activation scales and per-output-channel weight scales. Matrix products accumulate in INT32 and are dequantized to FP32 before normalization. The normalized query latent is dynamically quantized per token using nearest-even rounding, without an intermediate BF16 conversion. `weight_uk`, the absorbed query and both caches remain BF16. The caller quantizes `token_x` before invoking the operator; no INT8 KV cache is involved.

Mode 3 uses block-scaled FP8 inputs and weights. Its normalized query latent is converted to BF16 with `CAST_ROUND` (nearest, ties away from zero), then quantized in groups of 32 with E8M0 scales. FP8 conversion rounds to nearest, ties to even, with saturation to `[-448, 448]`. The reference follows these CANN conversion rules; fused reduction order can still cause numerical differences.

For C8, `quant_scale_ckv` is a **multiplicative quantization factor**, the reciprocal of the KV dequantization scale. Query quantization uses a dynamic scale for each token and head. After optional RoPE and BF16 conversion, `query_rope` receives inverse Q/KV scale compensation for downstream quantized attention; do not apply that compensation again. `kr_cache` remains BF16. An all-zero query row produces a zero Q scale; the reference follows CANN without replacing it by one or adding an epsilon.

With `query_norm_flag=True`, mode 2 returns INT8 `query_norm` and FP32 per-token scales; mode 3 returns the **FP8 query latent and its E8M0 group scales even in C16 mode**. A BF16 attention query does not imply that `query_norm` is BF16.

## Parameters and return value

All CANN tensor inputs are on the same NPU. The shapes below are logical shapes; `FRACTAL_NZ` is a physical format, not a request to reshape a weight manually. The reference accepts CPU or NPU tensors, performs its calculation on CPU, returns outputs to `token_x.device`, and copies cache updates back to the original buffers.

For the reference-supported cases, use `C=512`, `R=64`, `D` in `{128, 192}`, `Rq` in `{1536, 2048}`, and a valid CANN input width and local head count. Its separate-cache layouts are `PA_BSND` or `PA_NZ`. The CANN backend retains the broader upstream contract; passing a mode number does not make every combination of modes valid.

| Parameter | Type / logical shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `token_x` | Tensor `[T,H]` or `[B,S,H]`; mode 0: BF16, mode 2: INT8, mode 3: FP8 E4M3FN | Required | Contiguous ND activations; `T=B*S` for the flattened description. |
| `weight_dq` | Tensor `[H,Rq]`; BF16, INT8 or FP8 E4M3FN | Required | Query down-projection. CANN requires FRACTAL_NZ; keep the logical two-dimensional shape. |
| `weight_uq_qr` | Tensor `[Rq,N*(D+R)]`; BF16, INT8 or FP8 E4M3FN | Required | Up-projection weights for `q_c` and `q_r`, concatenated per head; FRACTAL_NZ for CANN. |
| `weight_uk` | Tensor `[N,D,C]`; BF16 | Required | Absorbed key up-projection, contiguous ND. |
| `weight_dkv_kr` | Tensor `[H,C+R]`; BF16, INT8 or FP8 E4M3FN | Required | Projects input `X` into KV latent `c_0` and key component `k_r`; FRACTAL_NZ for CANN. |
| `rmsnorm_gamma_cq` | Tensor `[Rq]`; BF16 | Required | Query latent RMSNorm weight, contiguous ND. |
| `rmsnorm_gamma_ckv` | Tensor `[C]`; BF16 | Required | KV latent RMSNorm weight, contiguous ND. |
| `kv_cache` | Tensor `[pages,P,1,C]`; BF16 for C16, FP8 E4M3FN for C8 | Required | Mutable shared latent cache. `PA_NZ` uses the corresponding physical tile order. |
| `kr_cache` | Tensor `[pages,P,1,R]`; BF16 | Required | Mutable cache for the R-dimensional projected key component `k_r`. The CANN-only packed modes have different storage requirements. |
| `rope_sin`, `rope_cos` | Optional BF16 tensors `[T,R]` or `[B,S,R]` | `None` | Contiguous ND, with the same shape as each other and the token dimensions of `token_x`. Two nonempty tensors enable RoPE; two absent/empty arguments disable it. Only one nonempty raises `ValueError`. |
| `cache_index` | Optional tensor `[T]` or `[B,S]`; int32/int64 | `None` | Required for the supported PA cases. Each entry identifies a valid logical slot; use distinct slots for a deterministic reference comparison. |
| `dequant_scale_x` | Mode 2: FP32 `[T,1]`; mode 3: E8M0 `[T,H/32]` | `None` | Required in modes 2/3: per-token or block input scales, respectively. |
| `dequant_scale_w_dq` | Mode 2: FP32 `[1,Rq]`; mode 3: E8M0 `[Rq,H/32]` | `None` | Required down-projection weight scales in modes 2/3, contiguous ND. |
| `dequant_scale_w_uq_qr` | Mode 2: FP32 `[1,N*(D+R)]`; mode 3: E8M0 `[N*(D+R),Rq/32]` | `None` | Required query up-projection weight scales in modes 2/3, contiguous ND. |
| `dequant_scale_w_dkv_kr` | Mode 2: FP32 `[1,C+R]`; mode 3: E8M0 `[C+R,H/32]` | `None` | Required dequantization scales for `weight_dkv_kr` in modes 2/3, contiguous ND. |
| `quant_scale_ckv` | Optional tensor `[1]`; FP32 for C8 | `None` | Positive multiplicative factor `1/s_c`, required in the C8 case. CANN per-channel mode uses `[1,C]`. |
| `quant_scale_ckr` | Optional tensor `[1,R]`; FP32 | `None` | CANN per-channel quantization input for `kr_cache`; reference requires `None`. |
| `smooth_scales_cq` | Optional tensor `[1,Rq]`; FP32 | `None` | CANN query-latent smoothing input for applicable modes; reference requires `None`. |
| `actual_seq_len` | Optional tensor `[B]`; int32 | `None` | Cumulative sequence lengths for applicable CANN block-indexed cache modes; reference requires `None`. |
| `k_nope_clip_alpha` | Optional tensor `[1]`; FP32 | `None` | CANN clipping factor for applicable grouped quantization modes; reference requires `None`. |
| `rmsnorm_epsilon_cq`, `rmsnorm_epsilon_ckv` | `float` | `1e-5` | Query/KV RMSNorm epsilon. |
| `cache_mode` | `str` | `"PA_BSND"` | Reference: `PA_BSND` or `PA_NZ`; CANN also exposes its other documented cache layouts. |
| `query_norm_flag` | `bool` | `False` | Whether to return the normalized query latent and, when quantized, its scales. |
| `weight_quant_mode` | `int` | `0` | CANN: `0` BF16, `1` partial INT8, `2` dynamic INT8, `3` MXFP8, `4` FP8, `5` HiFloat8. Reference: `0`, `2` and `3`, as listed above. |
| `kv_cache_quant_mode` | `int` | `0` | CANN: `0` non-quantized, `1` per-tensor, `2` per-channel, `3` per-token/group. Reference: `0`, or `1` with weight/query modes `3/1`. |
| `query_quant_mode` | `int` | `0` | `0` non-quantized; `1` per-token/head quantization. See the supported combinations above. |
| `ckvkr_repo_mode`, `quant_scale_repo_mode` | `int` | `0` | CANN storage modes: separate (`0`) or packed (`1`). Reference requires `0`. |
| `tile_size` | `int` | `128` | Group tile size for applicable CANN KV quantization modes; not an MXFP8 projection group size and not the cache page size. |
| `qc_qr_scale`, `kc_scale` | `float` | `1.0` | Query and shared-latent correction factors. |
| `token_x_dtype`, `weight_dq_dtype`, `weight_uq_qr_dtype`, `weight_dkv_kr_dtype`, `kv_cache_dtype` | Optional `int` | `None` | CANN logical-dtype overrides, such as HiFloat8 stored as bytes; reference requires `None`. |
| `impl` | `str` | `"auto"` | Keyword-only; `auto`, `cann`, or `torch_native`. |

All five results are tensors. Disabled optional outputs are empty tensors, not extra cache outputs.

| Return value | Shape | Dtype in reference-supported cases | Meaning |
| --- | --- | --- | --- |
| `query` | `[T,N,C]` or `[B,S,N,C]` | BF16 for C16; FP8 E4M3FN for C8 | Absorbed query. |
| `query_rope` | `[T,N,R]` or `[B,S,N,R]` | BF16 | R-dimensional query component after optional RoPE; includes scale compensation in C8 mode. |
| `dequant_scale_q_nope` | `[T,N,1]` or `[B*S,N,1]`; otherwise empty | FP32 | Per-token/head query dequantization scale in C8 mode. |
| `query_norm` | `[T,Rq]` or `[B,S,Rq]`; otherwise empty | Mode 0: BF16; mode 2: INT8; mode 3: FP8 E4M3FN | Normalized query latent when requested. |
| `dequant_scale_q_norm` | Mode 2: `[T,1]`; mode 3: `[T,Rq/32]`, with `T=B*S` for 3D inputs | Mode 2: FP32; mode 3: E8M0 | Per-token INT8 or group MXFP8 query-latent scales; empty in mode 0 or when not requested. |

## cann backend

Lazily imports `cann_ops_transformer.mla_prolog` and forwards its arguments. This document is based on ops-transformer commit [`d462156cf5d97712fcfeef95070d649f47d3bb6e`](https://gitcode.com/cann/ops-transformer/commit/d462156cf5d97712fcfeef95070d649f47d3bb6e), dated 2026-10-20 (UTC+8).

## torch_native backend

Uses independent PyTorch CPU calculations as an accuracy reference for the cases above, with and without RoPE, and `PA_BSND`/`PA_NZ` caches. Outputs return to the input device, and both supplied caches are updated in place.

## Example

This arch35 example uses BF16 projections and separate BF16 caches. Dimensions follow the [Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json), with 6 local Q heads at attention TP16; tensor values are randomly generated. `use_rope=False` selects Kimi's no-RoPE behavior. Set it to `True` to exercise RoPE using the same dimensions.

Accuracy tests use 96, 24 and 6 local Q heads (attention TP1/4/16) in the representative groups below. The four modes are BF16+C16, dynamic INT8+C16, MXFP8+C16 and MXFP8+C8. Each group covers both RoPE states unless stated otherwise; these are selected combinations, not a full cross-product.

| Group | Local Q heads | Tokens | Layout | Modes | Cases |
| --- | --- | --- | --- | --- | ---: |
| Performance shapes | 24, 6 | 1, 4, 16, 32, 64 | PA_BSND | All four | 80 |
| NZ layout | 6 | 1, 16, 64 | PA_NZ | All four | 24 |
| TP1 shape | 96 | 4 | PA_BSND | All four | 8 |
| Normalized-query output disabled | 6 | 17 | PA_BSND | Three C16 modes | 6 |
| Negative padding slot | 6 | 5 | PA_BSND, PA_NZ | All four | 16 |
| Remaining token lengths, no RoPE | 6 | 2, 3, 7, 8, 12, 24, 31, 40, 48, 56, 63 | PA_BSND | BF16+C16 | 11 |

The matrix contains 145 numerical/cache cases and 16 direct-CANN parity cases: 161 collected cases for one implementation/reference pair. Of these, 123 apply to arch35 and 87 to arch22; the remaining architecture-specific cases are skipped. Dynamic INT8+C16 cases target arch22, while MXFP8 cases target arch35. Paired cases use the same generated inputs and weights.

Performance selection is unchanged: TP4/TP16 shapes with `1, 4, 16, 32, 64` tokens in `PA_BSND`, both RoPE states, and the applicable modes (60 configurations on arch35, 40 on arch22). Use `--enable-perf=false` for accuracy-only CI runs.

BF16/MXFP8 precision criteria are unchanged. For INT8 `query_norm`, every element must differ by at most one integer code; FP32 scales are checked independently with `atol=1e-5, rtol=1e-5`. This is a test-specific INT8 criterion, not the FP8 mixed-tolerance rule or a claim of an official CANN threshold. Public-API/direct-CANN parity remains byte-exact.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import mlaprolog_v4

device = "npu"
dtype = torch.bfloat16
T, H, Rq, N, D, C, R, P = 1, 7168, 1536, 6, 128, 512, 64, 128

def nz_weight(shape):
    weight = torch.randn(shape, device=device, dtype=dtype) * 0.01
    return torch_npu.npu_format_cast(weight, 29)

x = torch.randn(T, H, device=device, dtype=dtype)
w_dq = nz_weight((H, Rq))
w_uq_qr = nz_weight((Rq, N * (D + R)))
w_uk = torch.randn(N, D, C, device=device, dtype=dtype) * 0.01
w_dkv_kr = nz_weight((H, C + R))
gamma_q = torch.ones(Rq, device=device, dtype=dtype)
gamma_k = torch.ones(C, device=device, dtype=dtype)
slots = torch.tensor([0], device=device, dtype=torch.int64)
kv_cache = torch.zeros(1, P, 1, C, device=device, dtype=dtype)
kr_cache = torch.zeros(1, P, 1, R, device=device, dtype=dtype)

use_rope = False
rope_sin = rope_cos = None
if use_rope:
    positions = torch.arange(1, T + 1, dtype=torch.float32)
    inv_freq = 10000.0 ** (-torch.arange(0, R, 2, dtype=torch.float32) / R)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    rope_sin = angles.sin().to(device=device, dtype=dtype)
    rope_cos = angles.cos().to(device=device, dtype=dtype)

args = (x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_q, gamma_k)
kwargs = dict(
    rope_sin=rope_sin,
    rope_cos=rope_cos,
    cache_index=slots,
    cache_mode="PA_BSND",
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    query_norm_flag=True,
)

# auto selects the CANN V4 entry point and updates these two caches.
query, query_rope, q_scale, q_norm, q_norm_scale = mlaprolog_v4(
    *args, kv_cache, kr_cache, **kwargs, impl="auto"
)
print(query.shape, query_rope.shape, q_norm.shape)

# Use fresh buffers because the API mutates the caches supplied to it.
reference = mlaprolog_v4(
    *args,
    torch.zeros_like(kv_cache),
    torch.zeros_like(kr_cache),
    **kwargs,
    impl="torch_native",
)
print(reference[0].shape, q_scale.numel(), q_norm_scale.numel())
```
