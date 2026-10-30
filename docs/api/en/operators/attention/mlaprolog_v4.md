# mlaprolog_v4

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann` | Supported | Supported |
| `torch_native` | Supported | Supported |

Both implementations are registered for `Generation.ARCH22` and `Generation.ARCH35`, and `auto` is an alias of `cann`.

On arch22, the supported scope is BF16+C16 with and without RoPE. The MXFP8+C16 and MXFP8+C8 cases remain limited to arch35. INT8 is outside the independent reference and test coverage. See the broader CANN V4 [upstream hardware contract](https://gitcode.com/cann/ops-transformer/blob/d462156cf5d97712fcfeef95070d649f47d3bb6e/attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md).

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

With RoPE enabled, adjacent projection channels form rotation pairs. The rotated even channels occupy the first half of `query_rope` and `kr_cache`, followed by the rotated odd channels in the second half. The sine/cosine tensors follow this half ordering and are shared across heads for each token; pass ordinary sine values without negating the first half. Rotation uses FP32 arithmetic on the projection results (BF16 in mode 0, FP32 in mode 3), then converts the rotated results to BF16. With RoPE disabled, the projected components retain their original channel order.

For a slot `s=cache_index[t]` in `PA_BSND`, the logical destination is `[s // P, s % P, 0, :]`, where `P` is the page size. `PA_NZ` writes the equivalent logical slot in tiled storage. Only the addressed cache slots are updated; callers own the lifetime and capacity of both cache tensors.

The independent reference covers the following cases with and without RoPE. `C16` means both caches use BF16; `C8` means `kv_cache` uses FP8 and `kr_cache` uses BF16.

| Case | `weight_quant_mode` | `kv_cache_quant_mode` | `query_quant_mode` | `query` / `kv_cache` / `kr_cache` |
| --- | --- | --- | --- | --- |
| BF16 projections, C16 | `0` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 projections, C16 | `3` | `0` | `0` | BF16 / BF16 / BF16 |
| MXFP8 projections, C8 | `3` | `1` | `1` | FP8 E4M3FN / FP8 E4M3FN / BF16 |

Mode 3 uses block-scaled FP8 inputs and weights. Its normalized query latent is converted to BF16 with `CAST_ROUND` (nearest, ties away from zero), then quantized in groups of 32 with E8M0 scales. FP8 conversion rounds to nearest, ties to even, with saturation to `[-448, 448]`. The reference follows these CANN conversion rules; fused reduction order can still cause numerical differences.

For C8, `quant_scale_ckv` is a **multiplicative quantization factor**, the reciprocal of the KV dequantization scale. Query quantization uses a dynamic scale for each token and head. After optional RoPE and BF16 conversion, `query_rope` receives inverse Q/KV scale compensation for downstream quantized attention; do not apply that compensation again. `kr_cache` remains BF16. An all-zero query row produces a zero Q scale; the reference follows CANN without replacing it by one or adding an epsilon.

With `query_norm_flag=True`, mode 3 returns the **FP8 query latent and its E8M0 group scales even in C16 mode**. A BF16 attention query does not imply that `query_norm` is BF16.

## Parameters and return value

All CANN tensor inputs are on the same NPU. The shapes below are logical shapes; `FRACTAL_NZ` is a physical format, not a request to reshape a weight manually. The reference accepts CPU or NPU tensors, performs its calculation on CPU, returns outputs to `token_x.device`, and copies cache updates back to the original buffers.

For the reference-supported cases, use `C=512`, `R=64`, `D` in `{128, 192}`, `Rq` in `{1536, 2048}`, and a valid CANN input width and local head count. Its separate-cache layouts are `PA_BSND` or `PA_NZ`. The CANN backend retains the broader upstream contract; passing a mode number does not make every combination of modes valid.

| Parameter | Type / logical shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `token_x` | Tensor `[T,H]` or `[B,S,H]`; BF16 in mode 0, FP8 E4M3FN in mode 3 | Required | Contiguous ND activations; `T=B*S` for the flattened description. |
| `weight_dq` | Tensor `[H,Rq]`; BF16 or FP8 E4M3FN | Required | Query down-projection. CANN requires FRACTAL_NZ; keep the logical two-dimensional shape. |
| `weight_uq_qr` | Tensor `[Rq,N*(D+R)]`; BF16 or FP8 E4M3FN | Required | Up-projection weights for `q_c` and `q_r`, concatenated per head; FRACTAL_NZ for CANN. |
| `weight_uk` | Tensor `[N,D,C]`; BF16 | Required | Absorbed key up-projection, contiguous ND. |
| `weight_dkv_kr` | Tensor `[H,C+R]`; BF16 or FP8 E4M3FN | Required | Projects input `X` into KV latent `c_0` and key component `k_r`; FRACTAL_NZ for CANN. |
| `rmsnorm_gamma_cq` | Tensor `[Rq]`; BF16 | Required | Query latent RMSNorm weight, contiguous ND. |
| `rmsnorm_gamma_ckv` | Tensor `[C]`; BF16 | Required | KV latent RMSNorm weight, contiguous ND. |
| `kv_cache` | Tensor `[pages,P,1,C]`; BF16 for C16, FP8 E4M3FN for C8 | Required | Mutable shared latent cache. `PA_NZ` uses the corresponding physical tile order. |
| `kr_cache` | Tensor `[pages,P,1,R]`; BF16 | Required | Mutable cache for the R-dimensional projected key component `k_r`. The CANN-only packed modes have different storage requirements. |
| `rope_sin`, `rope_cos` | Optional BF16 tensors `[T,R]` or `[B,S,R]` | `None` | Contiguous ND, with the same shape as each other and the token dimensions of `token_x`. Two nonempty tensors enable RoPE; two absent/empty arguments disable it. Only one nonempty raises `ValueError`. |
| `cache_index` | Optional tensor `[T]` or `[B,S]`; int32/int64 | `None` | Required for the supported PA cases. Each entry identifies a valid logical slot; use distinct slots for a deterministic reference comparison. |
| `dequant_scale_x` | Optional tensor `[T,H/32]`; E8M0 in mode 3 | `None` | Mode-3 input block scales. Other CANN quantization modes use their upstream scale layout. |
| `dequant_scale_w_dq` | Optional tensor `[Rq,H/32]`; E8M0 in mode 3 | `None` | Mode-3 down-projection weight scales, contiguous ND. |
| `dequant_scale_w_uq_qr` | Optional tensor `[N*(D+R),Rq/32]`; E8M0 in mode 3 | `None` | Mode-3 query up-projection weight scales, contiguous ND. |
| `dequant_scale_w_dkv_kr` | Optional tensor `[C+R,H/32]`; E8M0 in mode 3 | `None` | Mode-3 dequantization scales for `weight_dkv_kr`, contiguous ND. |
| `quant_scale_ckv` | Optional tensor `[1]`; FP32 for C8 | `None` | Positive multiplicative factor `1/s_c`, required in the C8 case. CANN per-channel mode uses `[1,C]`. |
| `quant_scale_ckr` | Optional tensor `[1,R]`; FP32 | `None` | CANN per-channel quantization input for `kr_cache`; reference requires `None`. |
| `smooth_scales_cq` | Optional tensor `[1,Rq]`; FP32 | `None` | CANN query-latent smoothing input for applicable modes; reference requires `None`. |
| `actual_seq_len` | Optional tensor `[B]`; int32 | `None` | Cumulative sequence lengths for applicable CANN block-indexed cache modes; reference requires `None`. |
| `k_nope_clip_alpha` | Optional tensor `[1]`; FP32 | `None` | CANN clipping factor for applicable grouped quantization modes; reference requires `None`. |
| `rmsnorm_epsilon_cq`, `rmsnorm_epsilon_ckv` | `float` | `1e-5` | Query/KV RMSNorm epsilon. |
| `cache_mode` | `str` | `"PA_BSND"` | Reference: `PA_BSND` or `PA_NZ`; CANN also exposes its other documented cache layouts. |
| `query_norm_flag` | `bool` | `False` | Whether to return the normalized query latent and, when quantized, its scales. |
| `weight_quant_mode` | `int` | `0` | CANN: `0` BF16, `1` partial INT8, `2` dynamic INT8, `3` MXFP8, `4` FP8, `5` HiFloat8. Reference: only `0` and `3`, as listed above. |
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
| `query_norm` | `[T,Rq]` or `[B,S,Rq]`; otherwise empty | Mode 0: BF16; mode 3: FP8 E4M3FN | Normalized query latent when requested. |
| `dequant_scale_q_norm` | `[T,Rq/32]` or `[B*S,Rq/32]`; otherwise empty | E8M0 for mode 3 | MXFP8 query-latent group scales; empty in mode 0 or when not requested. |

## cann backend

Lazily imports `cann_ops_transformer.mla_prolog` and forwards its arguments. This document is based on ops-transformer commit [`d462156cf5d97712fcfeef95070d649f47d3bb6e`](https://gitcode.com/cann/ops-transformer/commit/d462156cf5d97712fcfeef95070d649f47d3bb6e), dated 2026-10-20 (UTC+8).

## torch_native backend

Uses independent PyTorch CPU calculations as an accuracy reference for the three cases above, with and without RoPE, and `PA_BSND`/`PA_NZ` caches. Outputs return to the input device, and both supplied caches are updated in place.

## Example

This arch35 example uses BF16 projections and separate BF16 caches. Dimensions follow the [Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json), with 6 local Q heads at attention TP16; tensor values are randomly generated. `use_rope=False` selects Kimi's no-RoPE behavior. Set it to `True` to exercise RoPE using the same dimensions.

Accuracy cases cover token counts `1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 31, 32, 40, 48, 56, 63, 64`, each with local Q-head counts of 96/48/24/12/6/3, all three modes above, both `PA_BSND`/`PA_NZ` cache layouts, and RoPE enabled/disabled. On arch22, MXFP8 cases are skipped before input allocation; arch35 retains all three modes. Paired cases use the same generated inputs and weights. Additional T5 cases cover negative cache slots. Performance cases select TP4/TP16 shapes with `1, 4, 16, 32, 64` tokens in `PA_BSND`, also with and without RoPE. Existing precision criteria are retained, including independent Q scale checks and restoring the C8 `query_rope` compensation before applying its BF16 accuracy budget.

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
