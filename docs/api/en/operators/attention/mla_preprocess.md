# mla_preprocess

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Supported | Not supported |
| `torch_native` | Supported | Supported |

The CANN implementation and build are arch22-only; `torch_native` runs on any
architecture as an accuracy reference.

## Signature

```python
from sgl_kernel_npu.operators.attention import mla_preprocess

def mla_preprocess(
    hidden_state,
    gamma0,
    beta0,
    wdqkv,
    descale0,
    gamma1,
    beta1,
    wuq,
    descale1,
    gamma2,
    cos,
    sin,
    wuk,
    kv_cache,
    kv_cache_rope,
    slotmapping,
    *,
    quant_scale0,
    quant_offset0,
    bias0,
    quant_scale1,
    quant_offset1,
    bias1,
    ctkv_scale=None,
    q_nope_scale=None,
    cache_mode="krope_ctkv",
    quant_mode="per_tensor_quant_asymm",
    q_out0,
    kv_cache_out0,
    q_out1,
    kv_cache_out1,
    impl="auto",
):
    ...
```

`auto` selects `ascendc`; `torch_native` uses PyTorch operations.
An unavailable native implementation raises an error without switching backends.

The four output/cache tensors are caller-owned, written in place, and returned
in the same order: `(q_out0, kv_cache_out0, q_out1, kv_cache_out1)`.

## Description

The operator fuses the complete MLA (Multi-head Latent Attention) inference
pre-processing step into one call: RMSNorm, static quantization, quantized GEMM,
RoPE and paged KV-cache writes.

The computation is:

```text
q1        = round_fp16(hidden_state / quant_scale0 + quant_offset0)          # INT8
mm1       = dequant(wdqkv @ q1) + bias0                                      # [N, 2112]
latent, q = split(mm1, [576, 1536])
k_nope    = latent[:, :512]
k_pe      = latent[:, 512:]
q         = rmsnorm(q, gamma0) * gamma1 + beta1                              # 1536 wide
k_nope    = rmsnorm(k_nope, gamma2)                                          # 512 wide
q2        = round_fp16(q / quant_scale1 + quant_offset1)                     # INT8
mm2       = dequant(wuq @ q2) + bias1                                        # [N, headNum*192]
q_nope, q_pe = split(mm2.view(N, headNum, 192), [128, 64])
q_nope    = bmm(q_nope, wuk)                                                 # [N, headNum, 512]
q_pe      = rope(q_pe, cos, sin)
k_pe      = rope(k_pe, cos, sin)
```

`rmsnorm` uses `eps = 1e-6`. RoPE uses the half-rotation layout: each vector is
split into two halves and rotated to
`[x1*cos - x2*sin, x1*sin + x2*cos]`.

`cache_mode` selects the cache layout and output split:

| `cache_mode` | Cache layout | `q_out0` | `q_out1` |
| --- | --- | --- | --- |
| `kvcache` | ND; nope 512 and rope 64 combined into 576 | `[N, headNum, 576]` | Unused |
| `krope_ctkv` | ND; nope 512 and rope 64 written separately | `[N, headNum, 512]` | `[N, headNum, 64]` |
| `int8_nzcache` | NZ; nope INT8 in groups of 32 elements | `[N, headNum, 512]` INT8 | `[N, headNum, 64]` |
| `nzcache` | NZ; nope/rope in groups of 16 elements | `[N, headNum, 512]` | `[N, headNum, 64]` |

The NZ layout is `[blockNum, groups, blockSize, C0]`; `int8_nzcache` uses
`C0 = 32` for nope and `16` otherwise, and `blockSize` must be `128`.

`quant_mode` currently implements `per_tensor_quant_asymm` only (per-tensor
asymmetric quantization of `hidden_state`). Other values produce no useful
result.

## Parameters and return values

`N` is the token count, `H` the `hidden_state` width, `headNum` the query head
count, and `blockNum`/`blockSize` the paged-cache dimensions.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `hidden_state` | `[N, H]` | FP16 / BF16 | Input hidden state; `N` in `[1, 1024]`, `H` in `[2048, 8192]`. |
| `gamma0`, `beta0` | `[H]` | Input dtype | First RMSNorm weight and bias. |
| `wdqkv` | logical `[2112, H]` | INT8 | First quantized GEMM weight, FRACTAL_NZ on NPU. |
| `descale0` | `[2112]` | int64 for FP16, FP32 for BF16 | First GEMM dequantization scales. |
| `gamma1`, `beta1` | `[1536]` | Input dtype | Second RMSNorm weight and bias. |
| `wuq` | logical `[headNum*192, 1536]` | INT8 | Second quantized GEMM weight, FRACTAL_NZ on NPU. |
| `descale1` | `[headNum*192]` | int64 for FP16, FP32 for BF16 | Second GEMM dequantization scales. |
| `gamma2` | `[512]` | Input dtype | RMSNorm weight for `k_nope`. |
| `cos`, `sin` | `[N, 64]` | Input dtype | RoPE tables. |
| `wuk` | `[headNum, 128, 512]` | Input dtype | Q absorption weights. |
| `kv_cache` | See `cache_mode` | FP16 / BF16 / INT8 | KV nope cache; physical layout depends on `cache_mode`. |
| `kv_cache_rope` | `[blockNum, blockSize, 1, 64]` | Input dtype | KV rope cache; NZ physical layout in NZ modes. |
| `slotmapping` | `[N]` | int32, required | Paged slot written for each token. |
| `quant_scale0` | `[1]` | Input dtype | First quantization scale. |
| `quant_offset0` | `[1]` | INT8 | First quantization offset. |
| `bias0` | `[2112]` | INT32 | First GEMM bias. |
| `quant_scale1` | `[1]` | Input dtype | Second quantization scale. |
| `quant_offset1` | `[1]` | INT8 | Second quantization offset. |
| `bias1` | `[headNum*192]` | INT32 | Second GEMM bias. |
| `ctkv_scale` | `[1]` | Input dtype / `None` | `k_nope` quantization scale for `int8_nzcache`; ignored otherwise. |
| `q_nope_scale` | `[headNum]` | Input dtype / `None` | `q_nope` quantization scale for `int8_nzcache`; ignored otherwise. |
| `cache_mode` | — | `"krope_ctkv"` | One of `kvcache`, `krope_ctkv`, `int8_nzcache`, `nzcache`. |
| `quant_mode` | — | `"per_tensor_quant_asymm"` | Per-tensor asymmetric quantization only. |
| `q_out0` | See `cache_mode` | Input dtype / INT8 | `q_nope` output, written in place. |
| `kv_cache_out0` | Same as `kv_cache` | Same as `kv_cache` | The same tensor as `kv_cache`. |
| `q_out1` | `[N, headNum, 64]` | Input dtype | `q_pe` output, written in place. |
| `kv_cache_out1` | Same as `kv_cache_rope` | Same as `kv_cache_rope` | The same tensor as `kv_cache_rope`. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `q_out0` | See `cache_mode` | Input dtype / INT8 |
| `kv_cache_out0` | Same as `kv_cache` | Same as `kv_cache` |
| `q_out1` | `[N, headNum, 64]` | Input dtype |
| `kv_cache_out1` | Same as `kv_cache_rope` | Same as `kv_cache_rope` |

Usage notes:

- All tensors must be on the same NPU, and `slotmapping` must have length `N`.
- FP16 input supports `krope_ctkv` only; `int8_nzcache` and `nzcache` require
  BF16 input.
- `int8_nzcache` and `nzcache` require `blockSize = 128`; `int8_nzcache` also
  requires nonempty `ctkv_scale` and `q_nope_scale`.
- `wdqkv` and `wuq` must use FRACTAL_NZ storage. Convert with
  `torch_npu.npu_format_cast(weight, 29)` while
  `torch_npu.npu.config.allow_internal_format = True`.
- Computation matches the legacy operator. `torch_native` is an accuracy
  reference: it evaluates intermediates in FP32 and rounds to the target dtype
  at the same points as the legacy test path.
