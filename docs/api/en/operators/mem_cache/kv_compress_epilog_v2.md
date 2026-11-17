# kv_compress_epilog_v2

## Chip support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_recipes_infer` | Not supported | Supported (CANN 9.2.x) |
| `torch_native` | Supported | Supported |

## Prototype

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

def kv_compress_epilog_v2(
    cache, x, slot_mapping, *,
    quant_group_size=32, quant_mode="mxfp8_bf16",
    round_scale=True, x_scale=1.0, impl="auto",
): ...
```

The default `auto` selects `cann_recipes_infer` (the bundled CANN 9.2.0 recipes kernel, arch35 only).

## Description

Quantizes full-width BF16 KV rows and scatters them in place into the paged KV cache for the DeepSeek-V4.1 non-packed (non-448+64) KV write path on Ascend 950. Every `quant_group_size` columns form one group: in MXFP8 mode the scale is `max(amax, 1e-4) / FP8 max`, rounded up to a power of two when `round_scale=True`, and values are divided by the scale and saturating-cast to FP8; in MXFP4 mode the OCP MX convention gives scale = 2^(floor(log2 amax) − 2) with values saturating-cast to e2m1 (two nibbles per byte, even element in the low nibble). Cache rows hold `[values | uint16 scales]`: layout 1 (2D cache) zero-pads each row to 32 bytes; layout 2 (4D cache) stores rows compactly inside block pages with a 32-byte-aligned block stride (`cache.stride(0)`). Slots that are negative or beyond the cache are skipped and untouched regions keep their original bytes.

## Parameters and return value

All input and output tensors share one NPU (`torch_native` also works on CPU as a correctness reference).

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `cache` | `torch.Tensor` | `[N, C]` or `[B, S, 1, C]` | `float8_e4m3fn` / `float8_e5m2` (MXFP8) or `uint8` (MXFP4) | required | Caller-allocated cache. 2D must be contiguous with `C ≥ RoundUp32(values + scales)`; 4D requires dim 2 == 1, a contiguous last dimension, row width `C ≥ values + scales`, and a 32-byte-aligned block stride. |
| `x` | `torch.Tensor` | `[T, D]` | `bfloat16` | required | Contiguous input with `0 < D ≤ 8192` and `D % quant_group_size == 0`. |
| `slot_mapping` | `torch.Tensor` | `[T]` | `int32` / `int64` | required | Destination slot per row; negative or out-of-range slots are skipped. |
| `quant_group_size` | `int` | — | — | `32` | MXFP8 requires 32; MXFP4 accepts 16 or 32. |
| `quant_mode` | `str` | — | — | `"mxfp8_bf16"` | `"mxfp8_bf16"` or `"mxfp4_bf16"` (case-insensitive, surrounding whitespace ignored). |
| `round_scale` | `bool` | — | — | `True` | Whether MXFP8 scales round up to a power of two; ignored by MXFP4. |
| `x_scale` | `float` | — | — | `1.0` | Reserved; must be 1.0. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"cann_recipes_infer"`, `"torch_native"`. |

No return value; `cache` is updated in place.

## cann_recipes_infer backend

arch35 only and requires an initialized CANN 9.2.x SDK; the kernel ships in the wheel's bundled cann-recipes-infer vendor (`torch.ops.sgl_cann_recipes_infer.kv_compress_epilog_v2`). A missing binding raises a clear error instead of falling back to another backend. Input validation happens in that binding; cache row width and block stride are checked by host tiling.

## torch_native backend

An independent eager PyTorch reference that rewrites the same regions on the input device following the kernel's bit-level semantics (E8M0-style scales, RINT FP8/FP4 casts, paged addressing); for correctness checks only — it never substitutes for the production backend and is not guaranteed graph-capturable. Non-finite inputs are not bit-aligned with the hardware.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

x = torch.randn(128, 512, device="npu", dtype=torch.bfloat16)
slots = torch.arange(128, device="npu", dtype=torch.int32)
slots[-1] = -1
# MXFP4-g16 full-width row: 256 value bytes + 32 groups x 2 scale bytes = 320.
cache = torch.full((1, 128, 1, 320), 0, device="npu", dtype=torch.uint8)
kv_compress_epilog_v2(cache, x, slots, quant_group_size=16,
                      quant_mode="mxfp4_bf16", impl="auto")
```

Build and install the current wheel and initialize the SDK first. The example demonstrates the interface only and does not replace NPU accuracy, performance, or end-to-end validation.

```bash
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog_v2.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

These tests run only on CANN 9.2.x with arch35: a version mismatch skips the whole file, and arch22 is skipped by the registry.

[Upstream operator repository](https://gitcode.com/cann/cann-recipes-infer)
