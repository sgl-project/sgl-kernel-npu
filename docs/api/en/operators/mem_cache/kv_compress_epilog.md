# kv_compress_epilog

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Unsupported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(
    kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode,
    round_scale_flag, layout, *, impl="auto",
): ...
```

`auto` selects `ascendc`.

## Description

Quantize KV-cache rows and scatter them to the slots selected by slot_mapping, as used by the DeepSeek-V4 NPU cache path. For each row, the first D - 64 columns are quantized while the final 64 BF16 RoPE columns are copied byte for byte. Mode 1 uses FP8 values with FP32 scales for each 128-value group; mode 2 uses MXFP8 values with E8M0 exponent scales for groups of 64 or 128 values. Layout 1 stores a packed record per slot, while layout 2 separates the value and scale regions within each cache block. A slot of -1 skips the row; unwritten slots retain their contents. The function updates the cache in place and returns None.

## Parameters and return value

Use tensors on the same NPU.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `kv_compress_cache` | `torch.Tensor` | Layout-dependent | `float8_e4m3fn` / `float8_e5m2` | Required | Caller-owned packed cache. Layout 1 accepts slot rows, including [slots, 1, width]. Layout 2 uses [blocks, block_size, width] or [blocks, block_size, 1, width]. |
| `x` | `torch.Tensor` | `[N, D]` | `bfloat16` | Required | Dense input rows with N > 0 and a 64-column RoPE suffix. |
| `slot_mapping` | `torch.Tensor` | `[N]` | `int32` / `int64` | Required | Destination slot for each input row. A value of -1 skips that row. |
| `quant_group_size` | `int` | — | — | Required | MXFP8 group width, either 64 or 128. FP8 uses fixed groups of 128. |
| `quant_mode` | `int` | — | — | Required | 1 selects FP8 with FP32 scales. 2 selects MXFP8 with E8M0 exponent bytes. |
| `round_scale_flag` | `bool` | — | — | Required | In MXFP8 mode, increment each nonzero biased scale exponent by one when true. |
| `layout` | `int` | — | — | Required | 1 stores one slot per row using the cache row stride. FP8 packs quantized values, RoPE, then scales. MXFP8 packs RoPE, quantized values, then scales. 2 is MXFP8 only and packs all value rows before all scale rows in each block. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. Choices are `"auto"`, `"ascendc"`, `"torch_native"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | `NoneType` | Mutates the supplied cache. Unwritten slots retain their contents. |

## ascendc backend

Available on arch35. In layout 1, round each packed row width up to 128 bytes. In layout 2, reserve the scale count rounded up to 8 bytes per slot. Only the actual group exponents in a written slot have specified values.

## torch_native backend

Packs rows on the input device using the selected cache layout.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

x = torch.ones((2, 192), device="npu", dtype=torch.bfloat16)
slots = torch.tensor([1, -1], device="npu", dtype=torch.int32)
cache = torch.zeros((2, 384), device="npu", dtype=torch.uint8).view(torch.float8_e4m3fn)
kv_compress_epilog(cache, x, slots, quant_group_size=64, quant_mode=2,
                   round_scale_flag=True, layout=1, impl="ascendc")
```
