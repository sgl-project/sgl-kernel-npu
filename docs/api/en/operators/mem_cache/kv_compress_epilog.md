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

## SDK-selected DSV4 packed cache

### Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | unsupported | supported |
| `ascendc` | unsupported | supported |
| `torch_native` | unsupported | supported |

### Signature

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(kv_compress_cache, x, slot_mapping, *, quant_group_size=64, quant_mode=None, round_scale=True, x_scale=1.0, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

### Description

A5 only; 64-value groups and 448+64 input width. Allocate with get_npu_kv_quant_path() before calling.

Quantize NoPE in groups and scatter packed rows, keeping the RoPE tail unquantized. Negative slot -1 is skipped; slot 0 is valid. Compare semantic bytes, excluding alignment padding.

### Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| cache | Tensor | required | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | required | int32 [T] destination slot; -1 skips a write |
| quant_group_size | int | 64 | 64 for migrated packed KV |
| quant_mode | int / str | None | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| round_scale | bool | True | Scale rounding; CANN 9.1 normal FP8 ignores it |
| x_scale | float | 1.0 | Additional vendor quantization scale; migrated packed KV requires 1.0 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### Return value

| Name | Type | Contract |
| --- | --- | --- |
| cache | Tensor | The same mutated packed input cache. |

### cann_ops_transformers_cann920

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

### ascendc

Source: [sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp). The C++ wrapper directly launches the checked-in Ascend C kernel; this is not inferred from the Torch namespace. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

### torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

### Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

from sgl_kernel_npu.operators.mem_cache import get_npu_kv_quant_path
path = get_npu_kv_quant_path()  # Resolve before allocation; A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
kv_compress_epilog(cache, x, slots, impl="auto")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
