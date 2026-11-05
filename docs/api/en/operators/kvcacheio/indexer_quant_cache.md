# Indexer quantization cache writer

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | declared; validate SDK kernel | declared; validate SDK kernel |
| `torch_native` | declared; validate SDK kernel | declared; validate SDK kernel |

## Signature

```python
from sgl_kernel_npu.operators.kvcacheio import indexer_quant_cache

def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl='auto', **kwargs):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

The reference supports finite normal-FP8 rows up to 128 values; A3 availability depends on installed kernel binaries.

`scale = max(abs(x))/448`, `payload = cast_fp8(clamp(x/scale))`. CANN 9.1 uses unrounded FP32 scales and division. CANN 9.2 honors round_scale (round up to a power of two) and uses FP32 reciprocal multiplication. Zero rows store scale=0. x_scale is HiFloat8-only.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| cache | Tensor | required | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| scale_cache | Tensor | required | Mutable FP32 indexer block scales, one per cache row |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | required | int32 [T] destination slot; -1 skips a write |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### Keyword arguments

| Name | Type | Contract |
| --- | --- | --- |
| quant_mode / round_scale / x_scale | str / bool / float | Defaults fp8 / True / 1.0; reference supports normal FP8 only. |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| cache, scale_cache | Tensor tuple | The same mutated input buffers. |

## cann_ops_transformer

Retains the existing registration name; the compatibility wrapper selects original or released bindings by SDK. Metadata/compute pairing and A5 KV reader/writer format checks run before allocation or execution.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.kvcacheio import indexer_quant_cache

# empty avoids unsupported NPU zero_ on FP8; the selected row is overwritten.
cache = torch.empty(1, 128, 1, 128, device="npu", dtype=torch.float8_e4m3fn)
scales = torch.empty(1, 128, 1, 1, device="npu", dtype=torch.float32)
x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
indexer_quant_cache(cache, scales, x, slots, impl="auto")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/kvcacheio/test_indexer_quant_cache.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
