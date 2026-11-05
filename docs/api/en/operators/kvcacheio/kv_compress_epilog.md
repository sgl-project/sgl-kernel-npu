# Packed KV cache writer

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | unsupported | supported |
| `cann_9_1` | unsupported | supported |
| `torch_native` | unsupported | supported |

## Signature

```python
from sgl_kernel_npu.operators.kvcacheio import kv_compress_epilog

def kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64, quant_mode=None, round_scale=True, x_scale=1.0, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

A5 only; 64-value groups and 448+64 input width. Allocate with get_npu_kv_quant_path() before calling.

Quantize NoPE in groups and scatter packed rows, keeping the RoPE tail unquantized. Negative slot -1 is skipped; slot 0 is valid. Compare semantic bytes, excluding alignment padding.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| cache | Tensor | required | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | required | int32 [T] destination slot; -1 skips a write |
| quant_group_size | int | 64 | 64 for migrated packed KV |
| quant_mode | int / str | None | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| round_scale | bool | True | Scale rounding; cann_9_1 normal FP8 ignores it |
| x_scale | float | 1.0 | Additional vendor quantization scale; migrated packed KV requires 1.0 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| cache | Tensor | The same mutated packed input cache. |

## cann_ops_transformer

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

## cann_9_1

Uses the CANN 9.1 binding and performs input conversion in this operator module. It requires the initialized CANN 9.1 SDK and never retries a CANN 9.2 binding.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.kvcacheio import kv_compress_epilog

from sgl_kernel_npu.operators.kvcacheio import get_npu_kv_quant_path
path = get_npu_kv_quant_path()  # Resolve before allocation; A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
kv_compress_epilog(cache, x, slots, impl="auto")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/kvcacheio/test_kv_compress_epilog.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
