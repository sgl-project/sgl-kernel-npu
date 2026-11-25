# In-place partial RoPE

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | supported | supported |
| `cann_recipes_infer_cann910` | supported | supported |
| `torch_native` | supported | supported |

## Signature

```python
from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

def inplace_partial_rotary_mul(x, cos, sin, *, rotary_mode='interleave', partial_slice=None, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

Mutates only the selected feature slice. Clone x before comparing implementations.

On each rotary pair `(a,b)`, compute `(a cos − b sin, b cos + a sin)`. The rotary_mode controls interleaved or half-pair ordering; other features are preserved.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| cos | Tensor | required | Broadcast-compatible cosine factors for selected rotary features |
| sin | Tensor | required | Broadcast-compatible sine factors, matching cos |
| rotary_mode | str / int | 'interleave' | RoPE public API: interleave or half; compressor uses native integer mode |
| partial_slice | sequence[int] / None | None | [begin,end), None selects the full last dimension; even rotary width |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| x | Tensor | The same mutated input tensor. |

## cann_ops_transformers_cann920

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnInplacePartialRotaryMul](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/inplace_partial_rotary_mul.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

x = torch.randn(1, 1, 128, device="npu", dtype=torch.bfloat16)
cos = torch.ones(1, 1, 64, device="npu", dtype=x.dtype)
sin = torch.zeros_like(cos)
y = inplace_partial_rotary_mul(x, cos, sin, partial_slice=[64, 128], impl="torch_native")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/rope/test_inplace_partial_rotary_mul.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
