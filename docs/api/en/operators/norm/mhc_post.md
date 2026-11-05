# mHC post mixing

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | declared; validate SDK kernel | declared; validate SDK kernel |
| `torch_native` | declared; validate SDK kernel | declared; validate SDK kernel |

## Signature

```python
from sgl_kernel_npu.operators.norm import mhc_post

def mhc_post(x, residual, post, comb, *, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

Combines a layer output with n residual streams; public operand order is stable across SDKs.

`out[...,i,d] = post[...,i] × x[...,d] + Σ_j comb[...,j,i] × residual[...,j,d]`. Accumulate in FP32 in the reference and cast to x.dtype.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| residual | Tensor | required | [T,n,C] residual streams |
| post | Tensor | required | [T,n] post gates |
| comb | Tensor | required | [T,n,n] residual matrix |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| output | Tensor | [T,n,C], x.dtype. |

## cann_ops_transformer

Retains the existing registration name; the compatibility wrapper selects original or released bindings by SDK. Metadata/compute pairing and A5 KV reader/writer format checks run before allocation or execution.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import mhc_post

x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
residual = torch.randn(1, 4, 128, device="npu", dtype=x.dtype)
post = torch.ones(1, 4, device="npu", dtype=torch.float32)
comb = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
y = mhc_post(x, residual, post, comb, impl="torch_native")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/norm/test_mhc_post.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
