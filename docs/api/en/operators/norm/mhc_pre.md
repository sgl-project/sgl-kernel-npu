# mHC pre mixing

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | declared; validate SDK kernel | declared; validate SDK kernel |
| `torch_native` | declared; validate SDK kernel | declared; validate SDK kernel |

## Signature

```python
from sgl_kernel_npu.operators.norm import mhc_pre

def mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

Input [T,n,C]; does not fuse the model input layernorm.

RMS-normalize the flattened streams for projection, form sigmoid pre/post gates and a Sinkhorn-normalized residual matrix. `y = Σ_i pre_i × x_i`. The activation preserves x.dtype; post and comb are FP32 in the reference.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Input activation; shape described below; production device NPU |
| fn | Tensor | required | [n*n+2*n,n*C] projection matrix |
| scale | Tensor | required | [3] gate/matrix scales |
| base | Tensor | required | [n*n+2*n] projection bias |
| hc_mult | int | required | Number n of residual streams |
| sinkhorn_iters | int | required | Positive Sinkhorn iteration count |
| hc_eps | float | required | Gate and Sinkhorn stability epsilon |
| rms_norm_eps | float | required | Internal projection RMS epsilon, not the model layernorm |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| y | Tensor | [T,C], x.dtype. |
| post | Tensor | [T,n], reference FP32. |
| comb | Tensor | [T,n,n], reference FP32. |

## cann_ops_transformer

Retains the existing registration name; the compatibility wrapper selects original or released bindings by SDK. Metadata/compute pairing and A5 KV reader/writer format checks run before allocation or execution.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import mhc_pre

x = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
fn = torch.randn(24, 512, device="npu", dtype=torch.bfloat16)
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl="torch_native")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/norm/test_mhc_pre.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
