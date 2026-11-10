# mHC post mixing

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | supported | supported |
| `cann_recipes_infer_cann910` | supported | supported |
| `torch_native` | supported | supported |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_post_dpskv4

def mhc_post_dpskv4(x, residual, post, comb, *, impl='auto'):
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

## cann_ops_transformers_cann920

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding.

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnHcPost](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_post.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_post_dpskv4

x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
residual = torch.randn(1, 4, 128, device="npu", dtype=x.dtype)
post = torch.ones(1, 4, device="npu", dtype=torch.float32)
comb = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
y = mhc_post_dpskv4(x, residual, post, comb, impl="torch_native")
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_post_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
