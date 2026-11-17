# mHC pre mixing

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | Supported | Supported |
| `cann_recipes_infer_cann910` | Supported | Supported |
| `cann_recipes_infer_cann920` | Unsupported | Supported |
| `torch_native` | Supported | Supported |

`cann_recipes_infer_cann920` additionally requires an initialized CANN 9.2.x SDK and serves the DSV4.1 `pre_mix` input and pre output; it is registered on arch35 only, and selecting it on ARCH22 raises `UnsupportedGenerationError`.

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

def mhc_pre_dpskv4(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps,
                   *, pre_mix=None, return_pre=False, impl='auto'):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend. The DSV4.1 fused kernel must be selected explicitly as `cann_recipes_infer_cann920`.

## Description

Input [T,n,C]; does not fuse the model input layernorm.

RMS-normalize the flattened streams for projection, form sigmoid pre/post gates and a Sinkhorn-normalized residual matrix. `y = Σ_i pre_i × x_i`. The activation preserves x.dtype; post and comb are FP32 in the reference.

A supplied `pre_mix` is used directly as the pre weights for the y accumulation instead of being derived from the projection; `return_pre=True` appends the pre weights as a fourth return value. Both options require `cann_recipes_infer_cann920` or `torch_native`.

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
| pre_mix | Tensor | None | Keyword-only; [T,n] FP32 pre weights, requires `cann_recipes_infer_cann920` or `torch_native` |
| return_pre | bool | False | Keyword-only; append the pre output when True |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| y | Tensor | [T,C], x.dtype. |
| post | Tensor | [T,n], reference FP32. |
| comb | Tensor | [T,n,n], reference FP32. |
| pre | Tensor | [T,n] FP32; returned only with `return_pre=True`. |

## cann_ops_transformers_cann920

Uses the CANN 9.2 binding and rejects a different initialized SDK. `auto` dispatches by the cached SDK version. A5 KV reader/writer formats are derived from the same version before allocation; each operator requires only its own binding. No pre output.

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnHcPre](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_pre.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## cann_recipes_infer_cann920

DSV4.1 fused path, CANN 9.2.x and A5 only; `npu_hc_pre_v2` is built for arch35. Calls the wheel-bundled `torch.ops.sgl_cann_recipes_infer.npu_hc_pre_v2`, supporting the `pre_mix` input and the pre output. A missing binding raises a clear error instead of falling back.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

x = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
fn = torch.randn(24, 512, device="npu", dtype=torch.bfloat16)
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_dpskv4(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl="torch_native")

# DSV4.1 fused kernel: needs a CANN 9.2.0 wheel with the recipes vendor.
y, post, comb, pre = mhc_pre_dpskv4(
    x, fn, scale, base, 4, 20, 1e-6, 1e-6,
    return_pre=True, impl="cann_recipes_infer_cann920",
)
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_pre_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

The `cann_recipes_infer_cann920` DSV4.1 cases run only on CANN 9.2.x with arch35 and are skipped elsewhere; the baseline cases use `auto`, whose backend follows the SDK version.

[Upstream operators](https://github.com/Ascend/ops-transformer)
