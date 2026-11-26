# mHC pre mixing

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | Supported | Supported |
| `cann_ops_transformers_cann920_v2` | Supported | |
| `cann_recipes_infer_cann910` | Supported | Supported |
| `cann_recipes_infer_cann920` | | Supported |
| `torch_native` | Supported | Supported |

DeepSeek V4.1 requires a CANN 9.2 wheel that bundles the fused backend for the target architecture.

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

def mhc_pre_dpskv4(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps,
                   *, pre_mix=None, return_pre=False, impl='auto'):
    ...
```

When the caller supplies `pre_mix` or requests `return_pre=True`, `auto` selects CANN MhcPreSinkhornV2 on A3 and Recipes HcPre on A5. For calls with neither option, the initialized SDK selects the DeepSeek V4 path.

## Description

Input [T,n,C]; does not fuse the model input layernorm.

RMS-normalize the flattened streams for projection, form sigmoid pre/post gates and a Sinkhorn-normalized residual matrix. `y = Σ_i pre_i × x_i`. The activation preserves x.dtype; post and comb are FP32 in the reference.

A supplied `pre_mix` provides the weights for `y`. `return_pre=True` appends the pre weights computed from the current residual, including when `pre_mix` is supplied. Both options are available with the A3 and A5 fused backends and `torch_native`.

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
| pre_mix | Tensor | None | Keyword-only [T,n] FP32 weights for the y accumulation |
| return_pre | bool | False | Keyword-only; append the pre output when True |
| impl | str | 'auto' | Registered implementation, or `auto` for production dispatch |

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

Calls the wheel-bundled Recipes `npu_hc_pre_v2` on A5.

## cann_ops_transformers_cann920_v2

On A3, uses the wheel-bundled MhcPreSinkhornV2 with HF32 enabled. The wheel registers it as `aclnnSglMhcPreSinkhornV2` so the DeepSeek V4 and V4.1 kernels can coexist. The backend uses four residual streams, 20 Sinkhorn iterations and a hidden dimension aligned to 128. [CANN operator contract](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2.md)

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

x = torch.randn(1, 4, 5120, device="npu", dtype=torch.bfloat16)
fn = torch.randn(24, 4 * 5120, device="npu", dtype=torch.float32)
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_dpskv4(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl="torch_native")

# DeepSeek V4.1 with the previous sublayer's pre weights.
pre_mix = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=x.device)
y, post, comb, pre = mhc_pre_dpskv4(
    x, fn, scale, base, 4, 20, 1e-6, 1e-6,
    pre_mix=pre_mix,
    return_pre=True,
)
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_pre_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

The DeepSeek V4.1 cases run when the installed wheel bundles the fused backend.

[Upstream operators](https://github.com/Ascend/ops-transformer)
