# mhc_post

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| cann | Unverified | Unverified |
| torch_native | Unverified on NPU | Unverified on NPU |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_post

def mhc_post(x, h_res, h_out, h_post, *, impl="auto"):
    ...
```

`auto` selects `cann`; no fallback.

## Description

Compute y[t,i,d] = h_post[t,i] * h_out[t,d] + sum_j h_res[t,i,j] * x[t,j,d] in FP32, then cast once to x.dtype.

## Parameters and return value

| Parameter | Type | Constraint | Default |
| --- | --- | --- | --- |
| x | Tensor | [T,N,D], FP16/BF16 | required |
| h_res | Tensor | [T,N,N], FP32 | required |
| h_out | Tensor | [T,D], same dtype as x | required |
| h_post | Tensor | [T,N], FP32 | required |
| impl | str | auto / cann / torch_native | auto |

The facade forwards inputs without imposing additional dtype, stride, device, shape or scalar restrictions. The selected backend determines supported inputs and reports errors. The examples use GLM stream count N=4 and FP16/BF16 activations with FP32 parameters. T=0 returns empty tensors without launching CANN (same dtype as x).

| Return | Type | Shape and dtype |
| --- | --- | --- |
| output | Tensor | y [T,N,D], vendor dtype for cann |

## cann

Requires CANN 9.1's `cann_ops_transformer.ops.mhc_post` public API. Imported and cached on first nonempty execution. Missing dependency and execution errors propagate; legacy custom_ops is never used. No architecture whitelist is imposed; this does not establish hardware support or completed NPU validation.

Argument order matches the vendor API: residual streams, residual matrix, layer output, post gates.

## torch_native

Independent torch.bmm residual mapping plus post gating; no vendor calls.

FP32 intermediates, output activation cast to input dtype. An independent correctness baseline, not an automatic fallback. Accuracy thresholds and performance remain subject to NPU acceptance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_post

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
h_res = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
h_out = torch.zeros(1, 4096, device="npu", dtype=x.dtype)
h_post = torch.ones(1, 4, device="npu", dtype=torch.float32)
y = mhc_post(x, h_res, h_out, h_post, impl="cann")
```
