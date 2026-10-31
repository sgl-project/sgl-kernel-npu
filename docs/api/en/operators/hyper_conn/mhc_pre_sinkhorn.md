# mhc_pre_sinkhorn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| cann | Unverified | Unverified |
| torch_native | Unverified on NPU | Unverified on NPU |

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-6, *, impl="auto"):
    ...
```

`auto` selects `cann`; no fallback.

## Description

Normalize flattened streams in FP32, project with phi, apply sigmoid to pre/post gates (post multiplier 2), then alternate row/column Sinkhorn normalization. No output RMSNorm is fused.

For u = flatten(x), z = (u / sqrt(mean(u**2) + norm_eps)) @ phi.T. Split z into pre/post/res, scale by alpha[0:3] and add the corresponding bias.

## Parameters and return value

| Parameter | Type | Constraint | Default |
| --- | --- | --- | --- |
| x | Tensor | [T,N,D], FP16/BF16 | required |
| phi | Tensor | [(2+N)*N,N*D], FP32 | required |
| alpha | Tensor | [3], FP32 | required |
| bias | Tensor | [(2+N)*N], FP32 | required |
| hc_mult | int | Number of streams N; backend-defined support | 4 |
| num_iters | int | Sinkhorn iterations; backend-defined support | 20 |
| hc_eps | float | Forwarded to backend | 1e-6 |
| norm_eps | float | Forwarded to backend | 1e-6 |
| impl | str | auto / cann / torch_native | auto |

The facade forwards inputs without imposing additional dtype, stride, device, shape or scalar restrictions. The selected backend determines supported inputs and reports errors. The examples use GLM stream count N=4 and FP16/BF16 activations with FP32 parameters. T=0 returns empty tensors without launching CANN (activation dtype for hin, FP32 for post/res).

| Return | Type | Shape and dtype |
| --- | --- | --- |
| output | tuple[Tensor,Tensor,Tensor] | (hin [T,D] in vendor dtype for cann, hpost [T,N] FP32, hres [T,N,N] FP32) |

## cann

Requires CANN 9.1's `cann_ops_transformer.ops.mhc_pre_sinkhorn` public API. Imported and cached on first nonempty execution. Missing dependency and execution errors propagate; legacy custom_ops is never used. No architecture whitelist is imposed; this does not establish hardware support or completed NPU validation.

The facade expands [T,N,D] to [T,1,N,D] for the vendor BSND wrapper and squeezes the singleton sequence dimension of each output. Output dtypes are preserved; any activation conversion belongs to the SGLang caller, matching GitCode glmx PR #21.

## torch_native

Row softmax + hc_eps; divide by column sum + hc_eps; repeat (row sum + hc_eps, column sum + hc_eps) num_iters-1 times. hin = sum_i (sigmoid(pre_i)+hc_eps)*x_i.

FP32 intermediates, output activation cast to input dtype. An independent correctness baseline, not an automatic fallback. Accuracy thresholds and performance remain subject to NPU acceptance.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
phi = torch.randn(24, 16384, device="npu", dtype=torch.float32) / 128
alpha = torch.ones(3, device="npu", dtype=torch.float32)
bias = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_sinkhorn(x, phi, alpha, bias, impl="cann")
```
