# mhc_pre_sinkhorn

## Hardware support

| Implementation | arch22 (A3) | arch35 (950DT) |
| --- | --- | --- |
| cann_ops_transformer | Verified | Verified |
| cann_ops_transformer_v2 | Bundled vendor | — |
| torch_native | Verified on NPU | Verified on NPU |

CANN 9.1.0 supports small hidden sizes such as 128 and 256 on A3 and Ascend950. On Ascend950, CANN 9.2.0 only supports hidden sizes 4096 and 7168. The GLM-5.3-Flash hidden size of 4096 is supported by both versions.

## Signature

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-6, *, return_pre=False, pre_mix=None, impl="auto"):
    ...
```

`auto` selects `cann_ops_transformer`; no fallback.

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
| `return_pre` | bool | Append FP32 hpre [T,N] (sigmoid + hc_eps) when true | False |
| `pre_mix` | Tensor or None | FP32 [T,N] on the same device. Weights hin while the operator computes the current coefficients from x. Use `cann_ops_transformer_v2` or `torch_native`. | None |
| impl | str | auto / cann_ops_transformer / cann_ops_transformer_v2 / torch_native | auto |

The facade forwards inputs without imposing additional dtype, stride, device, shape or scalar restrictions. The selected backend determines supported inputs and reports errors. The examples use GLM stream count N=4 and FP16/BF16 activations with FP32 parameters. Empty/IDLE short-circuits remain the SGLang caller's responsibility, as in glmx PR #21; the facade adds none.

| Return | Type | Shape and dtype |
| --- | --- | --- |
| output | tuple[Tensor,Tensor,Tensor] | (hin [T,D] in vendor dtype for cann_ops_transformer, hpost [T,N] FP32, hres [T,N*N] FP32) |

The three-output ABI is the default. With `return_pre=True`, the operator appends hpre for mixing in the next V4.1 sublayer. With `pre_mix` omitted, the CANN backend requests the backward intermediates to obtain hpre.

## cann_ops_transformer

Loads the bundled binding on first use. Wheels with the V2 vendor use it for calls without `pre_mix` and preserve the existing output tuple. Other wheels use the bundled BSND binding by expanding [T,N,D] to [T,1,N,D] and squeezing the outputs.

## cann_ops_transformer_v2

The A3 `ops_transformer` and `ops_transformer_dsv4` build profiles ship the same mHC pre vendor. The native binding uses TND and enables HF32. With `pre_mix` and `return_pre=True`, it returns the current hpre without allocating the backward intermediates. [Pinned ACLNN API](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2_en.md)

## torch_native

Row softmax + hc_eps; divide by column sum + hc_eps; repeat (row sum + hc_eps, column sum + hc_eps) num_iters-1 times. hin = sum_i (sigmoid(pre_i)+hc_eps)*x_i.

FP32 intermediates, output activation cast to input dtype. An independent correctness baseline, not an automatic fallback.

CANN uses HF32 projection. On Ascend910_9362 with CANN 9.1, the FP32 post/comb comparison uses atol=5e-5 and rtol=5e-5, calibrated over nine seeds, FP16/BF16 inputs, both eps cases and the five nonzero random-data shapes. FP16 activation comparisons use atol=1e-5, rtol=2e-3; BF16 uses atol=1e-5, rtol=1e-2, avoiding the shared absolute tolerance of 0.05 masking errors in small activations. The same thresholds passed on both tested devices.

Inputs are synthetic with borrowed model-config dimensions, not captured activations or checkpoint weights, and passing them does not establish full-model accuracy.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
phi = torch.randn(24, 16384, device="npu", dtype=torch.float32) / 128
alpha = torch.ones(3, device="npu", dtype=torch.float32)
bias = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_sinkhorn(x, phi, alpha, bias, impl="cann_ops_transformer")

# A3 V4.1, with FP32 weights from the previous sublayer.
previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="npu")
y, post, comb, pre = mhc_pre_sinkhorn(
    x, phi, alpha, bias, pre_mix=previous, return_pre=True,
    impl="cann_ops_transformer_v2",
)
```
