# ARCH22 fused W8A8 grouped matmul, SwiGLU and quantization

This CANN 9.2 operator is registered for ARCH22 (A2/A3). It reuses the existing
`ops_transformer` profile, pinned CANN source, vendor packaging and `OpBuilder`.
No vLLM source or installation is required.

```bash
python setup.py --arch-version 2201 --ops apply_token_bitmask --third-party ops_transformer -j8
```

The profile adds an isolated GMM vendor containing both `ascend910b` and
`ascend910_93` kernels. A small patch supplies clipped SwiGLU, which the pinned
upstream W8A8 API lacks. Private operator and ACLNN names preserve the toolkit ABI.
Existing mainline/experimental vendors retain their own source revisions.

The Torch bridge follows the existing CANN `OpBuilder` flow: the first call may
compile it into the shared Torch extension cache. Warm up before graph capture,
as for the other CANN operators; graph replay does not compile the bridge.

```python
from sgl_kernel_npu.operators.moe import grouped_matmul_swiglu_quant
```

`grouped_matmul_swiglu_quant(x, weight, weight_scale, x_scale, group_list, swiglu_limit=0.0, *, impl="auto")`
accepts `auto`/`cann_ops_transformers_cann920` (ARCH22 CANN) or `torch_native` (independent eager reference).

The operator fuses grouped INT8 matrix multiplication, dequantization, SwiGLU and
per-token dynamic quantization:

- `x`: INT8 `[R,K]` activations.
- `weight`: INT8 NZ weights, logical shape `[E,K,N]`.
- `weight_scale`: BF16 `[E,N]` channel dequantization scales.
- `x_scale`: FP32 `[R]` activation dequantization scales.
- `group_list`: cumulative INT64 `[E]` expert ends on device.
- `swiglu_limit`: positive values clamp gate above by `swiglu_limit` and up to `[-swiglu_limit,swiglu_limit]`;
  zero disables clipping.

Returns INT8 `[R,N/2]` and FP32 `[R]` dequantization scales. Dynamic quantization
uses per-row maximum absolute value divided by 127, then rounds and saturates to
INT8. Only rows before `group_list[-1]` are valid; padding contents are unspecified.
All tensors must be on the same NPU. W4A8 packed weights are unsupported.

To prepare NZ weights, enable `torch.npu.config.allow_internal_format = True`
before `torch_npu.npu_format_cast(weight, 29)`. The API does not change that
process setting. The native path supports graph capture and changed-input replay;
the eager reference reads expert ends on the host and must not be captured.

Hardware tests compare dequantized valid outputs against the independent reference
(relative RMS < 0.01), including empty/uneven expert groups, clipping enabled/disabled,
and changed activations/expert ends during graph replay. Bitwise equality with an
unfused implementation that materializes BF16 intermediates is not guaranteed.

Hardware numerical and graph-replay validation was performed on A3; A2 kernel compilation and packaging were verified, but an A2 device run is pending.
