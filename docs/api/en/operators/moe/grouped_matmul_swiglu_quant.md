# A2/A3 fused W8A8 and W4A8 grouped matmul, SwiGLU and quantization

This CANN 9.2 operator is registered for A2/A3. It reuses the existing
`ops_transformer` profile, pinned CANN source, vendor packaging and `OpBuilder`.
No vLLM source or installation is required.

```bash
python setup.py --arch-version 2201 --ops apply_token_bitmask --third-party ops_transformer -j8
```

The profile adds an isolated GMM vendor containing both `ascend910b` and
`ascend910_93` kernels. A small patch supplies clipped SwiGLU, which the pinned
upstream operator lacks. Private operator and ACLNN names preserve the toolkit ABI.
Existing mainline/experimental vendors retain their own source revisions.

The Torch bridge follows the existing CANN `OpBuilder` flow: the first call may
compile it into the shared Torch extension cache. Warm up before graph capture,
as for the other CANN operators; graph replay does not compile the bridge.

```python
from sgl_kernel_npu.operators.moe import grouped_matmul_swiglu_quant
```

`grouped_matmul_swiglu_quant(x, weight, weight_scale, x_scale, group_list, swiglu_limit=0.0, *, weight_assist_matrix=None, impl="auto")`
accepts `auto`/`cann_ops_transformers_cann920` (A2/A3 CANN) or `torch_native` (independent eager reference).

The operator fuses grouped matrix multiplication, dequantization, SwiGLU and
per-token dynamic quantization:

- `x`: INT8 `[R,K]` activations.
- `weight`: INT8 NZ `[E,K,N]` for W8A8 or packed INT32 NZ `[E,K,N/8]` for W4A8.
- `weight_scale`: BF16 `[E,N]` for W8A8. W4A8 uses UINT64 scale bits carried
  in an INT64 tensor of shape `[E,N]` or `[E,K_groups,N]`.
- `weight_assist_matrix`: FP32 `[E,N]` activation-shift compensation required
  for W4A8. Omit this argument for W8A8.
- `x_scale`: FP32 `[R]` activation dequantization scales.
- `group_list`: cumulative INT64 `[E]` expert ends on device.
- `swiglu_limit`: positive values clamp gate above by `swiglu_limit` and up to `[-swiglu_limit,swiglu_limit]`;
  zero disables clipping.

Returns INT8 `[R,N/2]` and FP32 `[R]` dequantization scales. Dynamic quantization
uses per-row maximum absolute value divided by 127, then rounds and saturates to
INT8. Only rows before `group_list[-1]` are valid; padding contents are unspecified.
All tensors must be on the same NPU.

To prepare NZ weights, enable `torch.npu.config.allow_internal_format = True`
before `torch_npu.npu_format_cast(weight, 29)`. For W4A8, cast the packed INT8
`[E,K,N/2]` bytes to NZ before viewing them as INT32, as in SGLang's ModelSlim
weight loading. The native path supports graph capture and changed-input replay.
The eager reference reads expert ends on the host and must stay outside capture.

The numerical tests compare valid dequantized outputs against the eager reference
with relative RMS tolerances of 0.01 for W8A8 and 0.02 for W4A8. They cover empty
and uneven expert groups and changed activations and expert ends during graph
replay. W4A8 tests exercise both per-channel and per-group scales.
