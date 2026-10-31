# split_qkv_rmsnorm_rope_pos_cache_half_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
    *,
    impl="auto",
):
    ...
```

`auto` selects the same lazily imported entry as `triton`; there is no fallback.
The implementations declare all known generations, but hardware execution remains unverified.

## Description

Split Q/K/V and optionally RMS-normalize Q/K per head. Read NeoX rotation frequencies from packed [cos_half,sin_half] cache rows selected by positions. Indices are converted to int32, then clamped on-device. Normalized values optionally round through BF16 before RoPE; all final Q/K features, including the unrotated tail, round through BF16 before being converted to input dtype. V retains input dtype and values.

$$
p=\mathrm{clamp}(\mathrm{int32}(\mathrm{positions}),0,M-1),\quad A=\mathrm{optionalRMSNorm}(X),\quad C=\begin{cases}\mathrm{BF16}(A),&\mathrm{cast\_norm\_to\_bf16},\\A,&\text{otherwise},\end{cases}\quad Y_{q,k}=\mathrm{cast}_{\mathrm{input.dtype}}(\mathrm{BF16}(\mathrm{RoPE}_R(C))).
$$


## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input_tensor` | Tensor | [N,Q+2K] | floating | required | Contiguous concatenated Q/K/V. |
| `positions` | Tensor | N values | int32/int64 | required | One index per token, on input device; int32 conversion precedes clamping. |
| `cos_sin_cache` | Tensor | [M,R], M>0 | floating | required | Packed cos_half then sin_half, on input device; R equals rope_dim. |
| `q_hidden_size` | int | — | — | required | Positive Q; divisible by K. |
| `kv_hidden_size` | int | — | — | required | Positive K; divisible by head_dim. |
| `head_dim` | int | — | — | required | Positive per-head dimension D; excluding bool. |
| `eps` | int/float/None | — | — | None | None skips normalization; otherwise a finite positive Python scalar. |
| `q_weight` | Tensor/None | at least D values | floating | None | Required when eps is set, contiguous and on input device. |
| `k_weight` | Tensor/None | at least D values | floating | None | Required when eps is set, contiguous and on input device. |
| `q_bias` | Tensor/None | at least D values | floating | None | Optional, paired with k_bias; applied only during normalization. |
| `k_bias` | Tensor/None | at least D values | floating | None | Optional, paired with q_bias; contiguous and on input device. |
| `rope_dim` | int/None | — | — | None | Defaults to D; must be positive, even and <= D. |
| `cast_norm_to_bf16` | bool | — | — | True | Round normalized Q/K through BF16 before RoPE, even when norm is skipped. |
| `wide_grid_min_tokens` | int | — | — | 1024 | Positive token threshold selecting the wide Triton launch. |
| `impl` | str | — | — | `"auto"` | Keyword-only: auto, triton, torch_native. |

Floating means float16, bfloat16 or float32; read tensors must share the input device. Norm vectors are contiguous and may use a different floating dtype. N may be zero. Use finite participating values for accuracy comparisons. Positive integer options exclude bool.

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | New contiguous output on the input device. |
| `k` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |
| `v` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |

No input tensor, weight or bias is modified. Each backend validates metadata with its own norm/norm_utils.py; the implementations have no shared validation dependency. Triton additionally requires an NPU. Invalid metadata raises AssertionError before a launch; no device values are read back for validation.

## triton backend

Retains the kernel, deterministic pre-RoPE BF16 RNE helper and grid heuristic. Below wide_grid_min_tokens, columns are KV heads; at/above it, Q widths up to 8192 use the wide grid. Wider Q projections retain the per-head grid. head_dim must be a power of two. Positions and cache may be non-contiguous: the wrapper makes them contiguous. There is no host read of index values, so graph capture remains possible. Empty batches avoid a launch.

The original script under `sgl_kernel_npu.norm` is unchanged. The checks documented here belong to the migrated implementation.

## torch_native backend

Uses independent FP32 PyTorch normalization and rotation with explicit BF16 cast boundaries matching the fused kernel, rather than the legacy golden's lower-precision unfused sequence. No additional 1 is added to weights: MiniMax already passes its Gemma-adjusted multiplier. wide_grid_min_tokens is validated but affects only Triton's scheduling. The legacy golden also lacks the kernel's index clamping; this reference includes it.

No Triton kernel is called. The reference is usable on CPU as well as NPU. Reduction order, rsqrt approximation and multiply-add fusion may cause numerical differences; dtype-specific tolerance checks do not prove bitwise equivalence to the framework unfused path.

CPU arithmetic support does not change package initialization: ordinary public
imports still require torch_npu, the native library and ACL hardware discovery.

Supplementary FP16/FP32-storage accuracy cases use BF16 tolerances because Q/K
still pass through BF16 quantization. Separate exact-value oracles check the
pre-RoPE rounding switch, final Q/K rounding, unchanged V and input preservation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope_pos_cache_half_npu

x = torch.randn(4, 256, device='npu', dtype=torch.bfloat16)
positions = torch.arange(4, device='npu', dtype=torch.int64)
cache = torch.cat((torch.ones(64, 16, device='npu'), torch.zeros(64, 16, device='npu')), dim=-1)
wq = torch.ones(64, device='npu')
wk = torch.ones_like(wq)
args = (x, positions, cache, 128, 64, 64)
options = dict(eps=1e-6, q_weight=wq, k_weight=wk, rope_dim=32)
outputs = split_qkv_rmsnorm_rope_pos_cache_half_npu(*args, **options)
reference = split_qkv_rmsnorm_rope_pos_cache_half_npu(*args, impl="torch_native", **options)
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
