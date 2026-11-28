# fused_conv1d_decode_gating

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Unverified | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`.

```python
from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_decode_gating

def fused_conv1d_decode_gating(
    projected_states_qkvz,
    projected_states_ba,
    conv_weights_t,
    conv_state,
    cache_indices,
    A_log,
    dt_bias,
    num_v_heads,
    *,
    impl="auto",
):
    ...
```

## Description

Fuses single-step decode depthwise causal `conv1d` (kernel size 4 with SiLU activation), in-place rolling `conv_state` shift, and GDN decay gating directly from packed `projected_states_qkvz` and `projected_states_ba`:

For each active sequence $b$ with slot $s = \text{cache\_indices}[b] \ge 0$ and $x_0 = \text{qkvz}[b, :D_{\text{qkv}}]$:

$$y_b = \operatorname{SiLU}\!\left(S_{s,0} w_0 + S_{s,1} w_1 + S_{s,2} w_2 + x_0 w_3\right), \quad (S_{s,0}, S_{s,1}, S_{s,2}) \leftarrow (S_{s,1}, S_{s,2}, x_0)$$

$$g_b = -\exp(A_{\log}) \cdot \operatorname{softplus}(a_b + \text{dt\_bias}), \quad \beta_b = b_{\text{raw}}$$

For padding slots ($s < 0$), $y_b = x_0$ and `conv_state` is untouched. `beta` outputs pre-sigmoid logits matching `recurrent_gated_delta_rule`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `projected_states_qkvz` | `torch.Tensor` | `[B, QKVZ_DIM]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous packed projection tensor; leading `QKV_DIM` columns hold `qkv`. |
| `projected_states_ba` | `torch.Tensor` | `[B, 2*NV]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous packed `[b_raw, a_raw]` tensor. |
| `conv_weights_t` | `torch.Tensor` | `[4, QKV_DIM]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous transposed depthwise conv weights (`QKV_DIM % 512 == 0`). |
| `conv_state` | `torch.Tensor` | `[num_slots, 3, QKV_DIM]` | `torch.bfloat16`, `torch.float16` | Required | Contiguous writable rolling conv state pool updated in place. |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32`, `torch.int64` | Required | Slot indices per batch item (`< 0` denotes padding). |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | Required | Per-head log-A gating parameter. |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`, `torch.float16`, `torch.float32` | Required | Per-head decay bias parameter. |
| `num_v_heads` | `int` | — | — | Required | Number of value heads `NV`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[B, QKV_DIM]` | Same as `projected_states_qkvz` | Conv1d+SiLU output for active slots, passthrough for padding slots. |
| `g` | `torch.Tensor` | `[B, NV]` | `torch.float32` | Log-decay gate values. |
| `beta` | `torch.Tensor` | `[B, NV]` | Same as `projected_states_qkvz` | Unpacked pre-sigmoid beta logits. |

## triton backend

- Uses `BLOCK_D = 2048` when `B >= 2` and `QKV_DIM % 2048 == 0` (else `1024` or `512`) and `tl.int64` slot/batch addressing.
- Safe for NPU Decode Graph capture/replay.

## torch_native backend

- Independent PyTorch reference executing float32 accumulation, SiLU, softplus gating, and per-slot state shift.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_decode_gating

bs, nk, nv, d = 8, 16, 48, 128
qkv_dim = (2 * nk + nv) * d
qkvz = torch.randn(bs, qkv_dim + nv * d, device="npu", dtype=torch.bfloat16)
ba = torch.randn(bs, 2 * nv, device="npu", dtype=torch.bfloat16)
w_t = torch.randn(4, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
state = torch.randn(16, 3, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
cache_idx = torch.arange(bs, device="npu", dtype=torch.int32)
A_log = torch.randn(nv, device="npu", dtype=torch.float32)
dt_bias = torch.randn(nv, device="npu", dtype=torch.bfloat16)

mixed_qkv, g, beta = fused_conv1d_decode_gating(
    qkvz, ba, w_t, state, cache_idx, A_log, dt_bias, nv, impl="auto"
)
```
