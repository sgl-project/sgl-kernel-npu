# fused_conv1d_gdn_verify

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unsupported | Supported |
| `torch_native` | Unverified | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, registered to the same entry point as `triton` on `Generation.ARCH35`.

```python
from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_gdn_verify


def fused_conv1d_gdn_verify(
    qkvz,
    ba,
    conv_weights_t,
    conv_states,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    batch_size,
    k_ver,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    query_start_loc=None,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
): ...
```

## Description

Fuses speculative verification depthwise causal Conv1D (`kernel_size=4`, SiLU, in-place `conv_states` window shift), `Q/K` row-wise L2 normalization, GDN gating, and Cube WY-chunk verification on Ascend 950 (`arch35`). Reads strided `qkvz[:, :QKV_DIM]` and `ba` directly, avoiding `Slice`, standalone `causal_conv1d`, and intermediate `mixed_qkv` materialization.

For valid slots (`cache_indices[b] >= 0` and `query_start_loc[b+1] > query_start_loc[b]` when `query_start_loc` is supplied), reads `[old[W-3], old[W-2], old[W-1]]` and writes `[old[W-2], old[W-1], x_0, ..., x_{K-1}]` in place. Padded ACLGraph rows skip `conv_states` mutation so slot 0 is never corrupted.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `qkvz` | `torch.Tensor` | `[B * k_ver, qkvz_dim]` | `torch.bfloat16` | Required | Contiguous packed projection output with `qkvz_dim >= (2*NK + NV)*128`. |
| `ba` | `torch.Tensor` | `[B * k_ver, 2 * NV]` | `torch.bfloat16` | Required | Contiguous packed `[b \| a]` gating projections. |
| `conv_weights_t` | `torch.Tensor` | `[4, QKV_DIM]` | `torch.bfloat16` | Required | Contiguous transposed depthwise Conv1D weights. |
| `conv_states` | `torch.Tensor` | `[num_slots, k_ver + 2, QKV_DIM]` | `torch.bfloat16` | Required | Contiguous convolution history pool updated in place. |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | Required | Contiguous per-head log-decay parameter. |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous per-head decay bias parameter. |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16` | Required | Contiguous read-only SSM state pool. |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | Required | Active state slot index per request. |
| `batch_size` | `int` | — | — | Required | Number of requests `B`. |
| `k_ver` | `int` | — | — | Required | Verification width (`2 <= k_ver <= 16`). |
| `num_k_heads` | `int` | — | — | Required | Number of Q/K heads (must divide `num_v_heads`). |
| `num_v_heads` | `int` | — | — | Required | Number of V heads. |
| `head_dim` | `int` | — | — | `128` | Per-head dimension (`128` only). |
| `eps` | `float` | — | — | `1e-6` | L2Norm epsilon. |
| `query_start_loc` | `torch.Tensor` or `None` | `[B + 1]` | `torch.int32` | `None` | Optional cumulative offsets; rows with `qsl[b+1] <= qsl[b]` skip `conv_states` writes. |
| `workspace` | `FusedGDNVerifyWorkspace` or `None` | — | — | `None` | Optional persistent static scratch workspace. |
| `linear_layer_idx` | `int` | — | — | `0` | Layer slot index inside `workspace`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"` (`ARCH35` only), or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B, k_ver, NV, 128]` | `torch.bfloat16` | Speculative verify attention output. |
| `rec_k` | `torch.Tensor` | `[B, k_ver, NK, 128]` | `torch.bfloat16` | L2-normalized key commit record. |
| `rec_v` | `torch.Tensor` | `[B, k_ver, NV, 128]` | `torch.bfloat16` | WY innovation vector commit record. |
| `rec_g` | `torch.Tensor` | `[B, k_ver, NV]` | `torch.float32` | Cumulative log-decay commit record. |

## triton backend

- Registered for `Generation.ARCH35` (Ascend 950). Runs one 56-AIV fused Conv1D + L2Norm + Gating kernel followed by the same Cube WY verify pipeline as `fused_gdn_verify`.

## torch_native backend

- Independent FP32 PyTorch causal Conv1D + SiLU + step-by-step recurrent GDN reference updating `conv_states` in place.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_gdn_verify

B, K, NK, NV, D = 4, 4, 16, 48, 128
qkv_dim = (2 * NK + NV) * D
qkvz = torch.randn(B * K, qkv_dim + NV * D, device="npu", dtype=torch.bfloat16)
ba = torch.randn(B * K, 2 * NV, device="npu", dtype=torch.bfloat16)
conv_w_t = torch.randn(4, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
conv_states = torch.zeros(8, K + 2, qkv_dim, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(NV, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(NV, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(8, NV, D, D, device="npu", dtype=torch.bfloat16)
cache_indices = torch.arange(B, device="npu", dtype=torch.int32)

out, rec_k, rec_v, rec_g = fused_conv1d_gdn_verify(
    qkvz, ba, conv_w_t, conv_states, A_log, dt_bias, ssm_states,
    cache_indices, B, K, NK, NV, impl="auto",
)
```
