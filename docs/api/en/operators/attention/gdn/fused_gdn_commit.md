# fused_gdn_commit

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
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_commit


def fused_gdn_commit(
    ssm_states_all,
    rec_k_all,
    rec_v_all,
    rec_g_all,
    dst_slots,
    last_steps,
    *,
    track_slots=None,
    track_steps=None,
    impl="auto",
): ...
```

## Description

Commits accepted speculative GDN states across all linear-attention layers in place via a single persistent 28-AIC Cube launch on Ascend 950 (`arch35`):
$$S_{\text{last}} = \exp(G_{\text{last}}) \cdot S_0 + \sum_{t=0}^{\text{last}} \exp(G_{\text{last}} - G_t)\, v_t k_t^\top$$
When `track_slots` and `track_steps` are provided, also snapshots the intermediate Radix Cache page-boundary state $S_{\text{track}}$ from entering state $S_0$ into `ssm_states_all[:, track_slots[b]]` before overwriting `dst_slots[b]`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `ssm_states_all` | `torch.Tensor` | `[L, num_slots, NV, 128, 128]` | `torch.bfloat16` | Required | All-layer SSM state pool updated in place. |
| `rec_k_all` | `torch.Tensor` | `[L, max_bs, K_VER, NK, 128]` | `torch.bfloat16` | Required | Normalized key records produced during verify. |
| `rec_v_all` | `torch.Tensor` | `[L, max_bs, K_VER, NV, 128]` | `torch.bfloat16` | Required | WY innovation vector records produced during verify. |
| `rec_g_all` | `torch.Tensor` | `[L, max_bs, K_VER, NV]` | `torch.float32` | Required | Cumulative log-decay records produced during verify. |
| `dst_slots` | `torch.Tensor` | `[B]` | Integer | Required | Destination slot indices (`< 0` skips). |
| `last_steps` | `torch.Tensor` | `[B]` | Integer | Required | Accepted step indices in `[0, K_VER - 1]` (`< 0` skips). |
| `track_slots` | `torch.Tensor` or `None` | `[B]` | Integer | `None` | Optional Radix checkpoint slot indices (`< 0` skips). |
| `track_steps` | `torch.Tensor` or `None` | `[B]` | Integer | `None` | Optional Radix checkpoint step indices (`< 0` skips). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"` (`ARCH35` only), or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `result` | `torch.Tensor` | `[L, num_slots, NV, 128, 128]` | `torch.bfloat16` | Original `ssm_states_all` alias updated in place. |

## triton backend

- Registered for `Generation.ARCH35` (Ascend 950). Launches `min(L * B * NV, 28)` persistent Cube programs with `tl.int64` offsets, supporting arbitrary integer `GQA = NV // NK >= 1` and `2 <= K_VER <= 16`.

## torch_native backend

- Independent FP32 PyTorch reference updating `ssm_states_all` and optional checkpoint slots in place.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_commit

L, S, B, K, NK, NV, D = 4, 16, 4, 4, 16, 48, 128
ssm_all = torch.randn(L, S, NV, D, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_k = torch.randn(L, B, K, NK, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_v = torch.randn(L, B, K, NV, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_g = -torch.cumsum(torch.rand(L, B, K, NV, device="npu", dtype=torch.float32) * 0.1, dim=2)
dst_slots = torch.arange(B, device="npu", dtype=torch.int64)
last_steps = torch.tensor([0, 1, 2, 3], device="npu", dtype=torch.int64)

fused_gdn_commit(ssm_all, rec_k, rec_v, rec_g, dst_slots, last_steps, impl="auto")
```
