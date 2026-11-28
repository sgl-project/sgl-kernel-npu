# fused_gdn_verify

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
from sgl_kernel_npu.operators.attention.gdn import (
    create_gdn_verify_workspace,
    fused_gdn_verify,
)


def fused_gdn_verify(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
): ...
```

## Description

Executes Cube-accelerated Gated Delta Net (GDN) speculative target verification on Ascend 950 (`arch35`) from pre-convolved `(mixed_qkv, a, b)` tensors without mutating `ssm_states` or materializing `K_VER` full `[NV, 128, 128]` intermediate states:

1. **AIV Prep**: Computes row-wise `Q`/`K` L2 normalization (`eps = 1e-6`), cumulative log-decay $G_t = \sum_{j \le t} g_j$ where $g_t = -\exp(A_{\log})\cdot\operatorname{softplus}(a_t + \text{dt\_bias})$, $\beta_t = \sigma(b_t)$, and $V \cdot \beta$ across 56 AIV cores.
2. **Cube WY-Chunk Verify**:
   - For `B <= 8` or non-power-of-two `K_VER`, runs a single-stage `(B * NV,)` Cube kernel using exact FP16 Euler inversion $(I + L)^{-1}$ to form WY factors $(W, U)$, innovation vectors $V_{\text{new}} = U - W S_0^\top$, and output $O = (Q S_0^\top \exp(G) + \text{ATTN}\cdot V_{\text{new}}) D^{-1/2}$.
   - For `B > 8` and `K_VER in (2, 4, 8, 16)`, runs a two-stage R-packed pipeline packing `R_PACK = BT // K_VER` requests into one `[BT, 128]` Cube tile with GQA reuse of $K K^\top / Q K^\top$, followed by an unmasked Stage 2 Cube pass.
3. Returns `(out, rec_k, rec_v, rec_g)` for downstream single-launch commit via `fused_gdn_commit`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[B, K_VER, (2*NK + NV)*128]` | `torch.bfloat16` | Required | Contiguous pre-convolved packed QKV tensor on NPU. |
| `a` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous pre-softplus decay logits. |
| `b` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous pre-sigmoid beta logits. |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | Required | Contiguous per-head log-decay parameter. |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`, `torch.float32` | Required | Contiguous per-head decay bias parameter. |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16` | Required | Contiguous read-only SSM state pool. |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | Required | Active state slot per request (`< 0` uses zero initial state). |
| `num_k_heads` | `int` | — | — | Required | Number of Q/K heads (must divide `num_v_heads`). |
| `num_v_heads` | `int` | — | — | Required | Number of V heads. |
| `head_dim` | `int` | — | — | `128` | Per-head dimension (`128` only). |
| `eps` | `float` | — | — | `1e-6` | L2Norm epsilon. |
| `workspace` | `FusedGDNVerifyWorkspace` or `None` | — | — | `None` | Optional persistent static scratch from `create_gdn_verify_workspace()` for ACLGraph capture. |
| `linear_layer_idx` | `int` | — | — | `0` | Layer index inside `workspace`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"` (`ARCH35` only), or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B, K_VER, NV, 128]` | `torch.bfloat16` | Speculative verify attention output. |
| `rec_k` | `torch.Tensor` | `[B, K_VER, NK, 128]` | `torch.bfloat16` | L2-normalized key commit record. |
| `rec_v` | `torch.Tensor` | `[B, K_VER, NV, 128]` | `torch.bfloat16` | WY innovation vector $V_{\text{new}}$ commit record. |
| `rec_g` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.float32` | Cumulative log-decay $G_t$ commit record. |

## triton backend

- Registered for `Generation.ARCH35` (Ascend 950); supports `2 <= K_VER <= 16`, arbitrary integer `GQA = NV // NK >= 1`, and `head_dim = 128`.
- Reuses all scratch tensors statically when a pre-allocated `workspace` is supplied, keeping ACLGraph capture allocation-free.

## torch_native backend

- Independent step-by-step FP32 recurrent GDN reference (without WY chunk decomposition), producing exact Delta-rule innovation records `rec_v` and outputs `out`.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_verify

B, K, NK, NV, D = 4, 4, 16, 48, 128
mixed_qkv = torch.randn(B, K, (2 * NK + NV) * D, device="npu", dtype=torch.bfloat16)
a = torch.randn(B, K, NV, device="npu", dtype=torch.bfloat16)
b = torch.randn(B, K, NV, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(NV, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(NV, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(8, NV, D, D, device="npu", dtype=torch.bfloat16)
cache_indices = torch.arange(B, device="npu", dtype=torch.int32)

out, rec_k, rec_v, rec_g = fused_gdn_verify(
    mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, NK, NV, impl="auto"
)
```
