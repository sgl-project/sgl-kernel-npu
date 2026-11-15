# recurrent_gated_delta_rule

## Hardware support

| Implementation | arch 2201 | arch 3510 |
| --- | --- | --- |
| `ascendc` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` selects the same entry point as `ascendc`.

```python
from sgl_kernel_npu.operators.attention.gdn import recurrent_gated_delta_rule

def recurrent_gated_delta_rule(
    mix_qkv, recurrent_state, beta, scale, actual_seq_lengths,
    ssm_state_indices, nk, nv, intermediate_state=None,
    cache_indices=None, num_accepted_tokens=None, g=None, gk=None,
    *, impl="auto",
):
    ...
```

The legacy name `torch.ops.npu.recurrent_gated_delta_rule` is retained.
The public API uses this name to call the installed package's implementation
for `auto` / `ascendc`, and independent equations for `torch_native`.

## Description

For value head `h`, use Q/K head `h // (Nv // Nk)`:

```text
q = scale * q_raw / sqrt(sum(q_raw*q_raw) + 1e-6)
k = k_raw / sqrt(sum(k_raw*k_raw) + 1e-6)
D_t = exp(g_t) * H_(t-1)                 # g=None: D_t = H_(t-1)
delta_t = sigmoid(beta_t) * (v_t - D_t @ k_t)
H_t = D_t + outer(delta_t, k_t)
out_t = H_t @ q_t
```

The state matrix has layout `[Dv,Dk]`. FP32 state is kept across tokens; BF16
snapshots are written after each token without reloading the rounded snapshots.

1. With no intermediate pool, read and write `recurrent_state`.
2. With an intermediate pool, read and write that working pool. The original
   recurrent pool supplies seeds and is unchanged for nonoverlapping pools.
3. The initial slot is `indices[b,accepted[b]-1]` when accepted counts exist,
   otherwise `indices[b,0]`. Accepted counts must lie in `[1,S]`.
4. In intermediate mode, if the selected slot is divisible by `S`, initialize
   from `recurrent_state[cache_indices[b]]`; otherwise read the selected working
   slot. Write token `t` to `indices[b,t]` in the working pool.

## Parameters and return value

| Parameter | Shape / dtype | Meaning |
| --- | --- | --- |
| `mix_qkv` | `[B,S,2*Nk*Dk+Nv*Dv]`, BF16 | Packed Q, K, V in this order. |
| `recurrent_state` | `[N,Nv,Dv,Dk]`, BF16 | Mutable state pool, or recurrent seed pool when an intermediate pool is supplied. |
| `beta` | `[B,S,Nv]`, BF16 | Pre-sigmoid logits. |
| `scale` | Required float | Q multiplier; `0.0` is preserved. Narrowed to FP32 as in the old launch. |
| `actual_seq_lengths` | `[B]`, int32 | Per-sequence lengths, not cumulative boundaries. Each sequence starts at its fixed `b*S` offset. |
| `ssm_state_indices` | `[B,S]`, int32 | Initial-state selection and per-token output slots. |
| `nk`, `nv` | Positive integers | Q/K and value head counts; `Nv % Nk == 0`. |
| `intermediate_state` | Complete `[Nv,Dv,Dk]` slots, BF16 / `None` | Default `None`; working pool, commonly `[N*S,Nv,Dv,Dk]`. |
| `cache_indices` | At least `B` elements / `None` | Default `None`; recurrent seed slot per sequence; required with an intermediate pool. Cast to contiguous int32. |
| `num_accepted_tokens` | At least `B` elements / `None` | Default `None` (column 0); selects initial-state column `accepted[b]-1`; cast to contiguous int32. Does not shorten the sequence. |
| `g` | At least `B*S*Nv` elements / `None` | Default `None`; log decay, cast to contiguous FP32, then exponentiated. Missing means decay 1. |
| `gk` | Optional tensor | Default `None`; converted to contiguous FP32 for compatibility; the legacy computation ignores it. |
| `impl` | `str`, default `"auto"` | Keyword-only; `auto`, `ascendc` or `torch_native`. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B,S,Nv,Dv]` | BF16 | Newly allocated output; only tokens below each valid length are defined. State pools are updated in place. |

Dimensions are positive, `1 <= S <= 8`, and valid lengths
must lie in `[1,S]`. All tensors must be on the same NPU for `ascendc` or on the
same device for `torch_native`. State slots must be valid for their respective
pools. The caller must avoid conflicting state reads/writes between sequences;
native metadata validation does not read tensor values back to the host.

## ascendc backend

Required QKV, beta, lengths, indices and recurrent state retain the old
`data_ptr()` interpretation: read dense physical spans starting at their storage
offsets, irrespective of logical Tensor strides. They are not made contiguous.
The underlying storage must cover those spans. Optional indices/gates are
converted and made contiguous. An already contiguous intermediate pool is
updated in place; a noncontiguous one uses a temporary with **no copyback**,
matching the old Host. Use contiguous pools for visible state updates.

Native gate copies require `S*Nv*2 <= 65535` bytes for beta and, when `g`
is supplied, `S*Nv*4 <= 65535` bytes for g. On arch 2201, the aligned FP32 row
stride must also fit 255 blocks of 32 bytes. Invalid sizes are rejected before launch.

## torch_native backend

Uses independent PyTorch equations, with FP32 recurrence and BF16 output/state
snapshots. It follows the same state-selection and storage-span semantics; the
native copy-field limits do not apply to this reference.

## Example

This uses the existing `(B,S,Nk,Nv,Dk,Dv)=(2,1,8,16,128,128)` configuration.
Backend calls use separate initial state pools.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import recurrent_gated_delta_rule

b, s, nk, nv, dk, dv = 2, 1, 8, 16, 128, 128
device, dtype = "npu", torch.bfloat16
q = torch.randn(b, s, nk, dk, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn(b, s, nv, dv, device=device, dtype=dtype)
state = 0.1 * torch.randn(b * s, nv, dv, dk, device=device, dtype=dtype)
state_ref = state.clone()
inputs = dict(
    mix_qkv=torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1),
    beta=torch.randn(b, s, nv, device=device, dtype=dtype),
    scale=dk**-0.5,
    actual_seq_lengths=torch.full((b,), s, device=device, dtype=torch.int32),
    ssm_state_indices=torch.arange(b * s, device=device, dtype=torch.int32).view(b, s),
    nk=nk,
    nv=nv,
    g=-(0.05 + 0.95 * torch.rand(b, s, nv, device=device)),
)
out = recurrent_gated_delta_rule(**inputs, recurrent_state=state)  # auto: ascendc
out_ref = recurrent_gated_delta_rule(
    **inputs, recurrent_state=state_ref, impl="torch_native"
)
print(out.shape, out_ref.shape)  # torch.Size([2, 1, 16, 128])
```
