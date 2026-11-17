# chunk_kda_fwd

## Hardware support

| Implementation | ARCH22 | ARCH35 |
| --- | --- | --- |
| ascendc | Supported | Supported |
| torch_native | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

def chunk_kda_fwd(
    q, k, v, g, beta, *,
    a_log=None, dt_bias=None, initial_state=None,
    cu_seqlens=None, chunk_indices=None,
    layout="BSND", scale=1.0, chunk_size=64,
    safe_gate=False, lower_bound=-5.0, use_gate_in_kernel=False,
    state_v_first=False, output_final_state=True,
    output_gk=False, output_w=False, output_u=False,
    output_qg=False, output_kg=False, output_v_new=False, output_h=False,
    impl="auto",
):
    ...
```

## Description

Compute the fused Kimi Delta Attention (KDA) prefill pipeline with per-channel retention gates. The operator forms chunk-local cumulative gates and causal query/key coefficients, solves the delta-rule transformation, advances the recurrent state across chunks, and reconstructs attention outputs. It supports grouped query/key heads, packed variable-length sequences, two chunk widths, optional raw-gate activation, and optional intermediate outputs. Inputs are not modified; an 11-element tuple preserves fixed output positions with None for unrequested intermediates.

## Parameters

| Name | Type | Shape / value | Description |
| --- | --- | --- | --- |
| q / k | BF16 | BSND: [B,T,H,K]; BNSD: [B,H,T,K]; TND: [T,H,K]; NTD: [H,T,K] | Matching shapes. H > 0 and HV is a multiple of H. |
| v | BF16 | Same layout, with HV heads and width V | Attention values. |
| g | FP32 / BF16 | Same layout as v, with width K | Per-channel log decay, or raw gate when use_gate_in_kernel=True. |
| beta | FP32 / BF16 | v without its final dimension | Delta coefficient; already activated, no sigmoid is applied. |
| a_log | FP32 / None | [HV] | Required for in-kernel gate activation. |
| dt_bias | FP32 / None | [HV*K] | Optional raw gate bias. |
| initial_state | FP32 / None | [N,HV,K,V], or [N,HV,V,K] when state_v_first=True | One initial state per sequence; None starts from zero. |
| cu_seqlens | int64 / None | [N+1] | Packed sequence boundaries, starting at 0 and ending at T; nondecreasing, N <= 1024. Production packed inputs use B=1. |
| chunk_indices | int64 / None | [total_chunks,2] or its flat view | Canonical (sequence_id, local_chunk_id) pairs. Requires cu_seqlens; None derives canonical metadata. |
| layout | str | BSND / BNSD / TND / NTD | Default BSND. Rank-3 layouts represent one packed token axis. |
| scale | float | scalar | Default 1.0; normally K**-0.5. |
| chunk_size | int | 64 or 128 | Default 64. K and V are multiples of 16 in [16,256]; H and HV are at most 128; T is positive. |
| safe_gate | bool | scalar | Default False. With in-kernel activation, select lower_bound * sigmoid(exp(a_log) * (g + dt_bias)); otherwise use -exp(a_log) * softplus(g + dt_bias). |
| lower_bound | float | scalar | Default -5.0. Must be in [-5,0) when safe_gate and use_gate_in_kernel are both True. |
| use_gate_in_kernel | bool | scalar | Default False. Activate raw g internally; beta is unchanged. |
| state_v_first | bool | scalar | Default False. Select [V,K] instead of [K,V] for initial/final/chunk state presentations. |
| output_final_state | bool | scalar | Default True. Return final state in tuple slot 1. |
| output_gk / output_w / output_u / output_qg / output_kg / output_v_new / output_h | bool | scalar | Each defaults to False; omitted outputs occupy None slots, without changing tuple length. |
| impl | str | auto / ascendc / torch_native | Keyword-only; default auto selects ascendc. |

## Returns

| Slot | Name | Type | Shape | Description |
| --- | --- | --- | --- | --- |
| 0 | attn_out | BF16 | [B,T,HV,V], or [T,HV,V] for rank-3 inputs | Always returned, in sequence-major order even for BNSD/NTD input. |
| 1 | final_state | FP32 | [N,HV,K,V] or [N,HV,V,K] | None when output_final_state=False. |
| 2 | gk | FP32 | [B,HV,T,K] or [HV,T,K] | Chunk-local cumulative log2 decay; None unless output_gk=True. |
| 3 / 4 | aqk_out / akk_out | BF16 | [B,HV,T,C] or [HV,T,C] | Causal scaled query/key scores and solved key/key triangular coefficients. Always returned. |
| 5 / 7 / 8 | w / qg / kg | BF16 | [B,HV,T,K] or [HV,T,K] | Each controlled by its output flag. |
| 6 / 9 | u / v_new | BF16 | [B,HV,T,V] or [HV,T,V] | Each controlled by its output flag. |
| 10 | h | BF16 | [B,NC,HV,K,V] or [NC,HV,K,V] | Entering chunk states, with final K/V exchanged when state_v_first=True; NC is total_chunks for packed input. None unless output_h=True. |

## ascendc

Supports ARCH22 and ARCH35. Q/K/V use BF16; g and beta independently accept FP32 or BF16. Optional state tensors use the layouts above.

## torch_native

A device-preserving PyTorch implementation accepting the same arguments and output flags. It uses FP32 calculations with the BF16 intermediate/output boundaries and returns tensors on the input device.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

q = torch.nn.functional.normalize(torch.randn(1, 64, 1, 128, device="npu"), dim=-1).bfloat16()
k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
v = torch.randn_like(q)
g = torch.full(q.shape, -0.01, device=q.device, dtype=torch.float32)
beta = torch.full(q.shape[:-1], 0.5, device=q.device, dtype=torch.float32)
outputs = chunk_kda_fwd(q, k, v, g, beta, scale=128**-0.5, impl="ascendc")
out, final_state = outputs[:2]
print(out.shape, final_state.shape, len(outputs))
# torch.Size([1, 64, 1, 128]) torch.Size([1, 1, 128, 128]) 11
```
