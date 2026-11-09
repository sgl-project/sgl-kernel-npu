# chunk_gated_delta_rule

## Hardware support

| Implementation | State dtype | arch22 (2201) | arch35 (3510) |
| --- | --- | --- | --- |
| `ascendc` | BF16 | Supported | Supported |
| `ascendc` | FP32 | Unsupported | Unverified |
| `torch_native` | BF16 | Supported | Supported |
| `torch_native` | FP32 | Unverified | Unverified |

State dtype applies to `initial_state`, `final_state` and optional `chunk_state`.
The A5 `ascendc` kernel has an FP32 state path. The `torch_native` implementation
accepts FP32 state on both architectures. The accuracy tests compare `ascendc`
against `torch_native` in 15 BF16-state cases on each of A3 and A5. All cases use
`B=1, T=2048, Nk=8, Nv=16, Dk=Dv=128`, with five scale settings crossed with three
chunk-state layouts. See the
[test source](../../../../../tests/python/operators/attention/test_chunk_gated_delta_rule.py).

## Signature

```python
from sgl_kernel_npu.operators.attention import chunk_gated_delta_rule

def chunk_gated_delta_rule(
    query,
    key,
    value,
    *,
    beta=None,
    initial_state=None,
    actual_seq_lengths=None,
    scale=None,
    g=None,
    chunk_state=None,
    impl="auto",
):
    ...
```

`auto` selects `ascendc`; `torch_native` uses PyTorch operations.
An unavailable native implementation raises an error without switching backends.
`beta`, `initial_state` and `actual_seq_lengths` are required despite their
`None` defaults in the signature.

## Description

Computes chunked Gated Delta Rule attention, returning per-token output and the
final state of each sequence. Sequences are packed according to
`actual_seq_lengths` and update their states independently.

For each value head, the state matrix `S` has shape `[Dv, Dk]`. At token `t`:

```text
D_t = exp(g_t) * S_(t-1)      # No decay when g=None
u_t = beta_t * (v_t - D_t @ k_t)
S_t = D_t + outer(u_t, k_t)
out_t = scale * (S_t @ q_t)
```

Here, `outer` denotes an outer product. `scale=None` means `1.0`, and an
explicit `0.0` is valid. `torch_native` computes internally in FP32.

## Parameters and return values

`B` is the sequence count, `T` the total token count, `Nk` and `Nv` the query/key
and value head counts, and `Dk` and `Dv` their feature dimensions.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `query`, `key` | `[T, Nk, Dk]` | BF16 | Must have equal shapes. |
| `value` | `[T, Nv, Dv]` | BF16 | Value tensor. |
| `beta` | `[T, Nv]` | BF16, required | Delta update coefficient. |
| `initial_state` | `[B, Nv, Dv, Dk]` | BF16 / FP32, required | Initial state. See [hardware support](#hardware-support) for FP32 status. |
| `actual_seq_lengths` | `[B]` | int32, required | Sequence lengths, not cumulative boundaries; `ascendc` requires positive lengths summing to `T`. |
| `scale` | — | float / `None` | Output multiplier; `None` means `1.0`. |
| `g` | `[T, Nv]` | FP32 / `None` | Log decay; `None` means no decay. |
| `chunk_state` | `[totalChunks, Nv, Dv, Dk]` | Initial-state dtype / `None` | Optional chunk-state buffer; `totalChunks = sum(ceil(length / 64))`. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `out` | `[T, Nv, Dv]` | BF16 |
| `final_state` | `[B, Nv, Dv, Dk]` | Same as `initial_state` |

Usage notes:

- `ascendc` requires all tensors on the same NPU. Inputs may be noncontiguous
  and are converted to contiguous working tensors internally. `torch_native`
  requires all tensors on the same device.
- Dimensions must be positive; `Nk, Nv <= 64`, `Dk, Dv <= 128`, and `Nv % Nk == 0`.
  Value head `h` uses query/key head `h // (Nv // Nk)`.
- The caller must provide valid lengths and sufficient chunk-buffer capacity;
  native code does not check these. `torch_native` also accepts zero-length
  sequences and returns their initial states as final states.
- `initial_state` is unchanged. A contiguous `chunk_state` is updated in place
  with each chunk's entering state, ordered by sequence then chunk. A
  noncontiguous buffer uses a temporary copy without writing back to the input.
  Only `(out, final_state)` is returned.
