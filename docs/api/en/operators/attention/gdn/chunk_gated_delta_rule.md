# chunk_gated_delta_rule

## Hardware support

| Implementation | State dtype | arch 2201 | arch 3510 |
| --- | --- | --- | --- |
| `ascendc` | BF16 | Supported | Supported |
| `ascendc` | FP32 | Unsupported | Unverified |
| `torch_native` | BF16 | Supported | Supported |
| `torch_native` | FP32 | Unverified | Unverified |

State dtype is selected by `initial_state` (BF16 or FP32). Optional `chunk_state`
must use the same dtype, and the returned `final_state` retains that dtype.
`query`, `key`, `value`, `beta` and the returned `out` remain BF16; optional `g`
is FP32 and `actual_seq_lengths` is int32.

The `ascendc` FP32-state path on arch 3510 uses mixed precision. Some internal
matrix operations still use BF16. On arch 2201, `ascendc` accepts BF16 state
only. `torch_native` accepts FP32 state on both architectures. FP32-state device
validation on arch 3510 remains pending.

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule

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

## Test coverage

The 15 original BF16-state accuracy cases passed on arch 2201 and arch 3510.
Another 15 cases cover FP32 state on arch 3510 and await device validation.
These cases are skipped on arch 2201. Both sets use `B=1, T=2048, Nk=8, Nv=16, Dk=Dv=128`, with five scale settings
and three chunk-state layouts: absent, contiguous and noncontiguous.

Accuracy compares `ascendc` with `torch_native` using the shared dtype-specific
standards. Performance uses the shared framework with five warmup and five active
calls. For each state dtype, the default performance case uses explicit
`scale=128**-0.5` and no `chunk_state`. See the
[test source](../../../../../../tests/python/operators/attention/gdn/test_chunk_gated_delta_rule.py).
