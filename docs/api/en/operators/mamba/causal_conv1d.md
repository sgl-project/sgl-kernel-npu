# causal_conv1d

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` uses the same entry point as `ascendc`.

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d

def causal_conv1d(
    x, weight, conv_states,
    bias=None, query_start_loc=None, cache_indices=None,
    has_initial_state=None, num_accepted_tokens=None,
    activation_mode=0, pad_slot_id=-1, run_mode=0,
    *, impl="auto",
):
    ...
```

## Description

Computes per-channel causal 1D convolution with optional bias and SiLU,
for prefill and integrated update. Let `Z` concatenate `K-1` history rows
and the current input in time order:

```text
u[t,d] = sum(Z[t+j,d] * weight[j,d], j=0..K-1) + bias[d]
y[t,d] = u[t,d]                              if activation_mode == 0
         u[t,d] / (1 + exp(-u[t,d]))          otherwise
```

Omit bias when absent. Output is cast to `x.dtype`. `conv_states` is updated
in place with raw inputs/history, not convolution or activation results.

## Parameters and return value

`B` is the sequence count, `L` a common sequence length, `T` the packed token
count, `D` the channel count, `K` the convolution width, and `S` the state
capacity per slot. Input layout is token-major.

| Parameter | Shape / type | Description |
| --- | --- | --- |
| `x` | `[T,D]` or `[B,L,D]` | Contiguous BF16 or FP16. |
| `weight` | `[K,D]` | Contiguous, same dtype as `x`; `K` is 2, 3 or 4. |
| `conv_states` | `[slots,S,D]` | Contiguous, same dtype as `x`; updated in place. |
| `bias` | `[D]` or `None` | Optional channel bias with `x.dtype`; default `None`. |
| `query_start_loc` | `[B+1]` or `None` | Packed sequence boundaries, converted to int64. Required for 2D input; ignored for 3D input. |
| `cache_indices` | `[B]` or `None` | Sequence-to-slot mapping, converted to int64; defaults to slot `b` for sequence `b`. |
| `has_initial_state` | `[B]` or `None` | Nonzero after int64 conversion makes prefill read history; defaults to no history. Ignored by update. |
| `num_accepted_tokens` | `[B]` or `None` | Converted to int32; selects speculative history only for update with `K == 4`. |
| `activation_mode` | `int`, default `0` | Any nonzero value enables SiLU. |
| `pad_slot_id` | `int`, default `-1` | Skips sequences whose explicit cache index equals this value. |
| `run_mode` | `int`, default `0` | `1` selects integrated update; other values select prefill. |
| `impl` | `str`, default `"auto"` | Keyword-only; `auto`, `ascendc` or `torch_native`. |

| Return value | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `y` | Same as `x` | Same as `x` | Newly allocated output; state is written through `conv_states`. |

Empty optional tensors are treated as absent; 2D inputs still require valid
sequence boundaries. Indices, boundaries and capacity must be valid, with at
least `K-1` history rows per live slot. Sequences should use separate cache slots.
Padded sequences leave state unchanged and output rows uninitialized. The
default slot mapping does not check `pad_slot_id`.

## State updates

- Prefill: read `state[slot,:K-1]` when the initial-state flag is nonzero,
  otherwise use zero history. Write the last `K-1` concatenated rows to the
  first `K-1` state rows.
- Ordinary update: always read existing history; use the same writeback rule.
- Update with a nonempty accepted tensor and `K == 4`: set
  `o = min(max(accepted[b]-1,0),S-3)` and read old `state[slot,o:o+3]`.
  If `S >= 2+L`, write `[old_state[o+1],old_state[o+2],X]` to the first `2+L`
  rows; otherwise use ordinary writeback. Accepted counts do not truncate the
  calculation; `K == 2/3` uses ordinary update.

Unwritten state rows and unused slots remain unchanged. Update is implemented
inside this operator.

## ascendc backend

- Requires all tensors on the same NPU and contiguous `x`, `weight` and `conv_states`.
- Reads optional bias and metadata as dense physical spans after dtype conversion,
  without logical-stride reordering; callers should provide sufficient contiguous storage.

## torch_native backend

- Uses independent PyTorch operations with FP32 accumulation before bias, SiLU
  and output casting.
- Preserves history selection, state updates and optional-argument semantics.
