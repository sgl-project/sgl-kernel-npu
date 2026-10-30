# chunk_gated_delta_rule

## Signature and selection

```python
from sgl_kernel_npu.operators.attention import chunk_gated_delta_rule

out, final_state = chunk_gated_delta_rule(
    query, key, value, beta=beta, initial_state=initial_state,
    actual_seq_lengths=actual_seq_lengths, impl="auto"
)
```

Optional `g` and `chunk_state` default to `None`; `impl` defaults to `"auto"`.
`beta`, `initial_state` and `actual_seq_lengths` must be supplied despite their
optional Python defaults. `auto` selects `ascendc`, which calls the distinct new
`torch.ops.npu.sgl_chunk_gated_delta_rule` binding. Missing native registration or
execution failure propagates; there is no fallback. The old
`torch.ops.npu.chunk_gated_delta_rule` and its Python/Triton route are unchanged.
Select `impl="torch_native"` explicitly for the independent PyTorch reference.

Omitted or `None` scale means **1.0**. An explicit number, including **0.0**, is
preserved.

## Parameters and outputs

All tensor inputs to `ascendc` must be on the same NPU. Inputs are converted to
contiguous working tensors internally; callers need not change their layouts.

| Parameter | Shape and dtype |
| --- | --- |
| `query`, `key` | BF16 `[T, Nk, Dk]`, equal shapes |
| `value` | BF16 `[T, Nv, Dv]` |
| `beta` | Required BF16 `[T, Nv]`, delta update coefficient |
| `initial_state` | Required BF16 or FP32 `[B, Nv, Dv, Dk]`; FP32 native state requires arch35 / Ascend950 |
| `actual_seq_lengths` | Required int32 `[B]`, lengths of packed sequences in order; native usable scope requires every length to be strictly positive and their sum to equal `T` |
| `g` | Optional FP32 `[T, Nv]`, log decay; `None` means no decay |
| `scale` | Optional scalar, multiplies attention output only |
| `chunk_state` | Optional tensor `[totalChunks, Nv, Dv, Dk]`, same dtype as `initial_state`; caller allocates capacity for `sum(ceil(length / 64))` chunks |
| `impl` | `auto`, `ascendc`, `torch_native` |
| `out` | BF16 `[T, Nv, Dv]` |
| `final_state` | `[B, Nv, Dv, Dk]`, same dtype as `initial_state` |

Dimensions are positive; `Nk, Nv <= 64`, `Dk, Dv <= 128`, and `Nv % Nk == 0`.
Value head `h` uses query/key head `h // (Nv // Nk)`. Packed sequences have
independent initial/final states. Native validation preserves the old trailing
chunk dimensions; it does not prove the caller's first-dimension capacity or
length values. Callers must supply valid lengths and sufficient storage. The
independent reference checks nonnegative lengths and their sum explicitly;
this is not evidence that native validation reads those values.

Zero-length sequences are outside the native usable scope. The adapter allocates
`final_state` without initialization, and the inherited arch22/arch35 Kernel loops
skip all state writes for a zero-length sequence, leaving that sequence's
`final_state` unwritten and undefined. Native validation does not reject this
case. The independent `torch_native` reference accepts zero lengths and returns
the corresponding `initial_state` as the mathematical final state; that behavior
does not establish native support for zero-length sequences.

`initial_state` is never mutated. A **contiguous** `chunk_state` receives each
chunk's **entering** state, ordered first by sequence then by chunk. A
**noncontiguous** `chunk_state` is converted to a working copy and **does not
receive copyback**, preserving the old Host behavior. Only `(out, final_state)`
is returned; the temporary chunk buffer is not returned.

## Mathematics and independent reference

For each value head, let `S` have orientation `Dv × Dk`. At token `t`,

```text
D_t = exp(g_t) S_(t-1)             (g absent: D_t = S_(t-1))
u_t = beta_t (v_t - D_t k_t)
S_t = D_t + u_t k_t^T
out_t = scale S_t q_t
```

The reference evaluates the same recurrence in chunks of 64 using FP32 working
algebra. For one chunk, `G = cumsum(g)` and the causal decay matrix is
`E_ij = exp(G_i - G_j)` for `j <= i`. It masks discarded upper-triangle
exponents **before** exponentiation, without clipping valid causal values.
It solves the unit lower-triangular system
`A U = beta * (V - exp(G) K S^T)`, where
`A = I + tril(beta_i * (K K^T)_ij * E_ij, -1)`, by forward substitution using
basic PyTorch operations. Then it forms causal output and the exiting state.
The reference does not call the old/new custom operator, ACLNN or another backend.
FP32 algebraic equivalence does not establish agreement with Kernel rounding,
intermediate BF16 storage, or either NPU generation.

## Support and validation status

| Implementation | Declaration / implementation | Validation boundary |
| --- | --- | --- |
| `auto`, `ascendc` | arch22 (`2201`), arch35 (`3510`); arch22 BF16 state, arch35 BF16/FP32 state | Real generated ACLNN header, optional Ref support, SDK compilation, workspace compatibility and NPU precision remain pending |
| `torch_native` | Independent PyTorch computation with the shapes/dtypes above | Reference implementation; NPU comparison remains pending |

The normal package/public registry requires the installed NPU runtime.
No CPU fallback is added to public dispatch.

Precision acceptance compares old/new directly and each against the reference,
including caller-visible `chunk_state`. Tests use the repository's public
`assert_accuracy` without overrides: matching BF16 outputs use atol/rtol `0.05`,
matching FP32 outputs use `1e-5`. Output dtype is preserved when choosing that
standard. The ARC-013 fixture retains `B=1, T=2048, Nk=8, Nv=16, Dk=Dv=128`,
BF16 state and explicit `128**-0.5`; default/zero scales and optional
chunk-buffer layouts are separately labelled contract variants. No NPU precision or
performance result is claimed by the source implementation alone.
