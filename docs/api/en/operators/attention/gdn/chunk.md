# chunk_gated_delta_rule_npu

The sequence-first chunked GDN pipeline formerly exposed by `fla.chunk` is
available through the public operator framework:

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

out, final_state, h = chunk_gated_delta_rule_npu(
    q, k, v, g, beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    head_first=False,
    use_qk_l2norm_in_kernel=False,
    impl="auto",
)
```

`auto` and `triton` select the same lazy implementation. Its chunk pipeline
directly composes the migrated cumsum, KKT, triangular inverse, WY, state-update
and output implementations. It does not dispatch each leaf through its public
operator. No `torch_native` backend is registered for this outer interface.
Triton-Ascend and an Ascend NPU are required.

## Contract

- `q/k`: `[B,T,Hg,K]`; `v`: `[B,T,H,V]`, with `H % Hg == 0`.
- `g/beta`: `[B,T,H]`; `g` contains per-token log decay, before chunk cumsum.
- `q/k/v` share FP16 or BF16 dtype; the current pipeline uses chunks of 64.
  The state/output kernels use 128-wide key tiles and up to two 64-wide value
  tiles; the regression cases use the model dimensions `K=V=128`.
- `initial_state`: optional `[N,H,K,V]`, where N is the number of sequences.
- `cu_seqlens`: optional contiguous INT32/INT64 boundaries `[N+1]` on the input
  device. Packed input requires `B=1`, boundaries from 0 to T and positive lengths.
  Do not mutate cached boundary tensors in place between calls.
- `scale=None` uses `K**-0.5`; explicit zero remains zero. Optional q/k L2
  normalization uses squared-norm epsilon `1e-6` and retains the input dtype.
- The existing input guard makes tensor inputs contiguous and enters their NPU
  device context. `head_first=True` raises `DeprecationWarning` as before.
- Returns `(out, final_state, h)`: out has v's shape/dtype; final_state is FP32
  `[N,H,K,V]` when requested, otherwise None; h holds entering chunk states in
  key dtype, `[B,NT,H,K,V]`. For packed input NT sums chunks over sequences.
  The default `output_final_state=True` is preserved.

The inherited `GDN_USE_MEGA_GDN=1` path uses the existing mega native operator
and chunks of **128**. Its h therefore has a different chunk count. It requires
that optional native operator to be installed and retains its existing FP16
intermediate conversions. `GDN_RECOMPUTE_SUPPRESS_LEVEL` keeps its original
import-time behavior; h remains part of the public return at either setting.

## Existing limitations and interface distinction

Numerical regression targets fixed-length **B=1** and packed **B=1** multi-sequence
inputs. The inherited state/output kernels have incorrect fixed-length gate
offsets for B>1; the mega implementation also assumes B=1. This migration does
not repair those pre-existing kernels or claim numerical support for B>1.

This interface is separate from `gdn.chunk_gated_delta_rule`, the fused CANN
operator with query/key/value, actual_seq_lengths, chunk_state and two returns.
Its state orientation is different. Migrating this wrapper does not change
SGLang's standard Ascend fused prefill selection.

## Validation

```bash
python -m pytest tests/python/operators/attention/gdn/test_chunk.py \
  --enable-perf=false -v -rs
```

The test uses an independent tokenwise recurrent oracle for output, final state
and entering chunk states. It covers FP16/BF16, packed sequence restarts, chunk
tails, initial state, noncontiguous inputs, normalization and explicit zero scale.
Mega cases skip only when the optional native operator is unavailable.
