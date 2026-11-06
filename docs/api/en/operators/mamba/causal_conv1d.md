# causal_conv1d

## Signature

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d

causal_conv1d(
    x, weight, conv_states,
    bias=None, query_start_loc=None, cache_indices=None,
    has_initial_state=None, num_accepted_tokens=None,
    activation_mode=0, pad_slot_id=-1, run_mode=0,
    *, impl="auto",
) -> Tensor
```

`auto` selects `ascendc`, with no fallback. `torch_native` is an independent
PyTorch implementation. The new Ascend C binding is
`torch.ops.npu.sgl_causal_conv1d`; the legacy binding
`torch.ops.npu.causal_conv1d` and legacy source files remain unchanged.

## Hardware and validation status

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` / `auto` | Declared, BF16 / FP16 | Declared, BF16 / FP16 |
| `torch_native` | Registered | Registered |

These are source and registry declarations, not a claim of device validation.
Real CANN code generation, compilation, packaging and NPU execution must be
validated separately. The PyTorch backend can compute with CPU tensors, but
importing the public package/registry still requires its NPU environment.

## Parameters and result

`B` is the sequence count, `L` a common sequence length, `T` the packed token
count, `D` the number of independent channels, `K` the convolution width, and
`S` the state capacity per cache slot. Layout is **token-major**; the older
Python Mamba wrapper with `[B, D, L]` input and `[D, K]` weight has a different
interface.

| Parameter | Shape | Contract |
| --- | --- | --- |
| `x` | `[T, D]` or `[B, L, D]` | Contiguous BF16 or FP16. |
| `weight` | `[K, D]` | Contiguous, same dtype as `x`; `K` is 2, 3 or 4. |
| `conv_states` | `[slots, S, D]` | Contiguous, same dtype as `x`; updated **in place**. |
| `bias` | `[D]` or `None` | Optional per-channel bias; valid inputs use the input dtype. |
| `query_start_loc` | `[B + 1]` or `None` | Packed sequence boundaries. Required for every 2D `x`, including update. Ignored for 3D `x`. Converted to int64. |
| `cache_indices` | `[B]` or `None` | Sequence-to-slot mapping; converted to int64. Omitted means slot `b`. |
| `has_initial_state` | `[B]` or `None` | Prefill reads history where the converted int64 value is nonzero; omitted means no history. Ignored by update. |
| `num_accepted_tokens` | `[B]` or `None` | Converted to int32. Controls speculative update history only when `run_mode == 1` and `K == 4`. |
| `activation_mode` | integer, default `0` | Any nonzero value applies SiLU. |
| `pad_slot_id` | integer, default `-1` | Skip a sequence whose explicit cache index equals this value. |
| `run_mode` | integer, default `0` | Exactly `1` selects integrated update; every other value selects prefill. |
| `impl` | keyword-only string | `auto`, `ascendc` or `torch_native`. |
| Result `y` | Same shape/dtype as `x` | Newly allocated output. State is mutated, not returned as another tensor. |

Empty optional tensors are treated as absent. An empty/absent
`query_start_loc` does not satisfy the 2D input requirement. Tensors passed to
`ascendc` must belong to the same NPU. Metadata contains valid boundaries and
indices; live cache slots must have at least `K-1` history rows. The old Host
does not fully validate metadata values or state capacity; this refactor does
not redefine malformed inputs. Concurrent sequences writing the same cache
slot have no established ordering and are outside the independent reference
accuracy contract.

The old kernel reads optional buffers from their physical `data_ptr` order,
ignoring strides after the metadata dtype conversions. The new adapter and
reference preserve that behavior; a noncontiguous optional tensor is not
normalized to its logical element order. Callers should supply dense buffers
with sufficient storage. Arbitrary strided views are not a validated contract.

A padded sequence leaves its state unchanged and its `y` rows **uninitialized**.
Those rows must not be read as zero or included in numerical comparison. The
default sequence-to-slot mapping does not consult `pad_slot_id`.

## Calculation and state

For a sequence with tokens `X[0:L, :]`, concatenate `K-1` history rows `H` and
`X` to obtain `Z`. The oldest history row corresponds to weight tap zero:

```text
u[t, d] = sum(Z[t + j, d] * weight[j, d], j=0..K-1) + bias[d]
y[t, d] = cast_to_x_dtype(u[t, d])                  if activation_mode == 0
          cast_to_x_dtype(u[t, d] / (1 + exp(-u[t, d]))) otherwise
```

Bias is omitted when absent. The independent PyTorch calculation converts raw
history, tokens and weights to FP32, broadcasts each weight row across tokens,
adds taps in increasing order, then applies bias, SiLU and the final cast.
FP32 arithmetic expressed this way is a mathematical reference; different
hardware multiply-add/SiLU rounding need not be bitwise identical.

| Mode | History | State writeback |
| --- | --- | --- |
| Prefill (`run_mode != 1`) | `state[slot, :K-1]` when the sequence's initial-state flag is nonzero; otherwise zeros. | Write the last `K-1` raw rows of `Z` to the first `K-1` state rows. |
| Ordinary update (`run_mode == 1`) | Always read `state[slot, :K-1]`, regardless of the initial-state flag. | Same ordinary writeback. |
| Update, nonempty accepted tensor, `K == 4` | `o = min(max(accepted[b]-1, 0), S-3)`; read old `state[slot, o:o+3]`. | If `S >= 2+L`, write `[old_state[o+1], old_state[o+2], X]` into the first `2+L` rows. Otherwise use ordinary `K-1` writeback. |

The offset formula assumes valid capacity `S >= 3`. Accepted counts do not
truncate the number of tokens calculated. `K == 2/3` uses ordinary update even
with an accepted tensor. Unwritten rows and unused cache slots remain
unchanged. State contains raw inputs/history, never activated convolution
outputs. The reference snapshots old state before writing to avoid overwriting
history needed by the calculation.

Both update branches belong to the integrated `causal_conv1d` implementation.
They do not call or depend on the separate `csrc/causal_conv1d_update` operator,
which is outside this migration.

## Accuracy coverage

The formal tests retain the original nine prefill configurations, CPU random
input generation order and seed `20260326`. Each backend receives fresh state
storage. Valid `y` rows use shared `DEFAULT_STANDARDS` through `assert_accuracy`:
BF16 `atol=rtol=5e-2`, FP16 `atol=rtol=1e-3`. The **entire** state pool uses exact
comparison, including untouched rows/slots. Read-only inputs are also checked.

Nine ordinary update cases reuse those exact inputs with only `run_mode=1`.
Two speculative update cases use the confirmed configurations from the old
update test: FP16 `(B,L,D,K,S)=(1,4,1024,4,6)` with bias/SiLU, and BF16
`(32,4,12288,4,6)` with SiLU and no bias. All accepted counts are 4 and cache
slots are `0..B-1`. The FP16 state has zero leading rows and three random history
rows at the end; the BF16 state is random throughout. These two cases generate
CPU inputs with seed 42. All eleven cases call this operator with `run_mode=1`,
not the standalone update operator or the old Triton wrapper.

The test file additionally provides mutually exclusive `--capture-legacy PATH`
and `--compare-legacy PATH` command-line modes. Capture runs in the old package
environment and saves the original inputs and old outputs/state; comparison
runs in the new package environment, reuses those exact inputs, and checks
old/new, new/reference and old/reference separately. This does not require
loading both packages or registering both extensions in one process. A missing
or incompatible replay file is an error. Passing the default new/reference
pytest cases alone is not completion of the old/new regression requirement.

Add `--include-update` when capturing to save all 20 prefill/update cases in the
new replay format. Comparison infers the case set from the file; original
nine-case prefill replay files remain supported.

## Performance

All 20 configurations are selected by the separate `perf_shape` list. Accuracy
is checked first, then only `impl` is profiled by the shared helper with five
warmup and five active calls. Each call receives an independent copy of the
original state. All state copies are prepared in the unprofiled initial call,
so they are excluded from measured tasks and calls do not advance one another's
history. Inputs, shapes and dtypes are not reduced for profiling.

From the repository root, using the Python environment containing the new wheel:

```bash
SGL_TEST_ROOT="$(pwd -P)"
python3 -m pytest "$SGL_TEST_ROOT/tests/python/operators/mamba/test_causal_conv1d.py" \
  --impl ascendc --impl-ref torch_native --skip-acc-check=false \
  --enable-perf=true --csv-path "$SGL_TEST_ROOT/test-results/causal_conv1d.csv" \
  --prof-dir "$SGL_TEST_ROOT/test-results/profilings" -sv -ra
```

`--enable-perf=false` runs accuracy only. `--skip-acc-check=true` profiles without
calling the reference and records `accuracy_checked=False`. CSV times are mean
NPU device-task times, not end-to-end host latency. New update device accuracy,
old/new device replay, actual performance measurements and same-device
performance-regression comparison remain to be performed; adding these test
paths does not establish those hardware results.
