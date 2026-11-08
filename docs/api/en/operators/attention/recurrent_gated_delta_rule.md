# recurrent_gated_delta_rule

## Hardware and implementations

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` / `auto` | Declared; NPU validation pending | Declared; NPU validation pending |
| `torch_native` | Independent PyTorch reference | Independent PyTorch reference |

The native implementation preserves the legacy BF16 kernel, including its
arch35 internal branch. CPU reference checks do not establish NPU support.
Unavailable native implementations raise an error without switching backends.

## Signature

```python
from sgl_kernel_npu.operators.attention import recurrent_gated_delta_rule

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

## Inputs and result

| Parameter | Shape / dtype | Meaning |
| --- | --- | --- |
| `mix_qkv` | `[B,S,2*Nk*Dk+Nv*Dv]`, BF16 | Packed Q, K, V in this order. |
| `recurrent_state` | `[N,Nv,Dv,Dk]`, BF16 | Mutable state pool, or recurrent seed pool when an intermediate pool is supplied. |
| `beta` | `[B,S,Nv]`, BF16 | Pre-sigmoid logits. |
| `scale` | Required float | Q multiplier; `0.0` is preserved. Narrowed to FP32 as in the old launch. |
| `actual_seq_lengths` | `[B]`, int32 | Per-sequence lengths, not cumulative boundaries. Each sequence starts at its fixed `b*S` offset. |
| `ssm_state_indices` | `[B,S]`, int32 | Initial-state selection and per-token output slots. |
| `nk`, `nv` | Positive integers | Q/K and value head counts; `Nv % Nk == 0`. |
| `intermediate_state` | Complete `[Nv,Dv,Dk]` slots, BF16 / `None` | Optional working pool, commonly `[N*S,Nv,Dv,Dk]`. |
| `cache_indices` | At least `B` elements / `None` | Recurrent seed slot per sequence; required with an intermediate pool. Cast to contiguous int32. |
| `num_accepted_tokens` | At least `B` elements / `None` | Selects initial-state column `accepted[b]-1`; cast to contiguous int32. Does not shorten the sequence. |
| `g` | At least `B*S*Nv` elements / `None` | Log decay; cast to contiguous FP32, then exponentiated. Missing means decay 1. |
| `gk` | Optional tensor | Converted to contiguous FP32 for compatibility; the legacy computation ignores it. |
| `impl` | `auto`, `ascendc`, `torch_native` | Keyword-only implementation selector. |

Returns only BF16 `out` of shape `[B,S,Nv,Dv]`. State changes are caller-visible
side effects, not additional return values. Outputs after each sequence's valid
length are unspecified. Dimensions are positive, `1 <= S <= 8`, and valid lengths
must lie in `[1,S]`. All tensors must be on the same NPU for `ascendc` or on the
same device for `torch_native`. State slots must be valid for their respective
pools. The caller must avoid conflicting state reads/writes between sequences;
native metadata validation does not read tensor values back to the host.

## Equations and state protocol

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

Required QKV, beta, lengths, indices and recurrent state retain the old
`data_ptr()` interpretation: read dense physical spans starting at their storage
offsets, irrespective of logical Tensor strides. They are not made contiguous.
The underlying storage must cover those spans. Optional indices/gates are
converted and made contiguous. An already contiguous intermediate pool is
updated in place; a noncontiguous one uses a temporary with **no copyback**,
matching the old Host. Use contiguous pools for visible state updates.

## Validation

The public pytest suite preserves these eight source input families, all BF16:

| Family | `(B,S,Nk,Nv,Dk,Dv)` | Accepted counts | Default profiling |
| --- | --- | --- | --- |
| compatible | `(2,1,8,16,128,128)` | Original random counts | No |
| compatible | `(64,2,8,16,128,128)` | Original random counts | No |
| compatible | `(32,8,8,16,128,128)` | Original random counts | No |
| compatible | `(32,8,16,32,128,128)` | Original random counts | No |
| benchmark | `(2,1,8,16,128,128)` | `None`, seed 42 | Yes |
| benchmark | `(4,4,8,16,128,128)` | `None`, seed 42 | Yes |
| benchmark | `(8,8,8,16,128,128)` | `None`, seed 42 | Yes |
| benchmark | `(2,4,8,16,64,32)` | `None`, seed 42 | Yes |

Checks cover valid output tokens, every state slot, unchanged seed/unused slots,
and read-only inputs. Shared accuracy standards apply, plus the stricter source
thresholds: compatible `atol=rtol=2**-8`; benchmark `atol=0.01, rtol=0`.
The non-square case remains unchanged; establish its old/new result on each
target architecture before claiming support.

In the repository root, set an absolute root and your actual architecture:

```bash
export SGL_RGDR_ROOT="$(pwd -P)"
export SGL_RGDR_ARCH=YOUR_ARCH  # Replace with 2201 or 3510.
python "$SGL_RGDR_ROOT/setup.py" --arch-version "$SGL_RGDR_ARCH" \
  --ops recurrent_gated_delta_rule --artifact wheel --without-third-party
python -m pip install --force-reinstall --no-deps "$SGL_RGDR_ROOT"/output/sgl_kernel_npu-*.whl
python -m pytest "$SGL_RGDR_ROOT/tests/python/operators/attention/test_recurrent_gated_delta_rule.py" \
  -q --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false
```

After loading the server's normal CANN/PyTorch environment, use the same Python
for building, installing and testing. Setup prints the exact `ARTIFACT` path.
To also profile the four original benchmark cases, omit `--enable-perf=false`:

```bash
python -m pytest "$SGL_RGDR_ROOT/tests/python/operators/attention/test_recurrent_gated_delta_rule.py" \
  -q --impl=ascendc --impl-ref=torch_native --skip-acc-check=false \
  --prof-dir="$SGL_RGDR_ROOT/profilings/rgdr-new" --csv-path="$SGL_RGDR_ROOT/rgdr-new.csv"
```

Profiling uses the shared device-task measurement, with one unprofiled call,
5 warmup calls and 5 active calls, each using a separate initial working pool.
State copies are prepared before profiling. Explicit `--enable-perf=true` enables
all eight families and substantially increases memory use; the largest working
pool is 520 MiB and profiling prepares 11 working copies.

## Identical-input old/new replay

Use separate Python environments for the old wheel and the new wheel. Set
`SGL_RGDR_OLD_PYTHON` / `SGL_RGDR_NEW_PYTHON` to their absolute Python paths, and
set `SGL_RGDR_REPLAY` to a new absolute output directory. The test module is run
from the current checkout in both environments; it loads the wheel installed in
that environment. The compatible source cases have no fixed seed, so replay
stores inputs rather than inventing a seed.

```bash
cd "$SGL_RGDR_ROOT"
"$SGL_RGDR_OLD_PYTHON" -m tests.python.operators.attention.test_recurrent_gated_delta_rule \
  --capture-legacy "$SGL_RGDR_REPLAY" --benchmark
"$SGL_RGDR_NEW_PYTHON" -m tests.python.operators.attention.test_recurrent_gated_delta_rule \
  --compare-legacy "$SGL_RGDR_REPLAY" --benchmark --prof-dir "$SGL_RGDR_ROOT/profilings/rgdr-replay-new"
```

Replay requires bitwise equality between old and new outputs/state pools and
compares both with `torch_native`. The two `performance.csv` files measure the
same four cases with the same shared profiler. Record the device and matching
CANN, driver, compiler, PyTorch and torch_npu versions before comparing latency.
Do not compare these values directly with the old Triton script's wall time.
Replay/profile directories must not already exist. Without `--benchmark`, omit
`--prof-dir` too. These commands do not supply an NPU validation result themselves.
