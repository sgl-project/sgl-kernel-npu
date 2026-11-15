# D256 QSA prefill main attention (Ascend C)

Native sources behind `sgl_kernel_npu.operators.attention.qsa_prefill`. The
public contract, parameter tables and hardware support matrix are in
[`docs/api/en/operators/attention/qsa_prefill.md`](../../docs/api/en/operators/attention/qsa_prefill.md)
and its Chinese counterpart. This file documents the implementation and its
validation.

`SGL_KERNEL_ENABLE_A3_ONLY_OPS` gates these kernels. Despite the name it is an
arch22-family toggle, on for every SoC outside the Ascend950 series, so an A2
build contains them as well; only an Ascend950 build lacks
`torch.ops.npu.qsa_prefill_prepare` and `torch.ops.npu.qsa_prefill_runs`.
Measurements exist for A3 only, so the API pages mark A2 unverified. Both
entry points are internal helpers of the `ascendc` implementation: pass only
metadata produced by that implementation, not hand-written run descriptors.
Other dtypes, head layouts and compression ratios should use the caller's
existing fallback.

## Sources

| Path | Contents |
| --- | --- |
| `op_host/qsa_prefill.cpp` | Torch registration, argument preparation, immutable tiling cache and kernel launch. |
| `op_host/safe_launch.h` | Keeps tensor arguments alive until the queued launch executes. |
| `op_kernel/prepare.cpp` | Expands selections and builds physical run descriptors per sharing group. |
| `op_kernel/qsa_prefill.cpp` | Kernel entry; dispatches the policies in `policies.hpp`. |
| `op_kernel/qsa_prefill.hpp` | Task geometry, L1 residency and the pipeline across QK, softmax, PV and rescale. |
| `op_kernel/block_qk.hpp` | Selected-block QK with ND-to-NZ contiguous run copies. |
| `op_kernel/block_pv.hpp` | PV accumulation over the same runs. |
| `op_kernel/rescale_o.hpp` | Online-softmax rescaling and FP32 base output with LSE. |

## Implementation notes

1. Compare **all 512 selected blocks**, including interior entries, ordering,
   padding and duplicates. Share base attention only within exact 16/8/4/2/1
   query groups. Packing, native task geometry and the tail merge use the
   actual local Q/KV counts; Q/K/V are not duplicated to emulate 16/2. Each
   original query keeps its independent causal tail.
2. Pack Q with one contiguous 256-element copy per head row. Addressing a
   whole 8192-element tile through a computed `(row * H + head) * 256 + dim`
   index instead compiles to an element-wise gather, which dominates runtime,
   cannot be captured in an NPU graph, and leaves the vector cores in a state
   that deadlocks the following cross-core kernel when launches are queued.
3. Prepare physical run descriptors once per actual sharing group. Verify every
   request-table slot before using a contiguous-copy fast path; fragmented,
   invalid and repeated slots preserve their original semantics.
4. Load contiguous runs into L1 with ND-to-NZ copies. Reuse the group's Q in
   L1; use D256 cube tiles with 32-row L0 subdivisions.
5. Keep base attention output and LSE in FP32 for the tail merge, then produce
   the final BF16 output. This is not bitwise equivalent to other attention
   implementations.
6. Keep PRE_LAUNCH=2 with three GM workspace stages and the existing CrossCore
   pipeline. Pipeline-depth experiments without stable general gains are not
   included.
7. Tiling is immutable and cached by full shape, scale, device and both local
   head counts. First initialization uses a blocking PyTorch device copy so
   queued writes to reused allocator storage cannot overwrite the metadata.

The QSA policies are distinct specializations of shared attention
infrastructure, so the existing sparse-attention kernel's layout and arithmetic
are unchanged.

## Validation

After building and installing this branch in an initialized CANN environment:

```bash
python -m pytest tests/python/operators/attention/test_qsa_prefill.py \
  --skip-acc-check=false --enable-perf=false -v -s -rs
python benchmark/attention/bench_qsa_prefill.py --rows 128 4621 16384 --iterations 20
```

The suite compares the Ascend C result with the independent `torch_native`
reference over 16/8/4/2/1 sharing granularities, interior mismatches, invalid
and duplicate selections, fragmented mappings, partial groups, lengths
0/1/3/67, zero rows, tiling-cache reuse after allocator churn, launches
queued without an intervening synchronization, warmed graph replay with
rewritten inputs, unsupported layouts and sequential reconstruction of
TP2/TP8 shards. These are operator checks; they do not establish model task
accuracy.

### Operator measurements

Current baseline for the shapes the test suite monitors, measured with the
shared profiling utility on Ascend910_9382 + CANN 9.1.0 + torch-npu 2.10 +
triton-ascend 3.2.2, 512 blocks per query, five warmup and five active
iterations, average summed NPU task duration:

| Query rows | Local Q/KV | Selection | Average device task |
| --- | --- | --- | ---: |
| 4621 | 16/2 | Fully shared | 3202.016 us |
| 4621 | 16/2 | All different | 7929.967 us |
| 4621 | 8/1 | Fully shared | 1773.227 us |
| 4621 | 8/1 | All different | 4408.496 us |
| 4621 | 24/2 | Fully shared | 6347.383 us |
| 4621 | 24/2 | All different | 17513.162 us |
| 4621 | 12/1 | Fully shared | 3035.925 us |
| 4621 | 12/1 | All different | 8501.862 us |
| 4621 | 3/1 | Fully shared | 1180.368 us |
| 4621 | 3/1 | All different | 4044.745 us |

The 24/2 and 12/1 layouts exceed the kernel's eight queries per KV head and
are composed from four and two native launches over six-head slices, so their
cost per query head is 20% to 45% above a layout the kernel takes directly.
Narrow shards such as 3/1 amortize the shared base computation over fewer
query heads, which raises their cost per head for the same reason.

### Historical prototype comparison

On Ascend910_9362 + CANN 9.0 + torch-npu 2.10, an alternating-order 20-sample
comparison against a local independent-query D256 Ascend C prototype gives
(the original 16/2 extraction, before the TP extension):

| Query rows | Selection layout | Independent ms | This extraction ms |
| --- | --- | ---: | ---: |
| 16384 | Fully shared | 30.442 | 11.129 |
| 16384 | All different | 30.480 | 32.012 |
| 16384 | Mixed | 30.513 | 25.420 |
| 128 | Fully shared | 0.357 | 0.386 |
| 128 | All different | 0.360 | 0.559 |

The comparator is a retained local D256 prototype, not the upstream MiniMax
operator or a CANN serving baseline. Short and unshared workloads regress, so
this is an explicit API, not a default dispatch change. Raw samples are kept
local; `benchmark/attention/bench_qsa_prefill.py` reproduces the event
timings and the test suite reports the profiled baselines above.

This extraction comes from a local Whittle TP1 W8A8 experiment. Whole-model
results also contain indexer and GDN optimizations outside this change; those
combined throughput figures must not be attributed to this operator alone.
