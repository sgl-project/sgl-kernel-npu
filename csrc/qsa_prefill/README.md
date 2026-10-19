# D256 QSA prefill main attention

`sgl_kernel_npu.attention.qsa_prefill.qsa_prefill` adds an explicit A3 BF16
main-attention API for ratio-4 compressed QSA with D256. Supported local Q/KV
head counts are 16/2, 8/1, 6/1 and 2/1. The caller
supplies the indexer's 512 selected logical blocks for each query and CPU
sequence metadata. This operator does not compute indexer scores or Top-K.
Existing MiniMax sparse attention and model dispatch remain unchanged.
Hardware validation is on Ascend910_9362 (910C), CANN 9.0, torch-npu 2.10.
This explicit API is intended for long prefill; short or entirely unshared
selections may be slower than an independent-query implementation.

```python
from sgl_kernel_npu.attention.qsa_prefill import qsa_prefill

out = qsa_prefill(q, k, v, blocks, request_table, request_row,
                 length=sequence_length, base=sequence_length-q.shape[0])
```

- Q: contiguous BF16 `[R,H,256]`; K/V: contiguous BF16 `[capacity,KV,256]`.
  Supported `(H,KV)` are `(16,2)`, `(8,1)`, `(6,1)` and `(2,1)`.
  Whittle (16 Q / 2 KV globally) uses 8/1 at TP2 and 2/1 at TP8;
  Qwen3.8-Flash-Next (24 Q / 2 KV globally) uses 6/1 at TP4.
  The caller supplies the appropriate replicated KV head when TP exceeds the
  global KV-head count; this API performs no distributed communication.
- Blocks: contiguous int32 `[R,512]`. Each nonnegative block expands into
  four logical token positions, bounded by `length`, then mapped through the
  request table. Negative/out-of-range block positions and invalid physical
  slots are omitted; selection order and duplicates are preserved.
- Request table: int32 `[requests,max_length]` with unit column stride;
  `request_row`: contiguous int64 `[1]` NPU tensor. The caller must supply a
  valid table row. All inputs use the same NPU device.
- `length` and `base` are CPU integers; the wrapper performs no device-to-host
  metadata reads. Each query separately includes its 0--3 causal tail tokens.
- Output: BF16 `[R,H,256]`. Rows without selected or tail tokens are zero.
- Only arch22 builds with the A3-family target enabled expose the native helpers. Other dtypes, head shapes,
  and compression ratios should use the caller's existing fallback.

## Implementation and optimizations

The main QK/online-softmax/PV/rescale pipeline and physical run preparation are
Ascend C. Query packing, complete selection equality checks, and per-query tail
merge use Triton NPU; the complete API therefore requires both runtimes.

1. Compare **all 512 selected blocks**, including interior entries, ordering,
   padding and duplicates. Share base attention only within exact 16/8/4/2/1
   query groups. Packing, native task geometry and tail merge use the actual
   local Q/KV counts; Q/K/V are not duplicated to emulate 16/2. Each original
   query keeps its independent causal tail.
2. Prepare physical run descriptors once per actual sharing group. Verify every
   request-table slot before using a contiguous-copy fast path; fragmented,
   invalid and repeated slots preserve their original semantics.
3. Load contiguous runs into L1 with ND-to-NZ copies. Reuse the group's Q in L1;
   use D256 cube tiles with 32-row L0 subdivisions.
4. Keep base attention output and LSE in FP32 for the tail merge, then produce
   final BF16 output. This is not bitwise equivalent to every other attention
   implementation; correctness tests compare with FP32 attention over the same
   BF16 inputs, not generation-quality metrics.
5. Keep PRE_LAUNCH=2 with three GM workspace stages and the existing CrossCore
   pipeline. Pipeline-depth experiments without stable general gains are not
   included. Own tensor arguments until the queued native launch executes.

The QSA policies are distinct specializations of shared attention infrastructure,
so the existing sparse-attention kernel's layout and arithmetic are not changed.
`torch.ops.npu.qsa_prefill_prepare` and `qsa_prefill_runs` are internal helpers:
pass only metadata produced by this wrapper, not hand-written run descriptors.
Tiling is immutable and cached by full shape/scale/device values, including both
local head counts. First initialization uses a blocking PyTorch device copy so
queued writes to reused allocator storage cannot overwrite the metadata. Empty
sequences compute only sharing flags when details are requested, without
launching a discarded packed-Q write. Warm a shape
before graph capture; the repository tests include warmed graph replay.

## Validation

After building/installing this branch in an initialized CANN environment:

```bash
python -m unittest discover -s tests/python/sgl_kernel_npu -p test_qsa_prefill.py -v
python benchmark/attention/bench_qsa_prefill.py --rows 128 4621 16384 --iterations 20
```

The tests cover CPU FP32 reference attention, 16/8/4/2/1 sharing, interior
mismatches, invalid and duplicate selections, fragmented mappings, partial
sharing groups, lengths 0/1/3/67, empty inputs, temporary tensor lifetimes,
queued launches, warmed graph replay with changing Q/K/V and block selections,
interleaved head geometries in the tiling cache, unsupported layouts, and
sequential reconstruction of all TP2/TP8/TP4 shards. Relative L2 must
be below 0.01; this operator gate does not establish model task accuracy.

This extraction is based on a local Whittle TP1 W8A8 experiment. Whole-model
results also contain indexer and GDN optimizations outside this change; those
combined throughput figures must not be attributed to this operator alone.

### Extracted operator measurements

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
operator or a CANN serving baseline. This extraction is bitwise identical to
the previously validated shared-query prototype in these cases. Short and
unshared workloads regress, so this is an explicit API, not a default dispatch
change. Raw samples and geometry are recorded in
[the benchmark record](../../docs/benchmarks/qsa_prefill_910c.json).

### TP head-layout validation

The TP extension is validated separately on Ascend910_9382 (910C), CANN 9.1,
with the Ascend910_9362 compiler target,
torch/torch-npu 2.10. Build the current PR sources before running the suite;
reusing an older native library does not validate the new head geometry.
The shard reconstruction test runs each rank sequentially on one NPU and
compares concatenated outputs with global-head CPU FP32 attention. It checks
KV replication and head ordering, but does not establish full distributed
TP8/TP4 deployment or model GSM8K accuracy. These are operator tests; the
separate v18 model-serving profiles use a different runtime integration and
must not be presented as serving validation of this PR API.

The updated suite passes 10 tests with 231 relative-L2 comparisons (maximum
0.00240594). An independent cold-cache run passes all 114 TP shape cases on
the second visible device. See [the TP validation record](../../docs/benchmarks/qsa_prefill_tp_910c.json).
