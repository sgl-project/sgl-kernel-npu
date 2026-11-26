# D256 QSA prefill main attention (Ascend C)

Native CANN/ACLNN sources behind `sgl_kernel_npu.operators.attention.qsa_prefill`.
The public contract, parameter tables and hardware support matrix are in
[`docs/api/en/operators/attention/qsa_prefill.md`](../../../docs/api/en/operators/attention/qsa_prefill.md)
and its Chinese counterpart. This file documents the native implementation and
its build/validation workflow.

Both native operators (`qsa_prefill_prepare` / `SglQsaPrefillPrepare` and
`qsa_prefill_runs` / `SglQsaPrefillRuns`) are declared in `CMakeLists.txt` with
`EXECUTION_UNIT AICORE` and `SUPPORTED_ARCHS 2201`, covering the arch22 compute
units (`ascend910b` and `ascend910_93`). Measurements exist for A3 only, so the
API pages mark A2 unverified and arch35 unsupported. Both `torch.ops.npu` entry
points are internal helpers of the `ascendc` implementation: pass only metadata
produced by that implementation, not hand-written run descriptors.

## Sources

| Path | Contents |
| --- | --- |
| `CMakeLists.txt` | CANN AI Core operator declarations for `qsa_prefill_prepare` and `qsa_prefill_runs`. |
| `op_host/qsa_prefill_prepare_{def,infer,tiling}.cpp` | OpDef, shape/dtype inference and host tiling for `SglQsaPrefillPrepare`. |
| `op_host/qsa_prefill_runs_{def,infer,tiling}.cpp` | OpDef, shape/dtype inference and host tiling for `SglQsaPrefillRuns`. |
| `torch_adapter_prepare.cpp` | `torch.ops.npu.qsa_prefill_prepare` ACLNN adapter and padded-stride descriptor handling. |
| `torch_adapter_runs.cpp` | `torch.ops.npu.qsa_prefill_runs` ACLNN adapter and workspace lifecycle management. |
| `op_kernel/qsa_prefill_tiling.h` | Shared host/kernel tiling structs for both operators. |
| `op_kernel/qsa_prefill_prepare.cpp` | Expands selections and builds physical run descriptors per sharing group. |
| `op_kernel/qsa_prefill_runs.cpp` | Kernel entry; dispatches the policies in `policies.hpp`. |
| `op_kernel/qsa_prefill.hpp` | Task geometry, L1 residency and the pipeline across QK, softmax, PV and rescale. |
| `op_kernel/block_qk.hpp` | Selected-block QK with ND-to-NZ contiguous run copies and tile-bound descriptor guards. |
| `op_kernel/block_pv.hpp` | PV accumulation over the same runs with matching descriptor guards and NZ padding. |
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
   L1; use D256 cube tiles with 32-row L0 subdivisions. Both QK and PV guard
   against malformed zero/out-of-tile runs or out-of-range slots by filling
   remaining tile entries from valid slot 0 so MMAD never reads uninitialized
   L1 tiles.
5. Keep base attention output and LSE in FP32 for the tail merge, then produce
   the final BF16 output. This is not bitwise equivalent to other attention
   implementations.
6. Keep `PRE_LAUNCH=2` with three GM workspace stages and the existing CrossCore
   pipeline.
7. Both operators launch through CANN's generated ACLNN interfaces via
   `OpCommand::RunOpApi`, with `record_stream` and captured tensor/descriptor
   lifetimes protecting asynchronous launches and NPU graph replay.

The QSA policies are distinct specializations of shared attention
infrastructure in `csrc/attention/sparse_attention_score/op_kernel`, so the
existing sparse-attention kernel's layout and arithmetic are unchanged.

## Build and validation

Build a wheel from the repository root in an initialized CANN environment:

```bash
python3 setup.py --arch-version 2201 --ops qsa_prefill_prepare,qsa_prefill_runs --artifact wheel --without-third-party
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl

python3 -m pytest tests/python/operators/attention/test_qsa_prefill.py \
  --skip-acc-check=false --enable-perf=false -v -rs
python3 -m pytest tests/python/operators/attention/test_qsa_prefill.py \
  -m "" --skip-acc-check=false --enable-perf=false -v -rs
python3 benchmark/attention/bench_qsa_prefill.py --rows 128 4621 16384 --iterations 20
```
