# Native LIM MTP3 CopySFA resident-refresh overlap

## Behavior

A steady full-visible request previously selected the complete-copy path whenever
its request copy count was nonzero. This handled rewritten resident KV correctly,
but serialized every union copy before attention and disabled source-aware gather.

The native operator now accepts `resident_refresh_tail=False`. Existing callers
keep their previous semantics and can still supply resident entries in any order.
The SGLang native-LIM Q4 verify manager explicitly opts in with
`full_visible=True, resident_refresh_tail=True`; no environment switch is added.

The opt-in contract requires a native LIM request union: ordinary entries must be
represented in per-query miss prefixes, and rewritten resident hits must form an
appended suffix of at most one item per query. The kernel copies the final
`min(query_count, request_copy_count)` entries before a global barrier, then uses
source-aware attention for the remaining misses. Redundant copies of a few true
misses are safe. A batch containing a cold fill retains complete-copy handling.

The flag reuses the reserved field at byte164; the direct-launch tiling record
remains168 bytes. It requires full-visible mode. Graph padding retains positive
query-row segments and uses zero KV/cache/copy counts, as in the serving runtime.

## Validation

Verified on one assigned A3 die reported as `Ascend910_9362`, with CANN9.1,
Torch2.10, Python3.12 and BF16 KV. The native library was rebuilt from the full
CMake target and loaded from a private package rather than a globally replaced
wheel. Only the assigned die was used; no model weights were loaded.

- Four CPU adapter tests cover legacy keyword compatibility, explicit native
  opt-in and invalid flag combinations.
- Sixty distinct native-operator checks cover existing dynamic-TND, small/large
  heads, FP16/BF16, native plans, eager/Graph, cold fill, exact-K attention,
  resident counts0..4, zero/odd true misses, positive-width padding, resident
  payload changes after capture, and unordered legacy resident-first plans.
- Fourteen real-manager lifecycle cases cover shared ownership, rollback/reuse,
  H4/H8, Graph/eager, metadata fusion on/off, asynchronous assertions, and
  capacities11392/12288/16384/16768. These use the real framework manager and
  synthetic tensors; they are not a full-model accuracy or serving benchmark.

The initial48-case native run included10 suffix cases. The final22-case suffix
suite supersedes those10 and adds zero-miss and unordered-legacy coverage;
38+22=60 distinct native cases. Raw JUnit files are kept separately.

## Same-address performance

Both variants call the same production `npu` operator with the new flag false
or true. They share exact Host/HBM addresses, query and RoPE tensors, metadata
and allocation placement. CPU affinity is fixed, and HBM is restored outside
the measured events. Each entry averages three alternating Graph150 rounds.
B16/Q4 (MTP3), H4, BF16 KV, TopK2048 and a65536-row shared host allocation are
fixed. Random source IDs are sampled per request; misses are partitioned across
four queries, plus four rewritten resident entries per request.

| C | True miss union | Previous mode us | Native overlap us | Reduction |
|---:|---:|---:|---:|---:|
| 12288 | 400 | 517.646 | 360.429 | 30.37% |
| 12288 | 700 | 666.865 | 434.992 | 34.77% |
| 16384 | 400 | 531.501 | 366.095 | 31.12% |
| 16384 | 700 | 633.839 | 455.227 | 28.18% |

Per-query misses are100 or175 respectively, and copy-list lengths include the
four resident refreshes. These quantities are independent of serving prefix-hit
rate. All-request output comparisons and zero-tolerance cache-payload checks
passed. These are Q4 operator times, not TPOT; do not divide them by three or
substitute them for new end-to-end model measurements.

## Deployment

Use matching framework and kernel branches:
`fix/mtp3-copysfa-overlap-20261031` in both personal repositories. Rebuild and
install the kernel wheel before starting the updated framework, using the
repository's existing A3 `bash build.sh -a kernels` workflow. Restart any process
that already loaded the old library. Its schema must include
`bool resident_refresh_tail=False`; the native Q4 manager passes true itself.
Updating only the wheel leaves existing callers on their compatibility path.

A private Python3.12/AArch64 wheel was also built for the tested CANN9.1/Torch2.10
runtime, with a local `+mtp3fix.20261031` version suffix. Other runtime/ABI
combinations should rebuild from source. No global site-packages were modified
by validation, and no new full-model throughput, acceptance or accuracy result
is claimed here.
