# A5 native vs pre/post-migration LIM/CopySFA microbenchmark

## Locate the first tensor difference in the current framework

For the reported short-request issue, use this dedicated 16-token case:

```bash
cd /mnt/share/y00970600/personal/sgl-kernel-npu
python3 benchmark/copysfa_a5_ab/trace_short_alignment.py \
  --framework-prefix /mnt/share/y00970600/personal/sglang \
  --device 0 \
  --output /mnt/share/y00970600/a5-alignment-short-q4
```

It defaults to batch 1, 128 heads, Q4, C8192 and context 16. For ordinary Q1
decode, add `--query-width 1` and use another output directory. The wrapper
rejects long contexts, and both cases verify the actual short vendor-SFA path.
Metadata, Host publication, ring staging and gathered KV are checked alongside
attention outputs. For contexts <=2048, native and offload outputs are also
compared separately against an independent CPU causal attention oracle using
dequantized packed C8 KV. This helps distinguish an offload-only mismatch from
a shared operator discrepancy. This synthetic single-layer case does not
reproduce model scheduling or prove that a real-model curl has been fixed.

`trace_alignment.py` runs one synthetic A5 packed-C8 layer in eager mode,
first cold and then with the same inputs and warm residency. It calls the
installed native quantized LI/SFA operators for the no-offload reference, and
the selected framework checkout's real `C8SparseKVCacheManager`, LIM bridge,
publication, ring staging and attention functions for HiSparse. Pool shells
provide inputs and allocation geometry; they do not implement the algorithms.
KV is packed with the framework's production FP8 packer. Requests have separate
physical pages, and cache comparisons resolve logical tokens through page tables.

After initializing the Ascend environment and installing the paired A5 DT wheel:

```bash
cd /mnt/share/y00970600/personal/sgl-kernel-npu
python3 benchmark/copysfa_a5_ab/trace_alignment.py \
  --framework-prefix /mnt/share/y00970600/personal/sglang \
  --device 0 --heads 128 \
  --output /mnt/share/y00970600/a5-alignment-long-q4
```

The default is local batch 2, Q4, C8192 and context 9344. Use `--query-width 1`
for Q1. A short-path case is a separate single command with `--context 1024`
and a different output directory. Short Q4 follows the framework's native-LI
bypass, Host gather and vendor QSFA path; it does not claim LIM/CopySFA coverage.
Short Q1 still calls LIM but does not use CopySFA.

Only a compact parsed result appears in the terminal and `summary.txt`:
cold/warm pass counts, actual path, miss counts, first differing stage and error
metrics. Detailed checks are in `report.json`; `inputs.pt`, `cold.pt`, and
`warm.pt` contain CPU tensor snapshots, and `run.log` captures verbose Python
and native output. Existing output directories are never overwritten.

The diagnostic checks pre-LIM metadata, per-query TopK sets, current KV packing
and Host publication, ring-tail contents/state, CopySFA page tables/lengths,
query packing, and selected logical KV after copying. It compares attention
both with independently selected native TopK and with the exact LIM-selected
IDs to separate retrieval from attention arithmetic. Metadata and packed KV
require exact agreement; attention uses the recorded `--atol`, `--rtol`, and
`--max-relative-l2` gates. No reference output is substituted on a failure.

This baseline deliberately sets split-query, fused-ring-tail, MF and hit
recording off, and both metadata fusions on, inside this process only. Previous
flag values, source SHAs and the installed native library hash are recorded.
Synchronizations and snapshots make this unsuitable for performance measurement.
It does not load model weights, exercise the scheduler/four-stream overlap,
or validate graph capture, whole-model accuracy, or optional v2 fusion.
Exit codes: 0 = all checks complete and pass; 1 = numerical/metadata mismatch;
2 = setup/runtime failure. A partial run cannot pass.

CPU-only checks for these diagnostic oracles (not A5 device validation):

```bash
python3 -m pytest -q benchmark/copysfa_a5_ab/test_tensor_trace_cpu.py
```

This harness compares three isolated A5 arms on one deterministic fixture:

1. `native`: the SGLang no-offload A5 path, using
   `torch.ops.npu.npu_quant_lightning_indexer` (the stock FP8 dispatcher) followed by
   `torch_npu.npu_kv_quant_sparse_flash_attention` with the exact packed-C8
   arguments used by the current SGLang backend.
2. `pre_migration`: the pre-migration A5
   `npu::fused_quant_lightning_indexer_manage` +
   `npu::fused_quant_scatter_copy_sparse_flash_attention` pair.
3. `post_migration`: the post-migration pair with the same operator schemas.

This is not the old vLLM-original CopySFA comparison. The native baseline is
SGLang's stock A5 indexer plus packed-C8 sparse attention. The old and new
same-named `torch.ops` libraries are never loaded into one interpreter: every
arm has its own Python executable, framework prefix, kernel prefix and optional
explicit shared-library list and runs in a separate process.

## Scope and timing

The fixture fixes all data that can be shared across the three implementations:
BF16 Q/QRoPE, FP8 index query/key, FP32 dequant scales, 656-byte packed-C8 KV,
positions, TopK=2048, page tables and initial cache contents. Request-level
selected-prefix union misses alternate between 512 and 513, so a local batch of
12 records six requests on each side of that boundary. Q1 and Q4 run in eager
and graph modes.

Each result contains three NPU Event measurements:

- `indexer_or_lim`: only the native quant indexer or LIM launch;
- `attention_or_copysfa`: native C8 sparse attention or CopySFA with one shared
  frozen TopK/copy plan;
- `chain`: native indexer + native attention, or LIM + CopySFA using LIM's live
  output.

Fixture construction, registered-host setup, allocations, cache/slot cold reset,
output transfer and validation are outside all event windows. Graph capture is
also outside timing. Accuracy is checked against the packed-C8 CPU golden, and
the chain's selected IDs are canonicalized (sorted per route) before exact
comparison. The default CPU-golden gate is `atol=0.005`, `rtol=0.02`; direct
old/new-to-native parity uses `atol=0.002`, `rtol=0.005`. Both require
relative-L2 <= 0.08. These are the old benchmark's gates, not relaxed values.
CLI overrides are recorded in results and remain hard failure gates.

NPU Event and profiler `kernel_details` are not the same measurement scope. The
event samples cover the launch/replay boundary observed on the selected stream
and can include queued execution or submission gaps; `kernel_details` reports
individual device kernels such as LIM or CopySFA. Use `--profile` when comparing
against a single-kernel profile number. Profiling runs only after all formal
event samples and never contributes samples to the performance result. Each
component gets its own trace directory and defaults to three iterations, with
`a5_three_arm::<component>::reset` and `::op` record-function markers separating
state restoration from the measured operator/replay.

The native and LIM implementations can order equal TopK entries differently.
The fixture places explicit score gaps at each route's TopK boundary so the
selected set remains unique after BF16 score sorting. A linear FP32 ramp alone
does not preserve this property at long contexts. The canonical-ID device gate
remains required; the CPU checks do not establish A5 execution acceptance.

## CPU-only validation

The CPU checks do not import Torch and do not execute an operator:

```bash
cd /path/to/sgl-kernel-npu
python3 benchmark/copysfa_a5_ab/test_cpu.py
python3 benchmark/copysfa_a5_ab/run_matrix.py \
  --arm-config benchmark/copysfa_a5_ab/arm-config.example.json \
  --output-root /tmp/a5-three-arm-dry-run \
  --device 0 --batch 12 --heads 8 --index-heads 32 --capacity 8192 \
  --prefix 9216 --context 9345 --dry-run
```

The dry run validates the three-arm contract and prints all 12 isolated process
commands. It does not require the placeholder paths to exist.

## Run on A5

If you have the two wheel files, prepare private prefixes with CLI parameters:

```bash
python3 benchmark/copysfa_a5_ab/prepare_wheels.py \
  --pre-wheel /path/to/pre-migration.whl \
  --post-wheel /path/to/post-migration.whl \
  --framework-prefix /mnt/share/y00970600/personal/sglang \
  --output-root /tmp/a5-three-arm-packages
```

Use `/tmp/a5-three-arm-packages/arms.json` below. This extracts the wheels without
pip-installing or changing the running service. `--pre-framework-prefix` can
record the retained old checkout; otherwise the current checkout is recorded
for all arms. This harness invokes raw operators with frozen metadata, so the
framework prefix supplies source provenance rather than a full model forward.
The pre wheel must be the retained pre-migration A5 wheel, not another copy of
the new wheel. Keep the observed post-migration wheel for the first comparison;
use a separate output root to compare a schedule-fixed wheel afterward.

Copy `arm-config.example.json` and replace every placeholder. `python` must be
the private interpreter for that arm. `framework_prefix` and `kernel_prefix`
must identify the exact checkouts or installed prefixes. Put each separately
loaded LIM/CopySFA DSO in `libraries`; every explicit DSO and every `.so` below
the imported `sgl_kernel_npu` package is SHA256-recorded. Git prefixes also
record `rev-parse HEAD`, and resolved operator schemas are saved in every
`result.json`.

Source CANN/ATB in the existing container before running. Do not globally
install or switch wheels. Confirm that device 0 belongs to this task, then run:

```bash
cd /path/to/sgl-kernel-npu
python3 benchmark/copysfa_a5_ab/run_matrix.py \
  --arm-config /path/to/a5-three-arm.json \
  --output-root /path/to/new/a5-three-arm-result \
  --device 0 \
  --batch 12 \
  --heads <local-query-heads> \
  --index-heads <local-index-heads> \
  --capacity <C> \
  --prefix <page-aligned-prefix> \
  --context <actual-context> \
  --miss-counts 512,513 \
  --warmup 20 --rounds 3 --iterations 50 \
  --profile --profile-iterations 3
```

Device selection is only the explicit `--device` argument (default 0). The
runner removes `ASCEND_RT_VISIBLE_DEVICES` and `NPU_VISIBLE_DEVICES` from child
environments; it does not set a device environment variable. Batch, heads, C
and context must come from the intended launch receipt. The previously discussed
12-request launch suggests a local batch of 12 only if attention-DP routing
actually confirms it; the harness does not infer heads, C or length.

The LIM fixture is a warm request (`request_state=-1`): its initial slot map
contains the chosen resident prefix rows in real request slots `1..B` (slot 0
stays reserved), and every sample restores that map and HBM state outside the
event interval. It intentionally does not use the cold `-2` state, which would
turn the requested 512/513 diagnostic into a full cache fill. Native indexer
scores use FP32 scales with BF16-safe gaps at the selected-set boundary;
canonical-ID acceptance still requires the real A5 run and is not claimed by
CPU validation.

Success requires all 12 child processes to exit zero and `summary.json` to have
`status: passed`. Each case preserves raw event samples, medians/p95, fixture
hashes, per-request 512/513 misses, canonical-ID checks, CPU-golden errors,
schemas, SHAs and library hashes. A failed output directory is evidence; use a
new output root for a retry.

`--profile` is optional and defaults off. When enabled, every `result.json`
records the post-timing trace directories and markers under `profiling`; use the
trace's device-kernel table to inspect the actual LIM and CopySFA kernel rows.

The result labels the standalone frozen plan separately from the live chain.
For each LIM arm it records the real `topk_misses`, request-union `miss_counts`,
selected-prefix union size and hit rate, whether any request selected the >512
fallback path, and whether live union misses match the designed profile. A
mismatch is a hard failure rather than being silently compared as the same
workload.

Do not use only the mixed profile to diagnose the 512 boundary: because it
contains a 513-miss request, `fallback_any` is necessarily true. Run two new
output roots with every request fixed on either side, then optionally run the
mixed workload:

```bash
python3 benchmark/copysfa_a5_ab/run_matrix.py \
  --arm-config /path/to/a5-three-arm.json \
  --output-root /path/to/result-m512 \
  --device 0 --batch 12 --heads <H> --index-heads <IH> \
  --capacity <C> --prefix <P> --context <L> --miss-counts 512

python3 benchmark/copysfa_a5_ab/run_matrix.py \
  --arm-config /path/to/a5-three-arm.json \
  --output-root /path/to/result-m513 \
  --device 0 --batch 12 --heads <H> --index-heads <IH> \
  --capacity <C> --prefix <P> --context <L> --miss-counts 513
```

`--miss-counts` accepts a comma-separated repeating per-request profile, so
`512,513` produces the mixed alternating batch and a single value applies to
all requests.

## Evidence boundary

### Higher request-union misses at long context

The original shared-tail fixture selects highly overlapping routes; its selected
historical union is less than 2048 for the 140032-token configuration. Do not
request 4096 or 8192 misses from that fixture. `--selection disjoint` instead
assigns each route its own 2048 historical IDs spread across the prefix. Query
feature groups select these sets with a positive-versus-zero score gap that
survives BF16 sorting. Q4 has an actual union of 8192 IDs; request-union misses
are distributed round-robin across the four routes. Q1 supports at most 2048.
No selected tail tokens are included in this stress fixture. The initial C-pool
remains a full warm pool, rather than using the first-fill request state.

Use the existing run root containing `packages-unfixed/arms.json`,
`packages-fixed/arms.json` and a previous `unfixed-m512/run-receipt.json`:

```bash
python3 benchmark/copysfa_a5_ab/run_miss_sweep.py \
  --run-root /path/to/previous/a5-op-run \
  --device 0
```

The command reuses those wheel prefixes and the prior shape/timing parameters;
it neither installs wheels nor edits PATH/PYTHONPATH. It runs native/pre/post
for both unfixed and fixed post wheels, at 512, 513, 1024, 2048, 4096 and 8192
misses per request. Defaults are Q4 graph and no profiling (36 arm processes).
Use `--misses 1024 2048 4096 8192` to select only high-miss cases,
`--modes eager graph` for both execution modes, or `--profile` for extra traces.
`--dry-run` prints commands without creating results or executing an NPU.
All selected cases are geometry-validated before device execution.

Results and per-matrix logs go into a fresh `more-misses-*` directory. The final
comparison includes scheduling deltas for each miss count. A failed matrix
stops the sweep and preserves its log and receipt. This is a controlled stress
fixture, not a replay of the real serving request's ID distribution. Compare
pre/post within the same selection layout and never mix it with shared-tail
results as if the inputs were identical.

The console prints one timing line per case; raw samples remain in each
`result.json`. Compare already collected results without rerunning operators:

```bash
python3 benchmark/copysfa_a5_ab/compare.py /path/to/retry-results
```

This prints mean NPU Event times in microseconds and migration deltas for LIM,
CopySFA and their chain. Positive percentages mean slower. Use
`--stat median_us` to compare medians. Missing or failed cases are shown;
timings without successful fixture and numerical validation are explicitly
marked as diagnostic only. When both `unfixed-m512`/`fixed-m512` (or m513)
matrices exist, scheduling deltas for the post-migration arm are also printed.
The command reads existing files and does not initialize an NPU.

Frozen-plan attention acceptance is shown independently: it requires matching
fixture hashes, its CPU golden check and output parity to native attention.
It does not consume live indexer IDs, so a failed canonical-ID gate does not
automatically invalidate that component. The full-case gate stays unchanged.

The scripts and fixture/result contracts can be checked on CPU, but numerical,
graph and event-timing acceptance require an actual Ascend950 run. Historical
aggregate observations (old CopySFA 196.220 us / LIM 319.968 us; new CopySFA
242.619 us / LIM 348.538 us) are not embedded as thresholds because they lack
this fixture's Q, batch, heads, C, cache state and per-sample distributions.
