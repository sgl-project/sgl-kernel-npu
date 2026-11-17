# A5 native vs pre/post-migration LIM/CopySFA microbenchmark

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
