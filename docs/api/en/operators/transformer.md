# DSV4 released Transformer operators

The public APIs follow the existing category layout under
`sgl_kernel_npu.operators`: `attention`, `kvcacheio`, `norm`, `rope`, and `moe`. Each operator registers `auto`, `torch_native`, and a named backend through the
existing hardware-aware registry. The CANN adapters use `cann_ops_transformer`;
MoE gating uses `torch_npu`. `auto` aliases the corresponding production
compatibility wrapper, which selects complete released or original binding
families before execution. It never retries a failed kernel or automatically
uses the torch reference. Imports remain lazy. This package still requires a
built NPU wheel for normal production imports.

| Public operator | Released/native entry | Generation |
| --- | --- | --- |
| `sparse_flash_mla_metadata` | same name in `cann_ops_transformer` | A3/A5 |
| `sparse_flash_mla` | same name in `cann_ops_transformer` | A3/A5 |
| `mixed_quant_sparse_flash_mla_metadata` | same name in `cann_ops_transformer` | A5 |
| `mixed_quant_sparse_flash_mla` | same name in `cann_ops_transformer` | A5 |
| `quant_lightning_indexer_metadata` | same name in `cann_ops_transformer` | A3/A5 |
| `quant_lightning_indexer` | same name in `cann_ops_transformer` | A3/A5 |
| `kv_compress_epilog` | same name in `cann_ops_transformer` | A5 |
| `indexer_quant_cache` | same name in `cann_ops_transformer` | A3/A5 |
| `mhc_pre` | `cann_ops_transformer.mhc_pre_sinkhorn` | A3/A5 |
| `mhc_post` | `cann_ops_transformer.mhc_post` | A3/A5 |
| `inplace_partial_rotary_mul` | same name in `cann_ops_transformer` | A3/A5 |
| `compressor` | `torch.ops.npu.compressor` (existing A3/A5 binding) | A3/A5 |
| `moe_gating_top_k` | `torch.ops.npu.npu_moe_gating_top_k` (`torch_npu` backend) | A3/A5 |

The `cann_ops_transformer` implementation name identifies the migrated backend;
compressor intentionally retains its `npu` namespace binding from SGLang PR
47438. No C++ kernels or existing public operator APIs are removed.

## Contracts retained from SGLang PR 47438

- Attention/indexer use TND queries and PA_BBND paged KV, with separate original,
  compressed and residual lengths. Sparse indices are logical positions.
- `mhc_pre` takes `[T, n, c]` input and returns `(y, post, comb)` with shapes
  `[T, c]`, `[T, n]`, `[T, n, n]`. `y` preserves the activation dtype; the
  operation does not fold the model's input layernorm. The CANN adapter owns the
  BSND view and return-order adaptation.
- `mhc_post(x, residual, post, comb)` adapts to CANN's operand order internally.
- Cache writers and partial RoPE mutate their input storage. Callers must clone
  inputs when comparing implementations; do not run both on the same buffers.
- `get_npu_kv_quant_path()` supplies the selected 608B BF16-scale or original
  640B E8M0-scale contract before allocation. The retained `NPU_KV_QUANT_PATH`
  constant describes the released format only. The writer adapter owns raw byte
  views and vendor enum mapping. Invalid mixed-quant write slots use -1; slot 0
  remains valid.
- First-call CANN symbol lookup is protected from custom vendor shadowing. The
  temporary process environment change is serialized across operators, restored
  on failure, and does not run again after a successful first invocation.

## Reference scope

`torch_native` uses independent eager PyTorch computations and is for correctness
checks, not graph capture or an automatic production fallback. Its metadata is a
reference descriptor rather than an opaque CANN tiling tensor: build metadata
with the same implementation as its compute operator. It is not meaningful to
compare metadata bytes across backends.

The attention reference supports one shared KV head, TND/BSND queries, PA_BBND
pages, sliding-window or explicit original indices, and causal compressed KV.
The mixed-quant reference supports DSV4 448-NoPE/64-RoPE with both 608-byte
BF16-scale and original 640-byte E8M0-scale rows.
The indexer reference supports causal TND/PA_BBND FP8/INT8 inputs. The compressor
reference supports coff=1/2 and paged, explicit-table, or request-bank cycle state.
Unsupported layouts and modes raise rather than forwarding to CANN.

`moe_gating_top_k` follows the released [torch_npu schema](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml):
FP32/FP16/BF16 `[tokens, experts]` inputs, optional correction bias/hash mapping,
softmax/sigmoid/sqrt-softplus scoring, grouped selection, renormalization and
route scaling. It returns weights in the input dtype and int32 IDs. The unused
third native output is normalized to an empty FP32 tensor when `out_flag=False`;
when true, it contains the full FP32 normalized scores. The independent torch
reference supports those same mathematical paths; ties may have different
valid expert orders across implementations. On A2/A3, use `group_count=1` for
sqrt-softplus; native hardware constraints remain those of the installed CANN.

GPU-shared model logic, scheduling, graph replay, DSpark index construction,
cache allocation and ordinary torch/torch_npu operations remain in SGLang.

## Validation

Run operator accuracy tests on an Ascend host with a built wheel containing the
current Python APIs and the matching native operator libraries. The aggregate
CPU/mock test module has been removed; these tests execute the selected NPU
binding against independent `torch_native` results.

Initialize the target CANN SDK before starting pytest. CANN 9.1.x selects the
original bindings; CANN 9.2.x selects the released bindings. The `--impl` name
selects the registered compatibility wrapper, while SDK version metadata selects
the underlying bindings. No backend-selection environment variable is required.

```sh
python -m pytest tests/python/operators/attention/test_sparse_flash_mla.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py \
  tests/python/operators/norm/test_mhc_pre.py \
  tests/python/operators/norm/test_mhc_post.py \
  tests/python/operators/rope/test_inplace_partial_rotary_mul.py \
  tests/python/operators/kvcacheio/test_indexer_quant_cache.py \
  tests/python/operators/kvcacheio/test_kv_compress_epilog.py \
  --impl cann_ops_transformer --impl-ref torch_native \
  --npu-device 0 --enable-perf=false -v
```

mHC and MoE use the unchanged `check_accuracy` helper directly. In-place RoPE,
cache writers and attention with implementation-specific metadata use the
existing `assert_accuracy` and `make_executor` helpers within their test files.
They compare independent reference buffers/metadata and set `accuracy_checked`
only after all comparisons pass. All cases use `check_performance` for profiling
and honor the existing implementation-selection and accuracy/performance CLI
switches; skipping accuracy does not invoke the reference implementation.

Compute timing excludes metadata, KV initialization and comparison conversions.
KV writer checks compare semantic bytes exactly, excluding unspecified alignment
padding. Indexer FP8 values are widened locally in its test, with separate FP32
scale tolerances. No changes to the shared accuracy helper are required.

The A5 mixed-quant reader/writer tests skip on A3. The current on-device files
cover mHC pre/post, partial RoPE, cache writers, BF16/mixed-quant attention with
their metadata, MoE gating, the quantized lightning indexer and its metadata,
and compressor. Indexer cases use strictly ordered scores for exact integer
Top-K comparison, including ragged queries, page boundaries and empty causal
prefixes. Metadata has its own timing; semantic validation uses paired compute.
Compressor covers C4/C128, TH/BSH, paged/cycle caches, and multi-round state
retention across ring wrap. Output padding is excluded, while the entire state
cache is checked against an independent reference buffer. These tests do not
validate graph replay or model inference. Metadata descriptors are checked
through their paired compute, not compared byte-for-byte across backends.

Hardware comparisons remain required before promoting this migration out of
draft, especially packed KV scale rounding/byte order and compressor state
retention. Collection skips on a host without `torch_npu` are not NPU validation.

MoE accuracy and selected-shape performance tests:

```sh
python -m pytest tests/python/operators/moe/test_moe_gating_top_k.py \
  --impl torch_npu --impl-ref torch_native --enable-perf=false -v
```

Omit `--enable-perf=false` to profile only each file's selected representative
shapes, or pass `--enable-perf=true` to profile all selected cases. Existing
`--prof-dir` and `--csv-path` options control profile and CSV output. FP32/FP16/BF16
tolerances are initial on-device thresholds, not evidence of completed hardware
validation.

## Category layout

| Category | Public APIs |
| --- | --- |
| `attention` | sparse/mixed-quant attention and metadata, quantized indexer and metadata, compressor |
| `kvcacheio` | KV/indexer cache writers and shared packed-KV geometry |
| `norm` | mHC pre/post |
| `rope` | partial rotary multiplication |
| `moe` | MoE gating top-k |

Each public operator has its own file. The 12 CANN adapter bodies live directly
in their registered backend functions under `operators/<category>/`; there is no
separate `implementation/cann_ops_transformer` forwarding layer. The shared lazy
loader and first-call symbol guards live in
`operators/utils/_cann_ops_transformer.py`. Independent torch references keep
their matching paths under `implementation/torch_native/`, and the MoE production
backend is implemented directly in `operators/moe/moe_gating_top_k.py`. Registry
names use the same category prefix. No model-specific `operators/dsv4` package
is introduced.

Operator-specific released/legacy input and output conversions live in the
registered functions (or private helpers) of each operator's own file. Generic
binding lookup, backend preference and schema checks stay in
`operators/utils/_transformer_compat.py`. Attention-specific metadata/compute
and packed-KV writer selection is shared in `operators/attention/_backend.py`;
the KV layout query and writer use that same cached selection. Tensor-only
logical-to-physical sparse index conversion is shared in
`operators/attention/_sparse_indices.py`. These shared modules do not own
the individual operators' ABI argument mappings.

## Released and original operator compatibility

The public imports, `impl="auto"`, `torch_native`, and the existing production
implementation names stay stable. Production wrappers now resolve the original
custom/native binding or the released binding inside the kernel package. The initialized CANN SDK selects
the binding/ABI compatibility path: CANN 9.1.x uses the original bindings and
CANN 9.2.x uses the released bindings.

| Public API | Original production binding | Released/#939 production binding |
| --- | --- | --- |
| `sparse_flash_mla_metadata` | `custom.npu_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.sparse_flash_mla_metadata` |
| `sparse_flash_mla` | `custom.npu_sparse_attn_sharedkv` | `cann_ops_transformer.sparse_flash_mla` |
| `mixed_quant_sparse_flash_mla_metadata` | `custom.npu_kv_quant_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata` |
| `mixed_quant_sparse_flash_mla` | `custom.npu_kv_quant_sparse_attn_sharedkv` | `cann_ops_transformer.mixed_quant_sparse_flash_mla` |
| `quant_lightning_indexer_metadata` | `custom.npu_quant_lightning_indexer_metadata` | `cann_ops_transformer.quant_lightning_indexer_metadata` |
| `quant_lightning_indexer` | `custom.npu_quant_lightning_indexer` | `cann_ops_transformer.quant_lightning_indexer` |
| `kv_compress_epilog` | `npu.kv_compress_epilog` | `cann_ops_transformer.kv_compress_epilog` |
| `indexer_quant_cache` | `custom.indexer_compress_epilog` | `cann_ops_transformer.indexer_quant_cache` |
| `mhc_pre` | `custom.npu_hc_pre` | `cann_ops_transformer.mhc_pre_sinkhorn` |
| `mhc_post` | `custom.npu_hc_post` | `cann_ops_transformer.mhc_post` |
| `inplace_partial_rotary_mul` | `custom.inplace_partial_rotary_mul` | `cann_ops_transformer.inplace_partial_rotary_mul` |
| `compressor` | A3: `npu.compressor`; A5: `custom.compressor` | `npu.compressor` (fused normalization/RoPE contract) |
| `moe_gating_top_k` | `custom.npu_moe_gating_top_k` | `npu.npu_moe_gating_top_k` |

Names omit the `torch.ops` prefix. MoE production execution, binding selection
and legacy adaptation are implemented directly in its public operator module.

The kernel reads an exact `major.minor.patch` release from the initialized
`ASCEND_HOME_PATH`, checking `version.cfg` first and `compiler/version.info`
second. The `version` and `version_dir` keys are case-insensitive. No separate
backend-selection environment variable is needed.

| Detected SDK | Required binding path |
| --- | --- |
| CANN 9.1.x | Original custom/native bindings |
| CANN 9.2.x | Released bindings, retaining the documented fused native compressor |
| Missing/inexact metadata or another release | Error before operator execution |

Initialize the intended SDK using its `set_env.sh` before starting the service.
Version detection and family selection are lazy and cached before execution;
do not switch SDK installations after allocating KV pools or capturing graphs.
Metadata and compute are paired, and the quantized KV writer belongs to the
same version-selected family. Missing bindings, unsupported schema features
(including MoE hash inputs), broken packages and kernel failures are reported
without trying another version's bindings.

On the CANN 9.1 custom-binding path, the kernel package lazily imports
`custom_ops` to register the original operators. Already registered operators
remain usable when that package is absent; broken imports are reported.
Optionally, `SGL_KERNEL_NPU_LEGACY_OPS_SO` can identify a standalone registration
library. It is unnecessary when `custom_ops` supplies the required bindings. There is
no compilation, dependency installation, or retry after a kernel has executed.

Query `sgl_kernel_npu.operators.kvcacheio.get_npu_kv_quant_path()` before allocating
A5 KV pools. It returns the selected immutable format: released mode 1 uses
608-byte rows with BF16 scales; the original reader/writer use 640-byte rows with
E8M0 scales. `kv_compress_epilog(..., quant_mode=None)` uses this same selection.
SGLang supplies model lengths, page tables and buffers, without choosing vendor
enum values or a version-dependent row width. Never reuse already allocated
cache data after changing installations or backend selection.

Legacy DSpark uses the existing paired host metadata/native attention bindings.
Supply `cu_seqlens_q_cpu` and `seqused_ori_kv_cpu` to the public metadata API from
the framework's existing CPU int32 mirrors. Released and reference metadata
ignore these host-only inputs. No NPU-to-CPU copy is performed in this adapter.
Original sparse indices are resolved from logical positions through the page
table to native physical slots with tensor operations; padded lengths and `-1`
entries are retained. Legacy explicit original KV plus compressed KV in the
same call is rejected instead of silently changing the mask.

The compatibility scope is the DSV4 paths migrated in this PR: shared-KV,
mode-1 quantized KV with 64-element groups/64 RoPE features, the migrated
FP8/INT8 indexer modes and the fused compressor API. SDK prototypes or matching
schemas do not prove a kernel supports every generation/mode. On-device old/new
accuracy, packed-byte rounding, graph capture/replay and performance still need
validation with the actual deployed operator libraries.

### Indexer and compressor standalone tests

Run from the checkout root with a matching wheel and initialized CANN SDK:

```bash
python3 -m pytest \
  tests/python/operators/attention/test_quant_lightning_indexer.py \
  tests/python/operators/attention/test_quant_lightning_indexer_metadata.py \
  tests/python/operators/attention/test_compressor.py \
  --impl auto --impl-ref torch_native --npu-device 0 \
  --skip-acc-check=false --enable-perf=false -v -s -x -rs
```

Omit `--enable-perf=false` to profile selected shapes, or use
`--enable-perf=true` to profile every non-streaming case. Streaming cases only
check accuracy: repeated profiling would consume already-mutated history.
A3 uses INT8 indexer inputs with FP16 scales/weights; A5 uses FP8 inputs with
FP32 scales/weights. Run the same command separately under CANN 9.1 and 9.2;
collection or CPU reference checks do not establish native kernel accuracy.
