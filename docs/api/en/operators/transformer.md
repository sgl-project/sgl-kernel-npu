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

Select a binding path before starting pytest. Use `legacy` for original bindings
or `cann_ops_transformer` for released bindings; `auto` selects available complete
families. The `--impl` name selects the registered compatibility wrapper, while
the environment variable selects the underlying bindings.

```sh
export SGL_KERNEL_NPU_TRANSFORMER_BACKEND=legacy
# Or: export SGL_KERNEL_NPU_TRANSFORMER_BACKEND=cann_ops_transformer
python -m pytest tests/python/operators/attention/test_sparse_flash_mla.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py \
  tests/python/operators/norm/test_mhc.py \
  tests/python/operators/rope/test_inplace_partial_rotary_mul.py \
  tests/python/operators/kvcacheio/test_indexer_quant_cache.py \
  tests/python/operators/kvcacheio/test_kv_compress_epilog.py \
  --impl cann_ops_transformer --impl-ref torch_native \
  --npu-device 0 --enable-perf=false -v
```

All numerical comparisons use `check_accuracy` and profiling uses the returned
executor with `check_performance`. mHC pre/post have independent inputs and
preserve their original tolerances. In-place operators use separate reference
buffers via `args_ref`. Attention prepares each implementation's metadata once
and passes reference overrides via `kwargs_ref`; compute timing excludes metadata
and KV initialization. The KV writer's accuracy-only `output_transform` compares
semantic bytes exactly, excluding unspecified alignment padding. Indexer FP8
values are widened only for comparison, retaining FP8-specific tolerances and
separate FP32 scale tolerances.

The A5 mixed-quant reader/writer tests skip on A3. The current on-device files
cover mHC pre/post, partial RoPE, cache writers, BF16/mixed-quant attention with
their metadata, and MoE gating. They do not yet provide standalone accuracy or
performance coverage for the quantized lightning indexer or compressor, nor do
they validate graph replay or model inference. Metadata descriptors are checked
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
custom/native binding or the released binding inside the kernel package. This
is a binding/ABI compatibility layer, not a comparison of SDK version strings.

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

Selection is cached before execution. `auto` prefers complete released families
and otherwise selects complete legacy families. Metadata and compute are paired;
the quantized KV writer is part of the same family. A broken installed package
or an execution failure is raised to the caller, not retried on another backend.
Optional schema features, including MoE hash inputs, are checked before execution.

To force the original bindings when both installations are present, set this
before importing/using the operators and before constructing pools or graphs:

```sh
export SGL_KERNEL_NPU_TRANSFORMER_BACKEND=legacy
# Alternatives: auto (default), cann_ops_transformer (require released bindings).
```

Original custom operators must already be registered. Optionally,
`SGL_KERNEL_NPU_LEGACY_OPS_SO` can identify their standalone library. There is
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

