# DSV4 released Transformer operators

The public APIs live in `sgl_kernel_npu.operators.dsv4`. Each operator registers
`auto`, `torch_native`, and `cann_ops_transformer` through the existing
hardware-aware operator registry. `auto` aliases `cann_ops_transformer`; it does
not silently fall back when the CANN package is missing. Backend imports remain
lazy. This package still requires a built NPU wheel for normal production imports.

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
- `NPU_KV_QUANT_PATH` defines 64-element NoPE groups, bf16 scales, and 32-byte row
  alignment. `packed_dim(448, 64)` returns 608. The writer adapter owns the raw
  byte view. Invalid mixed-quant write slots use -1; slot 0 remains valid.
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
The mixed-quant reference supports the DSV4 448-NoPE/64-RoPE 608-byte format.
The indexer reference supports causal TND/PA_BBND FP8/INT8 inputs. The compressor
reference supports coff=1/2 and paged, explicit-table, or request-bank cycle state.
Unsupported layouts and modes raise rather than forwarding to CANN.

GPU-shared model logic, scheduling, graph replay, DSpark index construction,
cache allocation and ordinary torch/torch_npu operations remain in SGLang.

## Validation

CPU contracts and references (the tests isolate only the NPU bootstrap):

```sh
python -m pytest tests/unit/test_dsv4_transformer.py -q
```

On an Ascend host with a built wheel and the released CANN transformer package:

```sh
python -m pytest tests/python/operators/dsv4/test_transformer.py \
  --impl cann_ops_transformer --impl-ref torch_native -q
```

Hardware comparisons are required before promoting this migration out of draft,
especially packed KV scale rounding/byte order and compressor state retention.
