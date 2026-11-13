# CopySFA for Ascend sparse KV offload

The `npu::fused_scatter_copy_sparse_flash_attention` operator consumes a native
LIM selection/copy plan, updates a combined HBM KV cache, and computes sparse
attention. It supports registered combined host KV and separate registered CKV
and KPE host mappings.

## Contract

- Query and RoPE query are contiguous TND `[T,H,512]` and `[T,H,64]` tensors.
  Cumulative query lengths describe each request's query window.
- HBM KV is contiguous `[pages,128,1,576]`, with one KV head and CKV/KPE widths
  512/64. Floating tensors use FP16 or BF16 and share one NPU.
- TopK source/slot tensors are `[T,2048]`. Request-union miss buffers are
  contiguous `[B,M]`, with `[B]` valid counts; native LIM uses `M=32768`.
- The caller owns page tables, resident maps, registered host storage and the
  preallocated attention output. Active IDs and mapped addresses must be valid.
- `full_visible=True` consumes native LIM full-visible plans. Rewritten-resident
  refresh overlap additionally requires `resident_refresh_tail=True` and the
  matching suffix ordering in the LIM miss list.
- Supplying current KV, positions, valid masks, request slots, token mappings and
  resident extents enables short-history restoration and current-KV staging
  inside CopySFA. Keep these inputs alive until the consuming stream completes.
- Split host storage supplies separate registered CKV/KPE addresses and token
  strides. It does not change the combined HBM cache layout.

The schema and native validation are defined in `csrc/pytorch_extensions.cpp`
and `op_host/fused_copy_sfa.cpp`. SGLang selects the fused runtime with
`SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1` and
`SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA=1`.

## Tests

```bash
python -m pytest tests/test_copy_sfa_dynamic_tnd.py tests/test_copy_sfa_fused_resident.py
python -m pytest tests/test_lim_copy_sfa.py tests/test_lim_mtp_full_visible.py
```

These tests require a matching native extension and an available Ascend NPU.
They cover short/mixed requests, split host inputs, graph replay, causal current
KV, request reuse and resident refresh. The independent planner/reference tests
remain in `tests/test_sparsity_driven_kv_cache_planner.py`.

## Upstream provenance

The inherited MLA implementation originates from `vllm-project/vllm-ascend`
commit `85910cd528e2e260848e58e6337adf35cfd5d982`, under the CANN Open Software
License Agreement 2.0. `UPSTREAM_PROVENANCE.json` records original source hashes
and retained paths. Source copyright and license notices remain with the files.
