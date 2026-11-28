# CopySFA for Ascend sparse KV offload

The split Copy-SFA operator consumes a selection/copy plan, updates separate
CKV and KPE HBM caches, and computes sparse attention. Use
`sgl_kernel_npu.operators.attention.fused_scatter_copy_sparse_flash_attention_split`.

## Contract

- Query and RoPE query are contiguous TND `[T,H,512]` and `[T,H,64]` tensors.
  Cumulative query lengths describe each request's query window.
- HBM CKV and KPE are contiguous tensors of shape
  `[pages,128,1,512]` and `[pages,128,1,64]`. Floating tensors use FP16 or
  BF16 and share one NPU; the host pointers refer to separate registered regions.
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
The wheel registers the split Torch schema in `torch_adapter.cpp` and
uses ACLNN. Build the operator from the repository root:

```bash
python3 setup.py --arch-version 2201 --ops fused_scatter_copy_sparse_flash_attention_split --without-third-party
```

SGLang selects the fused runtime with `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1`
and `SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA=1`.

## Tests

```bash
python -m pytest tests/python/operators/attention/test_copy_sfa_split_hbm_resident.py
```

These tests require a matching native extension, a registered host-memory
allocator, and an available Ascend NPU. They cover split host inputs, graph
replay, causal current KV, request reuse and resident refresh.

## Upstream provenance

The inherited MLA implementation originates from `vllm-project/vllm-ascend`
commit `85910cd528e2e260848e58e6337adf35cfd5d982`, under the
[CANN Open Software License Agreement 2.0](https://github.com/hicann/ops-transformer/blob/91de0d9959ab57be7fcbef8b0592d16c242c5955/LICENSE). `UPSTREAM_PROVENANCE.json` records original source hashes
and retained paths. Source copyright and license notices remain with the files.
