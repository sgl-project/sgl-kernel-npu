# Copy-SFA for SGLang NPU sparse KV offload

This folder contains the SGLang-specific fused cache-copy and sparse-attention path. The combined-cache operator accepts dynamic TND queries; SGLang's current LIM integration continues to use non-speculative Q1 with full-visible TopK selection.

## SGLang integration contract

The registered operator is `torch.ops.npu.fused_scatter_copy_sparse_flash_attention`. The SGLang manager plans request-cache metadata before the fused launch. The fused operator gathers missing KV rows from the manager's registered, device-accessible host mapping into HBM, then computes sparse flash attention. HBM cache rows are `[CKV(512) | KPE(64)]` with a 576-element token stride; the caller supplies a preallocated output tensor and the cache is updated in place.

Supported integration scope:

- `query` is TND `[T, H, 512]`, `query_rope` is `[T, H, 64]`, and cumulative query lengths have one entry per request. The native host accepts `B <= T <= 16*B`; SGLang's framework route still requires `T=B` and disables speculation.
- CKV 512, KPE 64, TopK 2048, page size 128, and one KV head. The HBM page table is `[B, hbm_pages_per_request]`; the DRAM table is `[B, dram_pages_per_request]`. The native LIM framework supports resident capacities `C=2048`, `4096`, and `8192`, using 16, 32, and 64 HBM pages per request respectively. Attention still selects 2048 tokens; a cold fill copies all C resident payloads.
- FP16 or BF16 floating tensors; contiguous int32 metadata. Query, output, HBM cache, page tables, and metadata share one NPU. DRAM rows must be in the registered device-accessible host mapping used by the offload manager; arbitrary CPU allocations are unsupported.
- `combined_hbm_cache` is contiguous `[pages, 128, 1, 576]`. TopK source/destination arrays are `[T, 1, 2048]`; per-query TopK miss counts are `[T]`. Request-union miss source/destination arrays are contiguous `[B, M]`, with `[B]` miss counts. `M` may be 32768 to pass native LIM buffers directly without packing a strided prefix into a separate workspace.
- `attention_out` is caller-allocated and matches `query` shape and dtype. Active source IDs, slots, and block-table entries must be valid.

The SGLang path is enabled with both `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1` and `SGLANG_NPU_ENABLE_COPY_SFA=1`. Framework dispatch currently requires non-speculative Q1 decode, matching MLA/cache dimensions, and live sequence lengths of at least C tokens (graph capture uses the framework's capture gate). Shorter calls use the existing attention path and keep the original protection below 2048. C larger than TopK requires native LIM.

For a Q1 graph batch containing padded rows, `actual_seq_lengths_query` remains the dense cumulative prefix `[1, 2, ..., B]` so each row keeps its output position. A padded row has zero KV/cache-token lengths and zero miss counts; its unused source/slot contents are not read. The kernel writes zero output for that row and leaves its cache state untouched.

## LIM integration and dynamic TND changes

The full-visible Q1 framework uses LIM as the only residency owner, passes its complete contiguous miss plan into Copy-SFA, and retains the partial-vector eviction fixes in the LIM operator. Dynamic TND, combined-tail row-stride handling and the dense `C=0` miss-count guard are adapted from JinyanYi/sgl-kernel-npu commit `ffe1d44` (branch head `e74d7dfff432725ecccbc58149c7fd2e228c6b6c`). This does not switch the framework to prefix-tail attention or enable model-level MTP.

`tests/test_copy_sfa_dynamic_tnd.py` covers heterogeneous query counts/cache budgets, dense-to-sparse graph transitions, mixed first-fill/steady misses, tail page crossings and poisoned future tails. `tests/test_lim_copy_sfa.py` retains full-visible partial-page/padding/reuse coverage for both packed and direct 32768-wide miss plans.

Validated on Ascend A3 host 217 on 2026-10-23: a full normal-wheel build with CANN 9.1 and torch/torch_npu 2.10 completed, and all 53 selected native LIM, cache-planner and Copy-SFA tests passed. This is operator validation; it does not establish model-level MTP support or a model performance improvement.

The merged capacity integration was validated on 2026-10-24 with 61 native tests, including C2048/4096/8192 crossed with eager/graph and packed/direct miss plans. Real-manager checks also covered all three capacities with 16 graph batch rows and three request storage rows, bitwise resident KV and an FP32 attention oracle. These fixtures do not establish GPQA accuracy or repair of long generation loops.

## Cache planner behavior

The planner mutates the source-to-slot and slot-to-source maps in place. A sort-join matches requested sources against current slot owners, emits unmatched requested sources as a miss prefix, and emits retained hits as a suffix. Unmatched old slots are paired with misses in deterministic sort order; the slot policy does not promise that invalid/free slots are preferred over all evictable slots. The manager owns map reset/reuse semantics and preallocates stable metadata buffers for graph replay.

## Invocation shape

The schema is defined in `csrc/pytorch_extensions.cpp`:

```python
torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
    query, query_rope, combined_hbm_cache,
    actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens,
    topk_src_ids, topk_dst_slots, topk_miss_counts,
    miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, dram_kv_dev_ptr,
    attention_out, scale_value,
)
```

`dram_kv_dev_ptr` is the registered device address for the manager's combined mapped host rows. The wrapper derives the KPE address from the CKV base and element width; do not pass an arbitrary CPU pointer.

## Upstream provenance

The inherited MLA implementation originated from `vllm-project/vllm-ascend` commit `85910cd528e2e260848e58e6337adf35cfd5d982`, under the CANN Open Software License Agreement 2.0. `UPSTREAM_PROVENANCE.json` records every original file hash and whether that path is retained in this source tree. The pristine 19-file snapshot remains in the task artifact archive with SHA-256 `ebd771541c27c4f9a5962ee4826446a378e045fba0f164b994d185ab49dcbe4b`; it is not bundled in this source folder. The retained upstream implementation files are `first_fill_scatter_copy_kernel.h` and the four MLA common/kernel/service headers. The upstream Torch adapter and unused OPP/ACLNN host/kernel sources are preserved only in the task artifact archive.

## Validation boundary

Native integration tests are in `tests/test_copy_sfa_combined.py`; planner coverage is in `tests/test_sparsity_driven_kv_cache_planner.py`. Run them against a built extension on an available A3 NPU. CPU planner-reference tests do not validate the AscendC kernel. This interface makes no claim of MTP, split-source support, arbitrary cache dimensions, or performance improvements beyond separately reported measurements.
