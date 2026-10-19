# Copy-SFA for SGLang NPU sparse KV offload

This folder contains the SGLang-specific fused cache-copy and sparse-attention path. Its public operator is a combined-cache Q1 decode wrapper; it does not provide the variable-width MTP API described by the upstream vLLM Ascend implementation.

## SGLang integration contract

The registered operator is `torch.ops.npu.fused_scatter_copy_sparse_flash_attention`. The SGLang manager plans request-cache metadata before the fused launch. The fused operator gathers missing KV rows from the manager's registered, device-accessible host mapping into HBM, then computes sparse flash attention. HBM cache rows are `[CKV(512) | KPE(64)]` with a 576-element token stride; the caller supplies a preallocated output tensor and the cache is updated in place.

Supported integration scope:

- Non-speculative Q1 decode: `query` is TND `[B, H, 512]`, `query_rope` is `[B, H, 64]`, and each request contributes one query row. MTP and multi-query rows are unsupported by the SGLang wrapper.
- CKV 512, KPE 64, TopK/cache capacity 2048, page size 128, and one KV head. The HBM page table is `[B, 16]`; the DRAM table is `[B, dram_pages_per_request]`.
- FP16 or BF16 floating tensors; contiguous int32 metadata. Query, output, HBM cache, page tables, and metadata share one NPU. DRAM rows must be in the registered device-accessible host mapping used by the offload manager; arbitrary CPU allocations are unsupported.
- `combined_hbm_cache` is contiguous `[R*16, 128, 1, 576]`, where `R` is the manager's request-row capacity. TopK source/destination arrays are `[B, 1, 2048]`; request miss source/destination arrays are `[B, 2048]`.
- `attention_out` is caller-allocated and matches `query` shape and dtype. Active source IDs, slots, and block-table entries must be valid.

The SGLang path is enabled with both `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1` and `SGLANG_NPU_ENABLE_COPY_SFA=1`. Framework dispatch currently requires non-speculative Q1 decode, matching MLA/cache dimensions, and live sequence lengths of at least 2048 tokens (graph capture uses the framework's capture gate). Other calls use the existing attention path.

For a graph batch containing padded rows, `actual_seq_lengths_query` remains the dense cumulative prefix `[1, 2, ..., B]` so each row keeps its output position. A padded row has zero KV/cache-token lengths, sentinel planner IDs/slots, and zero miss counts. The kernel writes zero output for that row and leaves its cache state untouched.

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
