# A5 C8 scatter copy + sparse flash attention

`torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention` is an A5-only
normal-wheel operator ported from [vllm-ascend #18490](https://github.com/vllm-project/vllm-ascend/pull/18490),
commit `1b180deb840484c2fe33cbdd7abb68a615085172`. Donor source hashes and retained
licenses are recorded alongside the imported payloads. Use the explicit SoC
for the target, for example `bash build.sh -a kernels Ascend950DT_9582`.

The API takes:

```text
query, hbm_kv_bytes, dram_kv_bytes, dram_kv_dev_ptr,
hbm_block_table, dram_block_table, topk_dst_slots, topk_src_ids,
topk_miss_counts, actual_seq_lengths_query, actual_seq_lengths_kv,
num_cache_tokens, miss_src_ids, miss_dst_slots, miss_counts,
scale_value, attention_out
```

Query is BF16/FP16 `[T,H,576]`; output is `[T,H,512]` with the same dtype.
Supported head counts are 8, 16, 32, 64, and 128. Both KV tensors are opaque INT8
byte views `[pages,128,1,656]`: 512 FP8 NoPE bytes, 128 BF16 RoPE bytes, then
16 FP32 scale bytes. Copy preserves all 656 bytes.

The direct launch accepts a CPU tensor and its registered device-visible
address. HBM, query, metadata and output remain on one NPU. The caller must
retain registered storage throughout capture/replay and synchronize before
releasing it. The host also retains tensor and workspace owners while a
launch is queued.

`num_cache_tokens=0` means dense, using the real sequence length. Otherwise it is
the exposed HBM logical address-space length. `actual_seq_lengths_kv` remains
the real source length. TopK destination slots and both physical page tables
are authoritative: the operator does not derive a tail formula, invent a tail
mapping, or append tail tokens. Any caller-managed tail slots must already be
present in the supplied TopK destination list and covered by the HBM table.

TopK arrays are `[T,1,2048]`; miss counts are `[T]`. Request miss arrays are
`[B,1,32768]` with counts `[B]`. LIM's contiguous `[B,32768]` outputs can be
unsqueezed without copying. Metadata drives the fused miss path for at most 512
misses, or a complete union copy followed by attention inside the same kernel.
`actual_seq_lengths_query` contains the strictly increasing query end for each
of the `B` requests, and every segment may contain 1–16 rows. Thus Q4 MTP uses
four consecutive TopK rows for one request without changing the ABI. The
512/513 path decision is read from device metadata by the kernel, so a captured
graph can switch between fused miss gather and request-level union copy.

HBM/DRAM tables are `[B, pages_per_request]`; each entry is a physical block ID,
not an implicit contiguous-request offset. Rows may therefore be randomized or
share physical DRAM pages. They have at most 2048 pages per request, so the
effective source capacity is at most 262144 tokens at page size 128.
For a padded TND request, keep the positive query width and set the KV
length, cache size and miss counts to zero, with `-1` slots. With this
contract the kernel initializes the output and skips attention for the
padded request. An empty slot list alone does not trigger initialization.
Query/output storage stays BF16/FP16 even though KV storage is C8.

The fully visible entry `fused_quant_scatter_copy_sparse_flash_attention_v2`
replaces the packed query with separate BF16/FP16 `[T,H,512]` and `[T,H,64]`
inputs and accepts **`[B,32768]`** request miss arrays. The
[fully visible A5 LIM](../fused_quant_lightning_indexer_manage/README.md)
produces the causal TopK and copy plan in one launch. Long rows share one
request-union copy list across the query window, including retained slots
whose current KV must be refreshed. In steady state, routes with at most 512
ordinary misses use the source-aware SFA gather pipeline. If the union count
equals the largest route miss count, it contains no refresh entries. Otherwise
the kernel pre-copies the last (at most Q) union entries to cover all retained
refreshes. Cold rows, wider miss
routes, and legacy callers keep the complete-copy barrier. The overlap path is
enabled only when the caller also passes `native_lim_plan=True`, which asserts
the native LIM ordinary-prefix/refresh-suffix contract. Arbitrary-order callers
remain on the complete-copy path. Pass `current`, `positions`, `current_valid`,
`request_slots`, `req_to_token`, and the per-layer `resident_end` together to
enable resident staging. CopySFA restores missing
short history into identity slots and stages current KV. Plan entries for valid current positions
copy directly from the current layer's device buffer, including ordinary
misses gathered during attention; Host is used only for non-current sources.
Callers that stage short/current KV themselves can omit the resident tensors.

For a sparse row, supply `C + Q - 1` as the virtual scheduling length and
only `C / 128` HBM pages. The v2 kernel accepts LI order within each query.

```bash
python3 -m pytest -q tests/test_a5_full_visible_planner.py
```

CPU cases exercise the reference planner and the BF16 cutoff check. Ascend950
cases compare each Q1/Q4 TopK set with native Quant Lightning Indexer on
identical index inputs, allowing only substitutions tied at the cutoff.
The cases cover source IDs beyond 131072, resident refreshes with different
current and DRAM payloads, and graph replay. They check CopySFA v2 against the
platform C8 attention operator using the same selections.
