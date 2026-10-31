# Quantized sparse shared-KV attention

`torch.ops.npu.kv_quant_sparse_attn_sharedkv` computes attention over paged,
quantized KV caches on Ascend 950. It supports sliding-window attention over
original KV and attention that also reads a compressed KV cache. Multiple query
heads share one KV head. Callers can supply per-query sparse indices to select
original-KV cache slots; the operator consumes these indices rather than
computing a top-k selection.

## Interface

```python
import torch
import sgl_kernel_npu  # Register torch.ops.npu.* operators.

metadata = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    layout_q="TND",
    layout_kv="PA_ND",
    cu_seqlens_q=cu_seqlens_q_cpu,
    seqused_kv=seqused_kv_cpu,
    batch_size=batch_size,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=max_seqlen_kv,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=ori_win_left,
    ori_win_right=ori_win_right,
    has_ori_kv=True,
    has_cmp_kv=False,
)

attention_out, softmax_lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(
    q,
    ori_kv=ori_kv,
    ori_sparse_indices=ori_sparse_indices,
    ori_block_table=ori_block_table,
    cu_seqlens_q=cu_seqlens_q_npu,
    seqused_kv=seqused_kv_npu,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=softmax_scale,
    cmp_ratio=1,
    ori_win_left=ori_win_left,
    ori_win_right=ori_win_right,
    layout_q="TND",
    layout_kv="PA_ND",
)
```

This example uses original-KV sparse indices without a compressed cache. The
metadata builder accepts contiguous **CPU** `int32` sequence-length tensors and
returns an NPU `int32[1024]` task table. The attention operator takes its
sequence-length tensors on the NPU. The metadata table assigns work to 36 cube
core slots (9 integers each) and 72 vector core slots (8 integers each); use
`kv_quant_sparse_attn_sharedkv_metadata_host` to produce it.

## Inputs and outputs

| Tensor | Shape and dtype | Description |
| --- | --- | --- |
| `q` | Contiguous BF16 `[B, S, 64\|128, 512]` (`BSND`) or `[T, 64\|128, 512]` (`TND`) | Query; `T` is the total number of query tokens. |
| `ori_kv` | Contiguous FP8 E4M3FN `[num_blocks, block_size, 1, 640]` | Required packed original-KV cache. |
| `cmp_kv` | Same packed format as `ori_kv` | Optional compressed-KV cache; requires `cmp_block_table`. |
| `ori_block_table` | NPU `int32` | Required logical-to-physical block table for paged original KV. |
| `ori_sparse_indices` | Contiguous NPU `int32[T, 1, K]` | Optional physical cache-slot IDs for each query; supported with `TND` and `cmp_kv=None`. |
| `cmp_sparse_indices` | NPU `int32` | Optional compressed-KV selection indices; requires `cmp_kv`. |
| `cu_seqlens_q` | NPU `int32[B+1]` | Cumulative query lengths; required for `TND`. |
| `seqused_kv` | NPU `int32[B]` | Required valid KV lengths. |
| `sinks` | NPU FP32 | Required softmax sinks. |
| `metadata` | NPU `int32[1024]` | Required core-scheduling table. |
| `attention_out` | Same shape and BF16 dtype as `q` | Attention result. |
| `softmax_lse` | Empty FP32 tensor | Placeholder output; log-sum-exp is not available. |

The query head count is 64 or 128; the KV head count is one. KV block sizes
must be divisible by 16 and no greater than 1024. Each 640-byte cache row is
a packed structure containing 64 BF16 RoPE values, 448 FP8 NoPE values, seven
E8M0 scale bytes, and padding. A regular tensor of 640 independent FP8 values
is not a valid cache row.

Each `ori_sparse_indices[t, 0]` row contains **absolute physical cache slots**:
`block_id * block_size + offset`. Put valid slots first and pad the remaining
entries with `-1`; use `K=512` or `1024`. These slots have already been resolved
through the block table by the caller, so the sparse-index path reads them
directly. Without `ori_sparse_indices`, the paged-KV path uses the block table.
The current original-KV sparse-index path requires `TND` and does not accept
`cmp_kv` at the same time.

## Configuration

- `kv_quant_mode=1`, `tile_size=64`, `rope_head_dim=64`,
  `ori_mask_mode=4`, `cmp_mask_mode=3`, and `layout_kv="PA_ND"` are required.
- Use `cmp_ratio=1` for original-KV-only sliding-window attention. Compressed-KV
  attention uses `cmp_ratio=4` or `128`.
- `ori_win_left` and `ori_win_right` accept nonnegative `int32` values.
- `return_softmax_lse=True`, `cu_seqlens_ori_kv`, and `cu_seqlens_cmp_kv` are
  unsupported.

## Implementation

The host entry in [`op_host/kv_quant_sparse_attn_sharedkv.cpp`](op_host/kv_quant_sparse_attn_sharedkv.cpp)
validates inputs, computes tiling, allocates workspace, and launches the AscendC
kernel. The device entry in
[`op_kernel/kv_quant_sparse_attn_sharedkv.cpp`](op_kernel/kv_quant_sparse_attn_sharedkv.cpp)
selects the kernel specialization. The main loop reads the metadata task table,
loads and dequantizes selected KV rows, and performs `QKᵀ`, softmax, and the
weighted sum with V in blocks. The separate
[`metadata` operator](../kv_quant_sparse_attn_sharedkv_metadata/README.md)
generates the task table on the host.

## Build and test

On Ascend 950, run from the repository root:

```bash
./build.sh -a kernels Ascend950PR_9599
python -m pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
python -m pytest tests/python/operators/attention/ --npu-device=0 --enable-perf=false -v -s
```

The main test packs nonzero BF16 RoPE values, FP8 NoPE values, and E8M0 scales
into KV rows, then compares the BF16 output with an independent PyTorch
reference. It checks noncontiguous physical slots, the `-1` terminator, and a
selection that crosses the 128-row S2 tile boundary. The test requires Ascend
950. The shared test framework selects the NPU using `--npu-device`.

The public API is documented in
[`docs/api/en/operators/attention/kv_quant_sparse_attn_sharedkv.md`](../../docs/api/en/operators/attention/kv_quant_sparse_attn_sharedkv.md).
