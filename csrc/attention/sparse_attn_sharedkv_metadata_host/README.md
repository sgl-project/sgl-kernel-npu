# torch.ops.npu.sparse_attn_sharedkv_metadata_host

## Attribution

Reimplemented on the host CPU with reference to the vllm-ascend AICPU op
`SparseAttnSharedkvMetadata` (`vllm-ascend/csrc/attention/sparse_attn_sharedkv_metadata`,
Copyright (c) Huawei Technologies Co., Ltd., CANN Open Software License Version 2.0).
The scheduling algorithm is ported unchanged; the output is byte-identical.

## Function Description

Host-side (CPU) reimplementation of the `npu_sparse_attn_sharedkv_metadata` load-balancer.

The original op is an AICPU kernel that turns seq-lens and attention config into a fixed
`int32[1024]` core-distribution table (`SasMetaData`), consumed by the
`npu_sparse_attn_sharedkv` AICore kernel. The scheduler is pure integer logic (base-block
tiling → cost model → greedy core assignment), so it is rehosted on the server CPU. This
drops the CANN AICPU/dspark build dependency while producing **byte-identical** output to
the AICPU op.

The algorithm is ported from the AICPU kernel unchanged; only the I/O shim
(`CpuKernelContext`/`Tensor`/`GetAttrValue`) is replaced by plain pointers + `at::Tensor`.
Performance is improved by computing each row's cost state once in a parallel build pass
and reading it by reference thereafter (no redundant recompute, no copy).

## Interface Prototype

### Python Binding Definition

```python
import sgl_kernel_npu

torch.ops.npu.sparse_attn_sharedkv_metadata_host(
    num_heads_q: int,          # required, e.g. 64
    num_heads_kv: int,         # required, must be 1 (MQA)
    head_dim: int,             # required, e.g. 512
    layout_q: str,             # required, "TND"
    layout_kv: str,            # required, "PA_ND"
    cu_seqlens_q: Tensor|None, # optional, CPU int32 [B+1]; required for layout_q TND
    seqused_kv:  Tensor|None,  # optional, CPU int32 [B];   required for layout_kv PA_ND
    batch_size: int = 0,
    cmp_topk: int = 0,         # 0 / 512 / 1024 (only when has_cmp_kv)
    cmp_ratio: int = -1,       # 4 or 128 (only when has_cmp_kv)
    ori_mask_mode: int = 4,    # 0 (default) / 3 (right-down causal) / 4 (band)
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
) -> Tensor                    # device (NPU) int32 [1024]
```

Inputs are **CPU `int32` tensors**; the result is a **device (NPU) `int32[1024]`** tensor.
The op performs the 4 KB H2D internally, so it is functionally equivalent to the AICPU op
(which writes device memory directly) and needs no `.npu()` at the call site. No input D2H
is needed — sglang already keeps the seq-lens on the host. The device topology (cube/vector
core counts + SoC name) is queried inside the op via ACL and cached after the first call,
so callers never pass or know about it.

### Async H2D (stream-ordered result, zero host sync)

The internal 4 KB H2D is a **pinned + `non_blocking` copy enqueued on the caller's current
stream**. The host returns as soon as the copy is enqueued (~µs) and is never blocked by
in-flight stream work — the same async-enqueue contract as the AICPU op. Consequences for
callers:

- **Same-stream consumers need nothing**: any kernel enqueued after the op on the same
  stream (attention forward, `.copy_()` into graph-capture buffers) reads the completed
  table by stream order.
- **Cross-stream readers need an explicit event** (record on the producing stream, wait on
  the consuming stream) — exactly as they would for the AICPU op's output.
- The staging buffer is allocated with `pinned_memory(true)` through the caching host
  allocator, whose event tracking guarantees the block is not reused until the async copy
  has completed (validated by a 2,000-iteration allocate/copy/destroy stress loop with
  interleaved device load: zero corruption). A pageable staging buffer must not be
  substituted: with pageable memory torch_npu silently falls back to a host-synchronous
  copy that additionally drains all prior work on the current stream (measured: pinned
  ~0.4 ms vs pageable ~249 ms with ~250 ms of kernels in flight), which would serialize
  overlapped host-side metadata preparation with the running forward.

### Usage

```python
import torch
import sgl_kernel_npu

# cu_seqlens_q / seqused_kv built from sglang's CPU seq-lens (int32, on CPU)
metadata = torch.ops.npu.sparse_attn_sharedkv_metadata_host(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="TND", layout_kv="PA_ND",
    cu_seqlens_q=cu_seqlens_q_cpu, seqused_kv=seqused_kv_cpu,
    batch_size=B, cmp_topk=cmp_topk, cmp_ratio=cmp_ratio,
    ori_mask_mode=4, cmp_mask_mode=3,
    ori_win_left=127, ori_win_right=0,
    has_ori_kv=True, has_cmp_kv=has_cmp_kv,
)                                             # device tensor; 4 KB H2D enqueued async on the current stream
```

## Output Layout

The `int32[1024]` buffer is a flattened `SasMetaData`:

- `faMetadata[36][8]` — per AIC (cube) core: `[core_enable, bn2_start, m_start, s2_start,
  bn2_end, m_end, s2_end, first_fd_data_workspace_idx]`. Only `[0, aic_core_num)` entries are
  filled; the rest are zero. `core_enable == 0` marks inactive cores.
- `fdMetadata[72][8]` — per AIV (vector) core: the FlashDecode reduction plan (inert while
  `supportFd` is disabled, as in the upstream AICPU build).

The 36 / 72 dimensions are fixed buffer capacity (the consumer indexes with these strides);
the runtime `aic_core_num` / `aiv_core_num` only bound how many entries are enabled.

## Build

No CANN AICPU toolchain is needed — the op is plain host C++ compiled into `libsgl_kernel_npu.so`.
OpenMP is enabled (`-fopenmp`) for the parallel per-row build. The OpenMP team is **capped at 16
threads** (`ROW_PARALLEL_MAX_THREADS`); on high-core-count hosts the default `nproc` team makes the
fork/join overhead dominate and must not be used.

## Validation

Output is verified byte-identical to the AICPU op both against the original algorithm source
(compiled unchanged against a framework shim) and against the live AICPU op, across c1a / c4a /
c128a paths, batch sizes 1–32, and context lengths up to 100k.

Async behavior is validated on top: same-stream consumers read the completed table after the
non-blocking copy; a 2,000-iteration allocate→async-copy→destroy stress loop (pinned staging
dropped immediately, heavy device load interleaved) shows zero corruption, confirming the
caching-host-allocator event protection; and host wall-time of the op stays ~µs while ~250 ms
of kernels are in flight on the current stream (pageable staging, by contrast, blocks the host
for the full drain).

### Performance (time-to-device-ready, vs the live AICPU op)

| scenario | AICPU | host | speedup |
| --- | --- | --- | --- |
| decode B256 | ~300 µs | ~108 µs | ~2.6× |
| prefill T8k | ~358 µs | ~310 µs | ~1.15× |
| prefill T32k | ~889 µs | ~852 µs | ~1.04× |
| c4a prefill T8k | ~969 µs | ~339 µs | ~2.86× |

The host op wins or ties across decode and all prefill sizes, and is amortized once per batch
across all attention layers.
