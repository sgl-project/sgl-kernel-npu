# Fused Lightning Indexer and cache management (LIM)

The ARCH22 wheel exposes LIM through a Torch adapter and bundles its CANN
operator package. Its Python entry is:

```python
from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage
```

The registered torch operation is `torch.ops.npu.fused_lightning_indexer_manage`.
It retains the 17-tensor ABI and seven in-place output/state aliases. The
operation computes index TopK, resident-cache state and miss plans. Main KV
copies and sparse attention remain separate operations.

## Build and install

Build from the repository root in an initialized Linux Ascend environment:

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 setup.py --arch-version 2201 --ops fused_lightning_indexer_manage --without-third-party
python3 -m pip install --no-deps output/sgl_kernel_npu-*.whl
```

The native operator targets Ascend 910B and 910_93. The wheel includes its
ACLNN API, tiling library and AI Core kernels. The adapter exposes
`lim_full_visible_max_query_width` from the same compiled limits as the kernel.

## Contract

The native ABI supports FP16/BF16 index query/key/weights, index head counts
32/64, head dimension 128, page size 128 and TopK 2048. Metadata and outputs
are int32. TopK source and destination tables are `[T,2048]`; the query retains
its `[T,H,128]` head dimension. Inputs and outputs are contiguous
and on the same NPU. Per request, native queries occupy one contiguous range
of 1..14 rows. `torch.ops.npu.lim_full_visible_max_query_width()` returns the
compiled limit so the framework can size its buffers from the actual MTP config.

`cache_slots_pool` is persistent encoded state, not an ordinary slot-index
array. `request_state` values -3, -2 and -1 select non-offload, first-fill/reset
and steady paths. The -1 path assumes existing resident data; it does not
cold-fill an empty main-KV resident buffer. SGLang enters the sparse route
from a short request with -2, which the graph-transition tests exercise.
Miss source/slot arrays have shape `[B, 32768]`; only the prefix given
by `miss_counts` is valid. Capacity is a multiple of 128 in [2048, 32640].
Source capacity is the block-table width times 128 and is at most 2**21.
These device-resident metadata values must satisfy the upstream protocol.

The Copy-SFA integration additionally accepts the full visible sequence as
the candidate length for Q1..Q14 with K=2048 and C>=Q*2048. This includes a
partial final page. Combined Copy-SFA accepts page-aligned C in [2048,32640]
while keeping attention TopK=2048. Wider unions use a bounded eviction scan;
Q4 retains its optimized scan. Rewritten-resident refresh and causal score
masking use the actual query width.
Short/nonresident rows still use state -3. A new resident row must use -2.
The returned plans can be passed directly to combined-layout Copy-SFA after
retaining the valid miss prefix in `[B,32768]` buffers. A cold fill may
contain C misses, including residents outside the K selected tokens. No second
residency planner may operate on the LIM slot map.

The Torch adapter checks tensor shapes and dtypes before calling ACLNN.
CANN tiling selects the core schedule and reports the workspace size for
that launch. The adapter keeps the tensors and their descriptors alive on
the current stream. The legacy wheel uses the same device computation
through its direct-launch entry.

## Tests

```bash
python -m pytest tests/python/sgl_kernel_npu/test_fused_lightning_indexer_manage.py
```

The tests cover TopK/slot consistency, cold fill, steady misses, request reuse,
short sequences, rewritten residents and graph replay on an available Ascend NPU.

## Provenance

The device implementation comes from vllm-project/vllm-ascend commit
`3326f350c0de30a5170b86ecfce851322e3216e6` (PR #16640). Four headers under
`op_kernel/upstream/` remain byte-identical. Four adapted headers add
full-visible causal score masking, rewritten-resident refresh, and bounded
eviction with explicit vector-tail padding. Their upstream paths, original
Git blob IDs and SHA256 hashes, and current LF-normalized hashes are recorded
in `upstream.json`. The original entry hash is also verified against that
fixed upstream commit; its launch adaptations are listed in the same file.
The new host path replaces the aclnn adapter,
generated operator package, and generated tiling registration. The imported files retain their copyright notices. See the
[upstream Apache-2.0 license](https://github.com/vllm-project/vllm-ascend/blob/3326f350c0de30a5170b86ecfce851322e3216e6/LICENSE)
and the [CANN Open Software License Agreement 2.0](https://github.com/hicann/ops-transformer/blob/91de0d9959ab57be7fcbef8b0592d16c242c5955/LICENSE)
named in the Lightning Indexer headers.
