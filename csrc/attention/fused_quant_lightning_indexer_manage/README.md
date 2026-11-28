# Fully visible A5 fused quant Lightning Indexer and cache manager

`fused_quant_lightning_indexer_manage_full_visible` runs FP8
index scoring, per-query causal TopK, and shared request residency management
in one Ascend950 launch. Its arch35 device code is adapted from
[vllm-ascend #18383](https://github.com/vllm-project/vllm-ascend/pull/18383)
at `7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e`. The selective CANN entry uses
batch scheduling and the same 216 KiB SIMT local-memory layout as the donor.
[`UPSTREAM_PROVENANCE.json`](UPSTREAM_PROVENANCE.json) records the donor
paths and hashes at that revision.

For the public Python API and usage example, see the
[operator documentation](../../../docs/api/en/operators/indexer/fused_quant_lightning_indexer_manage_full_visible.md).
Each request has a cumulative query end, a final causal KV length, a source
block table, a cache size and a resident row ID. The kernel uses the same
source range for scoring and residency management. Each query clips its
candidate range to its own causal length. The cache size is page-aligned
and covers the maximum union of Q 脳 2048 candidates. All sparse destination
slots are in `[0, C)`, including selected current tokens.

- State `-3` emits the valid TopK and identity slots with zero misses.
- State `-2` fills C slots with the TopK union plus available fillers and
  overwrites the source map for the request.
- State `-1` reuses resident hits across the Q routes and emits one
  deduplicated miss list with replacement slots for the whole request.
  Retained slots in the current Q window are appended as refresh copies.
  The request count includes these refreshes. Each query's miss count still
  describes only its TopK misses.

The source map is a contiguous `int32` tensor with width `pages * 128`.
Row zero is reserved for graph padding and has a physical C8 payload row.
The final padding row is a metadata sentinel. Valid sparse rows start in state
`-2` before entering state `-1`. The caller holds the source map through capture
and replay and applies the output plan with C8 CopySFA.

Build and test the Ascend950 operator independently.

```bash
python3 setup.py --arch-version 3510 --ops fused_quant_lightning_indexer_manage_full_visible --without-third-party
python3 -m pytest -q tests/python/operators/indexer/test_fused_quant_lightning_indexer_manage_full_visible.py
```
