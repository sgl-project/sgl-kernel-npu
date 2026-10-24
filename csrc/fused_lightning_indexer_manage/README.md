# Fused Lightning Indexer and cache management (LIM)

LIM is compiled into the normal `libsgl_kernel_npu.so` and shipped by the
`sgl_kernel_npu` wheel. Its Python entry is:

```python
from sgl_kernel_npu.sparsity_driven_kv_offload import fused_lightning_indexer_manage
```

The registered torch operation is `torch.ops.npu.fused_lightning_indexer_manage`.
It retains the 19-tensor ABI and seven in-place output/state aliases. The
operation computes index TopK, resident-cache state and miss plans. Main KV
copies and sparse attention remain separate operations.

## Build and install

Use the repository's normal build in a Linux Ascend development environment:
initialize the normal project submodules as described in the root README.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a kernels Ascend910_9382
pip install --no-deps --force-reinstall output/sgl_kernel_npu*.whl
```

No vLLM checkout, custom LIM vendor installation, `libcust_opapi.so`, external
LIM library path or separate operator installer is needed. The normal project
requirements (CANN, driver, torch_npu and its build tools) still apply. LIM is
built with the other A2/A3-only operators and is absent from an A5 build.

## Contract

The native ABI supports FP16/BF16 index query/key/weights, index head counts
32/64, head dimension 128, page size 128 and TopK 2048. Metadata and outputs
are int32; scale placeholders are float32. Inputs and outputs are contiguous
and on the same NPU. Per request, native queries occupy one contiguous range
of 1..14 rows; the SGLang integration currently uses Q1 only.

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
the candidate length for Q1 with K=2048 and C>=K. This includes a partial final
page; other route counts retain the page-aligned prefix contract. Combined
Copy-SFA currently accepts C=2048,4096,8192 while keeping attention TopK=2048.
Short/nonresident rows still use state -3. A new resident row must use -2.
The returned plans can be passed directly to combined-layout Copy-SFA after
packing the valid miss prefix into contiguous `[B,C]` buffers. A cold fill may
contain C misses, including residents outside the K selected tokens. No second
residency planner may operate on the LIM slot map.

The host performs shape/dtype checks without reading device scalar values.
It chooses the same core schedule and workspace layout as the upstream MTP
tiler, allocates workspace through torch_npu and directly launches Ascend C.
The ten original uint32 tiling fields retain their layout; an appended dtype
selects the two compiled implementations instead of aclnn template dispatch.
Explicit zeroed reserved fields pad each serialized record to 64 bytes.
Tiling storage uses the shared per-device `TilingTensorCache`; warm up each
shape/dtype configuration eagerly before graph capture. Device metadata may
change during replay without a host read or retiling.

## Tests and benchmark

```bash
python -m pytest tests/python/sgl_kernel_npu/test_fused_lightning_indexer_manage.py
python benchmark/bench_fused_lightning_indexer_manage.py --length 32768 --capacity 4096
```

Tests reconstruct the resident source IDs from the returned miss plans and
verify TopK/slot consistency across cold fill, steady misses, request reuse,
short sequences and graph replay. The benchmark measures native launch time,
not model TPOT or H2D overlap.

Native validation on 217, 2026-10-21: the normal A3 kernel wheel builds with
CANN 9.1, torch 2.10.0 and torch_npu 2.10.0.post6. All 12 native tests pass
on 217, including graph entry from non-offload into cold reset at C=2048/4096.
The SGLang production host allocator path also passes graph replay, request
reuse, padding, new-visible-page TopK and bitwise selected-KV checks. Attention
outputs match the same-policy reference at rtol=0.01/atol=0.002 on that fixed
fixture. The loaded normal library SHA256 is
`ff93127d1294ef50b848816ba4710d44ef950ad54e4dc6498e88f4805be827a8`;
the adapted device entry SHA256 is
`3bb654ecfb9278cd56ecc1bc2c2fc74dede0c287ced181f3749cb0f216eb8778`.
No external LIM probe or custom vendor library is loaded by those tests.

Separate checkpoint-backed GLM-5.2-W4A8 A/B runs on 217 use this same normal
library and SGLang runtime 611760f1, with TP16/EP16 and BF16 KV. LIM C4096
TPOT is 118.510/127.061/123.482 ms for 8K->128 B1, 32K->512 B1 and 8K->128
B4; C2048 gives 123.423/135.411/134.488 ms. The full-HBM baseline is
47.976/48.273/49.727 ms. At 32K B1, median chip sampled peak HBM is
57031.0 MiB full HBM, 50390.5 MiB C4096 and 47337.5 MiB C2048. These are
whole-device observations in accepted timing windows, not exact watermarks.
Thus HBM is reduced, but the complete offload path remains much slower than
full HBM despite improving over the ordinary same-policy LI/manage reference.

Custom long-prompt quality is 30/32 full HBM, 28/32 same-policy reference,
30/32 LIM C4096 and 28/32 LIM C2048. Fixed-history, full-vocabulary KL uses
32 samples x 512 positions per path. Reference self-repeat mean KL is
0.00613559; reference -> LIM C4096/C2048 means are 0.00961504/0.01028223.
These results do not establish numerical equivalence or general quality
parity. A fresh decode trace records 126 native LIM calls and 468 five-copy/
SFA groups; all copy/SFA tasks use stream 187, with consumer overlap 0/468.
Copy-SFA fusion is not implemented by this operator.

Full results and reproduction tools are in the SGLang integration branch,
`benchmark/hisparse/LIM_NATIVE_217_RESULTS.txt` and `LIM_VALIDATION.txt`:
https://github.com/McZyWu/sgl-sglang/tree/feat/ascend-lim-kv-offload/benchmark/hisparse
Earlier external aclnn-probe measurements retain their own provenance and
must not be attributed to this direct launch path.

## Provenance

The device implementation comes from vllm-project/vllm-ascend commit
`3326f350c0de30a5170b86ecfce851322e3216e6` (PR #16640). Seven headers under
`op_kernel/upstream/` remain byte-identical. The union header has local
full-visible Q1 and eviction adaptations; its original and adapted hashes,
and the entry adaptations, are recorded in `upstream.json`.
The new host path replaces the aclnn adapter,
generated operator package, and generated tiling registration. Upstream
copyright notices are retained; Apache-2.0 and CANN-2.0 texts are in `licenses/`.
