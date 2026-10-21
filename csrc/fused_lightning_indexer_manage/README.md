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
array. States -3, -2 and -1 select non-offload, first-fill/reset and steady
paths. Miss source/slot arrays have shape `[B, 32768]`; only the prefix given
by `miss_counts` is valid. Capacity is a multiple of 128 in [2048, 32640].
Source capacity is the block-table width times 128 and is at most 2**21.
These device-resident metadata values must satisfy the upstream protocol.

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
CANN 9.1, torch 2.10.0 and torch_npu 2.10.0.post6. All 10 native tests pass.
The SGLang production host allocator path also passes graph replay, request
reuse, padding, new-visible-page TopK and bitwise selected-KV checks. Attention
outputs match the same-policy reference at rtol=0.01/atol=0.002 on that fixed
fixture. The loaded normal library SHA256 is
`ff93127d1294ef50b848816ba4710d44ef950ad54e4dc6498e88f4805be827a8`;
the adapted device entry SHA256 is
`3bb654ecfb9278cd56ecc1bc2c2fc74dede0c287ced181f3749cb0f216eb8778`.
No external LIM probe or custom vendor library is loaded by those tests.

Checkpoint-backed model accuracy, latency and HBM require separate A/B runs;
these operator checks do not establish those benefits. Earlier 151/217 model
measurements used the external aclnn probe and must not be attributed to this
new launch path. Copy-SFA fusion and the previously observed long-generation
numerical divergence are outside this migration.

## Provenance

The device implementation comes from vllm-project/vllm-ascend commit
`3326f350c0de30a5170b86ecfce851322e3216e6` (PR #16640). Eight headers under
`op_kernel/upstream/` remain byte-identical; their hashes and entry adaptation
are recorded in `upstream.json`. The new host path replaces the aclnn adapter,
generated operator package, and generated tiling registration. Upstream
copyright notices are retained; Apache-2.0 and CANN-2.0 texts are in `licenses/`.
