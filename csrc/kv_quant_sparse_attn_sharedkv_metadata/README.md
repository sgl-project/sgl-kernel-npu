# Quantized shared-KV metadata

`torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host` builds the
`int32[1024]` task table consumed by the Ascend 950
`kv_quant_sparse_attn_sharedkv` kernel. Sequence-length inputs are contiguous
CPU `int32` vectors. The result is copied to the current NPU device.

The layout follows the vLLM Ascend A5 operator at commit `683b6c009d`:
36 cube-core slots of 9 integers, then 72 vector-core slots of 8 integers.
When the query has 128 heads, each logical cube-core assignment is written to
two adjacent physical slots; the ninth integer records the maximum S2 block
count needed for synchronization. This differs from the 8-integer cube-core
layout of `sparse_attn_sharedkv_metadata_host` and must not be mixed with it.

The host scheduler shares the existing core-assignment algorithm, with A5
block sizing and the quantized kernel's 128-head pairing. It supports BSND
with `max_seqlen_q` and TND with `cu_seqlens_q`, and uses `seqused_kv` or
`max_seqlen_kv` for PageAttention KV lengths. `cmp_topk` controls the CFA/SCFA
cost path. The source AICPU implementation does not enable cross-core Flash
Decode splitting, and neither does this host implementation.

On an Ascend 950 machine, build the `kernels` wheel and run:

```bash
python tests/python/sgl_kernel_npu/test_kv_quant_sparse_attn_sharedkv_metadata.py
```
