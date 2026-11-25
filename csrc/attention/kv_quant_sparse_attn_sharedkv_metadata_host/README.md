# Quantized shared-KV metadata

`torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host` builds the
`int32[1024]` task table consumed by the Ascend 950
`kv_quant_sparse_attn_sharedkv` kernel. Sequence-length inputs are contiguous
CPU `int32` vectors. The result is copied to the current NPU device.
The operator also accepts SGLang's `kv_quant_mode=1`, `tile_size=64`, and
`rope_head_dim=64` keyword arguments. Other values are rejected because this
metadata implementation only supports that quantized A5 configuration.

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

## Reusing the existing CANN metadata in SGLang

SGLang already calls `torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata`
with NPU sequence-length tensors. The migrated A5 attention kernel retains the
source kernel's `int32[1024]` metadata layout, so reusing that CANN producer is
the preferred integration path when its output is compatible. It avoids a
per-step NPU-to-CPU length transfer or CPU scheduling plus a host-to-NPU copy.
The host operator above remains useful independently of that CANN dependency.
When using it with SGLang, the caller must supply its CPU sequence-length
mirrors. For NPU graph replay, generate fresh metadata before replay and copy
it into the stable NPU metadata tensor captured by the graph. The host metadata
operator itself is called outside graph capture.

External CANN metadata compatibility is not covered by the public API tests.
C4/C128 compressed attention still needs end-to-end validation before changing
SGLang operator selection in production.

On an Ascend 950 machine, build the `kernels` wheel and run:

```bash
python -m pytest tests/python/operators/attention/test_kv_quant_sparse_attn_sharedkv_metadata.py --npu-device=0 --enable-perf=false -v -s
```
