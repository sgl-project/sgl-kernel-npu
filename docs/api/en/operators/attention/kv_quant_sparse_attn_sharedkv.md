# Quantized sparse shared-KV attention

```python
from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv,
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

metadata = kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="BSND", layout_kv="PA_ND",
    seqused_kv=cpu_int32_lengths, batch_size=1,
    max_seqlen_q=1, max_seqlen_kv=256,
    has_ori_kv=True, has_cmp_kv=False,
)
output, empty_lse = kv_quant_sparse_attn_sharedkv(
    q, ori_kv=packed_fp8_kv, ori_block_table=block_table,
    seqused_kv=npu_int32_lengths, sinks=sinks, metadata=metadata,
)
```

The public attention API declares the native operator's parameters explicitly
and adds a keyword-only `impl` selector. `auto` and `ascendc` call the
existing `torch.ops.npu.kv_quant_sparse_attn_sharedkv` binding on Ascend 950.
The independent `torch_native` implementation covers only BF16 BSND/PA_ND
uncompressed paged sliding-window attention for accuracy validation. It copies
inputs to CPU and is unsuitable for production or graph capture. Compressed,
sparse-index and TND paths remain covered by the existing native tests.

The host metadata API accepts CPU contiguous int32 sequence lengths and returns
the int32[1024] schedule on NPU. Its `auto` and `host` implementations call
`torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host`. The A5 DSpArk
SGLang path uses these public functions; low-level `torch.ops.npu` bindings
remain available for existing callers.

## Build status under the CANN refactor

PR #880's new `--ops` build registers CANN AICORE kernels with a CANN op host,
generated ACLNN interface and Torch adapter. This operator still uses the
legacy Torch host and launch path, and the metadata scheduler is CPU-only.
Neither can be selected through `--ops kv_quant_sparse_attn_sharedkv` yet.
Build the legacy Ascend 950 wheel to run this API until those native paths
are migrated. Do not remove the legacy target or its PyTorch schemas.
