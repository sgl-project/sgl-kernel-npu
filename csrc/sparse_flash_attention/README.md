# Sparse Flash Attention

`npu_sparse_flash_attention` is the sgl-kernel-npu port of the enhanced vLLM
Ascend sparse-flash-attention operator. It includes the PyTorch adapter, custom
ACLNN host implementation and tiling, and AscendC kernels. It supports the DSA
inputs used by MLA (`TND` query and `PA_BSND` paged KV), including the softmax
max/sum outputs consumed by DCP output merging.

The operator is registered as
`torch.ops.sgl_kernel_npu.npu_sparse_flash_attention`, rather than
`torch.ops.npu.npu_sparse_flash_attention`: torch_npu already owns the latter
schema and PyTorch disallows replacing its PrivateUse1 implementation from an
extension.

`build.sh -a kernels <SOC_VERSION>` builds and packages a private
`vendors/sgl_kernel_npu` OPP. Importing `sgl_kernel_npu` prepends that vendor to
`ASCEND_CUSTOM_OPP_PATH` before loading the PyTorch extension, so the adapter
resolves this package's `aclnnSparseFlashAttention` instead of the system CANN
implementation.

```python
import torch
import sgl_kernel_npu  # Loads libsgl_kernel_npu.so and registers the op.

attention_out, softmax_max, softmax_sum = (
    torch.ops.sgl_kernel_npu.npu_sparse_flash_attention(
        query, key, value, sparse_indices, scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
)
softmax_lse = softmax_max + torch.log(softmax_sum)
```

The defaults match the vLLM enhanced adapter, including `attention_mode=2`.
All optional tensor arguments remain `None` when absent so ACLNN receives true
optional inputs rather than placeholder tensors.

## NPU Graph

The adapter invokes the packaged custom `aclnnSparseFlashAttention` through the
vLLM `EXEC_NPU_CMD` path, without redispatching through the conflicting
`torch.ops.npu` schema. It owns the enhanced PA_BSND softmax-output allocation
needed for DCP and uses torch_npu's `OpCommand` plus its NPU caching allocator
for graph-safe execution. Inputs and attributes must remain shape-stable during
a graph replay, as required by ACLNN dynamic-shape operators generally.

Unlike the CANN 9.0 built-in operator, the packaged tiling and kernel support
`return_softmax_lse=True` with `layout_kv="PA_BSND"` in eager execution and NPU
Graph capture/replay.
