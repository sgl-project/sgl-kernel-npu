# Sparse Flash Attention

`sgl_sparse_flash_attention` is the sgl-kernel-npu port of the enhanced vLLM
Ascend sparse-flash-attention operator. It uses the repository's native host
tiling and AscendC direct-launch integration. It supports the DSA
inputs used by MLA (`TND` query and `PA_BSND` paged KV), including the softmax
max/sum outputs consumed by DCP output merging.

The operator is registered as `torch.ops.npu.sgl_sparse_flash_attention`.
The `sgl_` prefix avoids the existing torch_npu
`torch.ops.npu.npu_sparse_flash_attention` schema while keeping the repository's
standard `npu` namespace.

```python
import torch
import sgl_kernel_npu  # Loads libsgl_kernel_npu.so and registers the op.

attention_out, softmax_max, softmax_sum = (
    torch.ops.npu.sgl_sparse_flash_attention(
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

## NPU Graph

The host implementation constructs the repository's tiling context, caches the
serialized tiling tensor, allocates workspace, and launches the AscendC kernel
through `EXEC_KERNEL_CMD`. Inputs and attributes must remain shape-stable during
a graph replay. Run one eager warmup with the same tensor shapes, dtypes,
optional inputs, and attributes before graph capture so the tiling configuration
is present in the cache.

The native tiling and kernel support `return_softmax_lse=True` with
`layout_kv="PA_BSND"`.
