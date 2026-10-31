# kpool_indexer_prepare_decode_input

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input

def kpool_indexer_prepare_decode_input(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
    *,
    impl="auto",
):
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Prepare three inputs for decode indexing: project hidden states into per-head weights, convert sequence lengths to complete-pool counts, and extract the pooled page table.

The projection is computed in FP32, scaled by `n_heads_pow` and `softmax_scale`, and returned as BF16. The page table is padded with zeros when needed before selecting the first entry of each page group. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, HIDDEN_SIZE]` | `torch.bfloat16` | Required | Decode hidden states. |
| `weights_proj_weight` | `torch.Tensor` | `[N_HEADS, HIDDEN_SIZE]` | `torch.float32` | Required | ReplicatedLinear projection weight. |
| `seqlens_32` | `torch.Tensor` | `[N]` | `torch.int32` | Required | Real KV length for each request. |
| `block_tables` | `torch.Tensor` | `[N, NUM_BLOCKS]` | `torch.int32` | Required | Contiguous token-page table. |
| `pool_size` | `int` | — | — | `4` | Pooling width; the current kernel requires exactly 4. |
| `n_heads_pow` | `float` | — | — | `32**-0.5` | First factor in the combined projection scale. |
| `softmax_scale` | `float` | — | — | `128**-0.5` | Second factor in the combined projection scale. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `w_bf16` | `torch.Tensor` | `[N, N_HEADS]` | `torch.bfloat16` | Scaled projection result. |
| `pool_seqlens` | `torch.Tensor` | `[N]` | `torch.int32` | Integer-divided pooled sequence lengths. |
| `pool_block_tables` | `torch.Tensor` | `[N, N_POOL]` | `torch.int32` | Newly allocated pooled page table. |

All tensors must be on the same NPU. Callers provide valid input values; the wrapper does not validate every value range.

## triton backend

Inputs must be contiguous, `N_HEADS` must be a power of two in [1, 128], and `pool_size` must be 4. Hidden width need not be divisible by 256.

`n_heads_pow` is used as supplied; its default remains `32**-0.5`. Pass `N_HEADS**-0.5` explicitly when using that normalization with a different head count.

## torch_native backend

Uses independent PyTorch FP32 matmul, padding, grouped selection, and BF16 conversion.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input
n, hidden, heads = (2, 256, 32)
x = torch.randn(n, hidden, device='npu', dtype=torch.bfloat16)
weight = torch.randn(heads, hidden, device='npu', dtype=torch.float32)
seqlens = torch.tensor([8, 12], device='npu', dtype=torch.int32)
block_tables = torch.arange(16, device='npu', dtype=torch.int32).reshape(n, 8)
w_bf16, pool_seqlens, pool_block_tables = kpool_indexer_prepare_decode_input(
    x,
    weight,
    seqlens,
    block_tables,
    impl='torch_native',
)
print(w_bf16.shape, pool_seqlens.shape, pool_block_tables.shape)
```
