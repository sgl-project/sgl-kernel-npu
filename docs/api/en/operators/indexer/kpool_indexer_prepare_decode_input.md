# kpool_indexer_prepare_decode_input

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

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

Let `B=N`, `H=HIDDEN_SIZE`, `A=N_HEADS`, and `K=pool_size=4`:

$$w=\operatorname{BF16}((\operatorname{FP32}(x)W^T)\cdot n_{heads}\cdot s),\qquad \ell_{pool}=\lfloor\ell/K\rfloor.$$

Here `n_heads=n_heads_pow` and `s=softmax_scale`. Zero-pad the table to `M_pad=max(K,ceil(NUM_BLOCKS/K)*K)`, then select columns `0,K,2K,...`. Thus `N_POOL=max(1,ceil(NUM_BLOCKS/K))`, including one output column for an empty input table. Results form a tuple in return-table order; inputs are read only.

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

Uses separate Cube projection and Vector metadata kernels. Inputs must be contiguous, `N_HEADS` must be a power of two in [1, 128], and `pool_size==4`. GEMM tiling depends only on the head count:

```text
BLOCK_B = clamp(floor_power_of_two(2048 // N_HEADS), 16, 64)
BLOCK_K = min(256, floor_power_of_two(24576 // (BLOCK_B + N_HEADS)))
```

The logical element budgets are 2048 for the accumulator and 24576 for the two operand tiles. For 16/32/64/128 heads, `(BLOCK_B, BLOCK_K)` is `(64,256)/(64,256)/(32,256)/(16,128)`. Partial batch and K tiles are zero-padded when loaded; only valid batch rows are written. H need not be divisible by BLOCK_K. Vector tile sizes remain 128 for sequence lengths and 4096 for page-table input elements.

`n_heads_pow` is used as supplied; its default remains `32**-0.5`. Pass `N_HEADS**-0.5` explicitly when using that normalization with a different head count.

## torch_native backend

Uses independent PyTorch FP32 matmul, padding, grouped selection, and BF16 conversion without using Triton tiling constraints.

Importing the public registry requires NPU architecture detection, including for `torch_native`. Support is limited to accepted A3 cases; architecture registration does not establish coverage of every chip, shape or dtype.

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
