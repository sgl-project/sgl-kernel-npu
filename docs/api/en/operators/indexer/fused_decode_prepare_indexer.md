# fused_decode_prepare_indexer

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

def fused_decode_prepare_indexer(
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

## Description

Fuses decode indexer input preparation into a Cube GEMM kernel and a Vector kernel:

- `w_bf16 = (x @ weights_proj_weight.T) * (n_heads_pow * softmax_scale)`, converted to BF16.
- `pool_seqlens = seqlens_32 // pool_size`.
- `pool_block_tables` selects every `pool_size`-th page-table column, equivalent to pooling with `block_tables[:, ::pool_size]` after the backend's internal padding.

`actual_seq_lengths_q = arange(1, N + 1)` is not produced by this operator and remains the caller's responsibility.

## Parameters

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, HIDDEN_SIZE]` | `torch.bfloat16` | Required | Decode hidden states. |
| `weights_proj_weight` | `torch.Tensor` | `[N_HEADS, HIDDEN_SIZE]` | `torch.float32` | Required | ReplicatedLinear projection weight. |
| `seqlens_32` | `torch.Tensor` | `[N]` | `torch.int32` | Required | Real KV length for each request. |
| `block_tables` | `torch.Tensor` | `[N, NUM_BLOCKS]` | `torch.int32` | Required | Contiguous token-page table. |
| `pool_size` | `int` | — | — | `4` | Pooling width; the current kernel requires exactly 4. |
| `n_heads_pow` | `float` | — | — | `32**-0.5` | First factor in the combined projection scale. |
| `softmax_scale` | `float` | — | — | `128**-0.5` | Second factor in the combined projection scale. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

## Return values

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `w_bf16` | `torch.Tensor` | `[N, N_HEADS]` | `torch.bfloat16` | Scaled projection result. |
| `pool_seqlens` | `torch.Tensor` | `[N]` | `torch.int32` | Integer-divided pooled sequence lengths. |
| `pool_block_tables` | `torch.Tensor` | `[N, N_POOL]` | `torch.int32` | Newly allocated pooled page table. |

Let `PADDED_BLOCKS = max(pool_size, ceil(NUM_BLOCKS / pool_size) * pool_size)`; then `N_POOL = PADDED_BLOCKS // pool_size`. The backend pads a narrow or unaligned table internally before extraction. Entries beyond each request's `pool_seqlens` are not intended for downstream consumption. Inputs are not modified.

## triton backend

- Requires `x.shape[1] == weights_proj_weight.shape[1]`, `HIDDEN_SIZE % 256 == 0`, `N_HEADS <= 32`, and `pool_size == 4`. The GEMM and vector layouts assume compatible contiguous inputs; `block_tables` is explicitly documented as contiguous by the backend.
- `auto` aliases `triton`, without dynamic selection or fallback. Registry import still requires NPU/ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

n, hidden, heads = 2, 256, 32
x = torch.randn(n, hidden, device="npu", dtype=torch.bfloat16)
weight = torch.randn(heads, hidden, device="npu", dtype=torch.float32)
seqlens = torch.tensor([8, 12], device="npu", dtype=torch.int32)
block_tables = torch.arange(16, device="npu", dtype=torch.int32).reshape(n, 8)

w_bf16, pool_seqlens, pool_block_tables = fused_decode_prepare_indexer(
    x, weight, seqlens, block_tables,
)
print(w_bf16.shape, pool_seqlens.shape, pool_block_tables.shape)
# torch.Size([2, 32]) torch.Size([2]) torch.Size([2, 2])
```
