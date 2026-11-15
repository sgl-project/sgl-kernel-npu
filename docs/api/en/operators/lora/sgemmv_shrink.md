# sgemmv_shrink

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.lora import sgemmv_shrink

def sgemmv_shrink(x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y,
                  slice_count=1, *, impl="auto"): ...
```

`impl="auto"` selects `ascendc`. Select `torch_native` explicitly to use PyTorch.

## Description

LoRA approximates a weight update with a low-rank factorization. For base weight
$W$, adapter $i$ learns $\Delta W_i = B_i A_i$, so the layer output is:

$$ y_t = x_t W^{\top} + \frac{\alpha_i}{r_i} x_t A_i^{\top} B_i^{\top} $$

`sgemmv_shrink` performs the first LoRA product. `x` is the layer input, `weight`
is $A_i$ and `lora_scales[i]` is the folded scaling $\alpha_i / r_i$. For every
token $t$ of a sequence block whose adapter $i$ has rank $r$, slice $s$ computes:

$$ h[t,\ s r:(s+1)r] = \lambda_i \, x[t,\ :] \, A_i[s r:(s+1)r,\ :]^{\top} $$

where $\lambda_i$ = `lora_scales[i]` and the per-adapter weight block stores its
slices back to back at `weight[i, s*r:(s+1)*r, :]`. A block with
`lora_indices[b] < 0`, or an adapter with rank 0, writes nothing and leaves those
rows unchanged. All dot products and the scaling stay in float32; `y` is float32,
so no rounding is involved.

The operator writes only the `slices * rank` covered columns of each token's row
into the caller's `y` and returns a Tensor aliasing the same storage. Remaining
columns of `y` keep their previous values. `x`, `weight` and the index tensors
are unchanged. This is the intermediate consumed by `sgemmv_expand`.

## Parameters and return value

All tensors must reside on the same NPU. `max_rank` is
`y.size(1) // slice_count`.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, hidden_in]` | `float16 / bfloat16` | Required | Contiguous. `tokens > 0`, `hidden_in > y.size(1)`. |
| `weight` | `torch.Tensor` | `[num_loras, slices * max_rank, hidden_in]` or `[num_loras, 1, slices * max_rank, hidden_in]` | same as `x` | Required | Contiguous. Slice `s` of adapter `i` occupies rows `s*rank_i : (s+1)*rank_i`. |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Adapter index per sequence block, either `-1` or in `[0, num_loras)`. |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Entry `b` is the number of tokens in block `b`. |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | Required | Adapter rank. Only `0`, `8`, `16`, `32` and `64` are supported. |
| `lora_scales` | `torch.Tensor` | `[num_loras]` | `float16` | Required | Scaling applied to every computed column. |
| `y` | `torch.Tensor` | `[tokens, slices * max_rank]` | `float32` | Required | Contiguous. Written in place. |
| `slice_count` | `int` | — | — | `1` | Number of fused sub-projections (1, 2 or 3 in SGLang). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. `auto`, `ascendc`, `torch_native` |

| Return | Type | Shape and dtype | Description |
| --- | --- | --- | --- |
| `sgemmv_shrink` | `torch.Tensor` | Same as `y` | Shares storage with `y`, which is updated in place. |

## ascendc backend

`lora_indices`, `seq_len`, `lora_ranks` and `lora_scales` are read on device, so
callers must keep them valid. Each `CopyOut` moves `rank` float32 values, so the
supported ranks (8, 16, 32, 64) satisfy the 32-byte `DataCopy` alignment;
`hidden_in` should be a multiple of 64 elements, which satisfies both the
half/bfloat16 input copy alignment and the 64-element float32 `ReduceSum` inside
each tile. The Kernel walks `hidden_in` in 11776-element tiles and handles a final
partial tile.

## torch_native backend

Uses the same computation, in-place write and return alias as `ascendc`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmv_shrink

tokens, slices, max_rank, hidden_in = 4, 1, 16, 2048
x = torch.randn(tokens, hidden_in, device="npu", dtype=torch.float16)
weight = torch.randn(2, slices * max_rank, hidden_in, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
lora_scales = torch.tensor([1.0, 0.5], device="npu", dtype=torch.float16)
y = torch.zeros(tokens, slices * max_rank, device="npu", dtype=torch.float32)

expected = sgemmv_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y.clone(),
    impl="torch_native",
)
result = sgemmv_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y
)  # auto -> ascendc; in place
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
```
