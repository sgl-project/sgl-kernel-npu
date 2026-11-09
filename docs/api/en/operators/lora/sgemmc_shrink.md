# sgemmc_shrink

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.lora import sgemmc_shrink

def sgemmc_shrink(x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y,
                  slice_count, *, impl="auto"): ...
```

`impl="auto"` selects `ascendc`. Select `torch_native` explicitly to use PyTorch.

## Description

The cube-based variant of `sgemmv_shrink`: it computes
$x_t A_i^{\top}$ for each token but stores the result in the activation dtype
instead of float32. For an adapter $i$ of rank $r$ and slice $s$:

$$ y[t,\ s r:(s+1)r] = \operatorname{round}\!\left(\lambda_i \, x[t,\ :] \, A_i[s r:(s+1)r,\ :]^{\top}\right) $$

with $\lambda_i$ = `lora_scales[i]`. Each (token, slice) product is produced by
the cube unit in float32, scaled in float32 and rounded once when stored. A block
with `lora_indices[b] < 0`, or an adapter with rank 0, writes nothing.

The operator writes only the `slice_count * rank` covered columns of each token's
row into the caller's `y` and returns a Tensor aliasing the same storage. The
remaining columns keep their previous values.

## Parameters and return value

All tensors must reside on the same NPU. `max_rank` is
`y.size(1) // slice_count`.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, hidden_in]` | `float16 / bfloat16` | Required | Contiguous. `tokens > 0`, `hidden_in > y.size(1)`. |
| `weight` | `torch.Tensor` | `[num_loras, slice_count * max_rank, hidden_in]` or `[num_loras, 1, slice_count * max_rank, hidden_in]` | same as `x` | Required | Contiguous. Slice `s` of adapter `i` occupies rows `s*rank_i : (s+1)*rank_i`. |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Adapter index per sequence block, either `-1` or in `[0, num_loras)`. |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Entry `b` is the number of tokens in block `b`. |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | Required | Adapter rank. `16`, `32` and `64` are exercised; rank 0 skips the block. |
| `lora_scales` | `torch.Tensor` | `[num_loras]` | `float16` | Required | Scaling applied to every computed column. |
| `y` | `torch.Tensor` | `[tokens, slice_count * max_rank]` | same as `x` | Required | Contiguous. Written in place. |
| `slice_count` | `int` | — | — | Required | Number of fused sub-projections. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. `auto`, `ascendc`, `torch_native` |

| Return | Type | Shape and dtype | Description |
| --- | --- | --- | --- |
| `sgemmc_shrink` | `torch.Tensor` | Same as `y` | Shares storage with `y`, which is updated in place. |

## ascendc backend

The four index/scaling tensors are read on device, so callers must keep them
valid. `hidden_in` and every slice width should be a multiple of 16 elements to
keep the two-byte `DataCopy` 32-byte aligned. The migrated Kernel feeds the rank
straight into the cube matmul without rank-8 padding, so ranks below 16 are not
supported.

## torch_native backend

Uses the same computation, in-place write and return alias as `ascendc`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmc_shrink

tokens, slices, max_rank, hidden_in = 4, 1, 16, 2048
x = torch.randn(tokens, hidden_in, device="npu", dtype=torch.float16)
weight = torch.randn(2, slices * max_rank, hidden_in, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
lora_scales = torch.tensor([1.0, 0.5], device="npu", dtype=torch.float16)
y = torch.zeros(tokens, slices * max_rank, device="npu", dtype=torch.float16)

expected = sgemmc_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y.clone(), slices,
    impl="torch_native",
)
result = sgemmc_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y, slices
)  # auto -> ascendc; in place
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
```
