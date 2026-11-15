# sgemmv_expand

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.lora import sgemmv_expand

def sgemmv_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y, *, impl="auto"): ...
```

`impl="auto"` selects `ascendc`. Select `torch_native` explicitly to use PyTorch.

## Description

LoRA approximates a weight update with a low-rank factorization. For base weight
$W$, adapter $i$ learns $\Delta W_i = B_i A_i$, so the layer output is:

$$ y_t = x_t W^{\top} + \frac{\alpha_i}{r_i} x_t A_i^{\top} B_i^{\top} $$

`sgemmv_expand` performs the second product. Its `x` is the low-rank intermediate
already produced by `sgemmv_shrink`
($h_t = \frac{\alpha_i}{r_i} x_t A_i^{\top}$, with the scaling folded in), and
`weight` is $B_i$. The operator computes $h_t B_i^{\top}$ and accumulates it in
place into the base output `y`; `y` may already contain the base GEMM result, or
be zero when only the LoRA branch is needed.

For multi-LoRA serving, every `seq_len[b]` consecutive tokens form one sequence
block that shares one adapter: `lora_indices[b]` selects it and `lora_ranks[i]`
gives its rank. A block with `lora_indices[b] < 0`, or an adapter with rank 0,
contributes no update and leaves those tokens unchanged.

A layer output may concatenate several sub-projections (for example fused QKV or
gate/up). `slice_offsets` splits it into `slice_count` output slices. For an
adapter of rank `r`, slice `s` reads the `s`-th rank-sized segment of `x` and
writes the corresponding output columns:

$$ y[t,\ o_s:o_{s+1}] \mathrel{+}= x[t,\ s r:(s+1)r] \, B_i[o_s:o_{s+1},\ :r]^{\top} $$

where `o_s = slice_offsets[s]`; columns of `weight` at or beyond `r` are never
read. Accumulation is performed in float32, and `y` is rounded once when written.
The operator mutates `y` in place and returns a Tensor aliasing the same storage;
`x`, `weight` and the four index tensors are unchanged.

## Parameters and return value

All tensors must reside on the same NPU. `slice_count` is
`slice_offsets.numel() - 1` and `max_rank` is `x.size(1) // slice_count`.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, slice_count * max_rank]` | `float32` | Required | Contiguous. `tokens > 0`. |
| `weight` | `torch.Tensor` | `[num_loras, hidden_out, max_rank]` or `[num_loras, 1, hidden_out, max_rank]` | `float16 / bfloat16` | Required | Contiguous. Must have the same dtype as `y`. |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Adapter index per sequence block, either `-1` or in `[0, num_loras)`. Indexed per block, not per token. |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Entry `b` is the number of tokens in block `b`. |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | Required | Adapter rank. Only `0`, `8`, `16`, `32` and `64` are supported. |
| `slice_offsets` | `torch.Tensor` | `[slice_count + 1]` | `int32` | Required | At least two entries. Must be increasing, with disjoint slices and a last entry not exceeding `y.size(1)`. |
| `y` | `torch.Tensor` | `[tokens, hidden_out]` | `float16 / bfloat16` | Required | Contiguous. Accumulated in place. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. `auto`, `ascendc`, `torch_native` |

| Return | Type | Shape and dtype | Description |
| --- | --- | --- | --- |
| `sgemmv_expand` | `torch.Tensor` | Same as `y` | Shares storage with `y`, which is updated in place. |

## ascendc backend

The values of `lora_indices`, `seq_len`, `lora_ranks` and `slice_offsets` are
read on device, so callers must keep them valid. For float16/bfloat16 `y`, every
`slice_offsets` entry must be a multiple of 16 elements to satisfy the underlying
`DataCopy` alignment.

## torch_native backend

Uses the same computation, in-place write and return alias as `ascendc`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmv_expand

tokens, slices, max_rank, hidden_out = 4, 1, 16, 4096
x = torch.randn(tokens, slices * max_rank, device="npu", dtype=torch.float32)
weight = torch.randn(2, hidden_out, max_rank, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
slice_offsets = torch.tensor([0, hidden_out], device="npu", dtype=torch.int32)
y = torch.zeros(tokens, hidden_out, device="npu", dtype=torch.float16)

expected = sgemmv_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y.clone(),
    impl="torch_native",
)
result = sgemmv_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
)  # auto -> ascendc; in place
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
```
