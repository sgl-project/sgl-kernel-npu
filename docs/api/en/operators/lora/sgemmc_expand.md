# sgemmc_expand

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.lora import sgemmc_expand

def sgemmc_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y,
                  *, impl="auto"): ...
```

`impl="auto"` selects `ascendc`. Select `torch_native` explicitly to use PyTorch.

## Description

The cube-based variant of `sgemmv_expand`: it computes the second LoRA product
$h_t B_i^{\top}$ and accumulates it into `y`, but both `x` and the output keep the
activation dtype instead of using a float32 intermediate. For an adapter $i$ of
rank $r$ and slice $s$ spanning output columns $[o_s, o_{s+1})$:

$$ y[t,\ o_s:o_{s+1}] \mathrel{+}= x[t,\ s r:(s+1)r] \, B_i[o_s:o_{s+1},\ :r]^{\top} $$

where $o_s$ = `slice_offsets[s]`. A block with `lora_indices[b] < 0`, or an
adapter with rank 0, contributes nothing and leaves those rows unchanged. Column
ranges outside `[slice_offsets[0], slice_offsets[-1])` are never written.

Each (token, slice) product is computed by the cube unit in float32 and added to
the existing `y` value in float32; `y` is rounded once when stored. The operator
mutates `y` in place and returns a Tensor aliasing the same storage.

## Parameters and return value

All tensors must reside on the same NPU. `slice_count` is
`slice_offsets.numel() - 1` and `max_rank` is `x.size(1) // slice_count`.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, slice_count * max_rank]` | `float16 / bfloat16` | Required | Contiguous. `tokens > 0`. |
| `weight` | `torch.Tensor` | `[num_loras, hidden_out, max_rank]` or `[num_loras, 1, hidden_out, max_rank]` | same as `y` | Required | Contiguous. Columns at or beyond an adapter's rank are never read. |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Adapter index per sequence block, either `-1` or in `[0, num_loras)`. |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | Required | Entry `b` is the number of tokens in block `b`. |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | Required | Adapter rank. `16`, `32` and `64` are exercised; rank 0 skips the block. |
| `slice_offsets` | `torch.Tensor` | `[slice_count + 1]` | `int32` | Required | At least two entries. Must be increasing; the last entry may be smaller than `y.size(1)`. |
| `y` | `torch.Tensor` | `[tokens, hidden_out]` | `float16 / bfloat16` | Required | Contiguous. Accumulated in place. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. `auto`, `ascendc`, `torch_native` |

| Return | Type | Shape and dtype | Description |
| --- | --- | --- | --- |
| `sgemmc_expand` | `torch.Tensor` | Same as `y` | Shares storage with `y`, which is updated in place. |

## ascendc backend

The values of `lora_indices`, `seq_len`, `lora_ranks` and `slice_offsets` are read
on device, so callers must keep them valid. Every slice width and `max_rank`
should be a multiple of 16 elements so the two-byte `DataCopy` stays 32-byte
aligned. The Kernel copies results in `baseN`-sized steps chosen by
`MatmulApiTiling`, so slice widths should also be powers of two (or any multiple
of `baseN`) to keep the last step inside the slice. The migrated Kernel feeds the
rank straight into the cube matmul without the rank-8 padding used by
`sgemmv_expand`, so ranks below 16 are not supported.

## torch_native backend

Uses the same computation, in-place accumulation and return alias as `ascendc`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmc_expand

tokens, slices, max_rank, hidden_out = 4, 1, 16, 4096
x = torch.randn(tokens, slices * max_rank, device="npu", dtype=torch.float16)
weight = torch.randn(2, hidden_out, max_rank, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
slice_offsets = torch.tensor([0, hidden_out], device="npu", dtype=torch.int32)
y = torch.zeros(tokens, hidden_out, device="npu", dtype=torch.float16)

expected = sgemmc_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y.clone(),
    impl="torch_native",
)
result = sgemmc_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
)  # auto -> ascendc; in place
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
```
