## Introduction

LoRA (Low-Rank Adaptation) batched operators for multi-adapter serving. Each operator evaluates a batched low-rank projection where every token (or batch row) selects its own adapter through an index tensor, and accumulates the result into an output buffer. The family is adapted from the vllm-project/vllm-ascend LoRA kernels.

Two projection directions are provided:

- **expand**: up-projection `y[b, off:off+size] += x[b, :rank] @ weight[idx[b], off:off+size, :rank]^T`, writing into a slice of `y` selected by `slice_offset` / `slice_size`.
- **shrink**: down-projection `y[b] += scale * x[b] @ weight[idx[b]]`, reducing back to the model hidden dimension.

Four selection schemes cover the common serving shapes:

| Scheme | Adapter selection | Slice / rank selection | Ops |
|--------|-------------------|------------------------|-----|
| bgmv | per-batch `indices` | fixed `slice_offset` / `slice_size` (expand) or implicit full slice (shrink) | `bgmv_expand`, `bgmv_shrink` |
| sgmv | per-sequence `lora_indices` + `seq_len` | fixed `slice_offset` / `slice_size` (expand) or implicit full slice (shrink) | `sgmv_expand`, `sgmv_shrink` |
| sgemmv | per-sequence `lora_indices` + `seq_len` | per-adapter `lora_ranks` + `sliceOffsets` | `sgemmv_expand`, `sgemmv_shrink` |
| sgemmc | per-sequence `lora_indices` + `seq_len` | per-adapter `lora_ranks` + `sliceOffsets` (migrated to the independent CANN operator framework) | `sgemmc_expand`, `sgemmc_shrink` |

All operators are registered under the `npu` namespace, e.g. `torch.ops.npu.bgmv_expand`. The `sgemmc_*` operators are the framework-migrated implementations of the `sgemmv_*` interface family and keep the same call signature.

## Sheet 1: Parameters

### bgmv_expand / bgmv_shrink

| Parameter | Dimension | Data Type | Format | Description |
|-----------|-----------|-----------|--------|-------------|
| x | [batch_size, hidden_in] | float16/bfloat16 | ND | Input activations. |
| weight | [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in] | float16/bfloat16 | ND | LoRA weights, one matrix per adapter. |
| indices | [batch_size] | int64 | ND | Adapter index per batch row. |
| y | [batch_size, hidden_out] | float16/bfloat16 | ND | Output buffer, mutated in place. |
| slice_offset / slice_size | scalar | int64 | / | expand only: output slice window inside `hidden_out`. |
| scale | scalar | float | / | shrink only: scaling applied to the projection. |

### sgmv_expand / sgmv_shrink

Same as `bgmv_*`, except the adapter selection is per sequence: `lora_indices` is `[num_seqs]` and `seq_len` is `[num_seqs]`, so `x` and `y` are laid out as concatenated sequences `[total_tokens, hidden]`.

### sgemmv_expand / sgemmv_shrink / sgemmc_expand / sgemmc_shrink

| Parameter | Dimension | Data Type | Format | Description |
|-----------|-----------|-----------|--------|-------------|
| x | [tokens, slices * max_rank] | float16/bfloat16 | ND | Input activations. |
| weight | [num_loras, hidden_out, max_rank] or [num_loras, 1, hidden_out, max_rank] | float16/bfloat16 | ND | LoRA weights. |
| lora_indices | [num_seqs] | int32 | ND | Adapter index per sequence. |
| seq_len | [num_seqs] | int32 | ND | Token count per sequence. |
| lora_ranks | [num_loras] | int32 | ND | Active rank per adapter. |
| sliceOffsets / slice_offsets | [num_loras] | int32 | ND | Output slice offset per adapter. |
| lora_scales | [num_loras] | float | ND | shrink only: scaling per adapter. |
| slice_count | scalar | int | / | shrink only, default 1: number of slices per adapter. |
| y | [tokens, hidden_out] | float16/bfloat16 | ND | Output buffer, mutated in place. |

## Restrictions

1. Only Ascend A2/A3 is supported.
2. `x`, `weight` and `y` must share one dtype, either float16 or bfloat16.
3. `x` must be 2-D `[batch_size, hidden_in]`; `y` must be 2-D `[batch_size, hidden_out]`.
4. `weight` must be 3-D `[num_loras, hidden_out, hidden_in]` or 4-D `[num_loras, 1, hidden_out, hidden_in]`.
5. The first dimension of `x`, `y` and `indices` must be the same.
6. For expand ops: `hidden_in <= slice_size`, `slice_offset >= 0`, and `slice_offset + slice_size <= hidden_out`.
7. For shrink ops: `hidden_in > hidden_out`.
8. For `sgemmv_*` / `sgemmc_*`: `lora_ranks` values must be 0, 8, 16, 32 or 64 (`sgemmc_*` additionally requires 0 or at least 16), and all index tensors must live on the same NPU as `y`.

## Sample Code

```python
import torch
import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401

dtype = torch.float16
batch_size = 8
hidden_in = 512
hidden_out = 1024
num_loras = 4

x = torch.randn(batch_size, hidden_in, dtype=dtype, device="npu")
weight = torch.randn(num_loras, hidden_out, hidden_in, dtype=dtype, device="npu")
indices = torch.randint(0, num_loras, (batch_size,), dtype=torch.int64, device="npu")
y = torch.zeros(batch_size, hidden_out, dtype=dtype, device="npu")

# expand: y[:, :hidden_in] += x @ weight[indices]
y = torch.ops.npu.bgmv_expand(x, weight, indices, y, 0, hidden_in)

# shrink: y += scale * x @ weight[indices]
x_down = torch.randn(batch_size, hidden_out, dtype=dtype, device="npu")
weight_down = torch.randn(num_loras, hidden_in, hidden_out, dtype=dtype, device="npu")
y_down = torch.zeros(batch_size, hidden_in, dtype=dtype, device="npu")
torch.ops.npu.bgmv_shrink(x_down, weight_down, indices, y_down, 1.0)
```

## Python API

The `sgemmv_*` and `sgemmc_*` operators are also exposed through the Python operator registry with explicit implementation selection:

```python
from sgl_kernel_npu.operators.lora import sgemmc_expand, sgemmc_shrink

# impl="auto" selects the Ascend C kernel without fallback;
# impl="torch_native" selects the reference implementation.
y = sgemmc_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y)
```
