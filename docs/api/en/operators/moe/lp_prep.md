# lp_prep

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.moe import lp_prep


def lp_prep(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
    *,
    impl="auto"
): ...
```

## Description

Builds the interior-point inputs of the LPLB (load-balanced placement) solver. The token counts of the logical experts are normalised by their total and split into the single-copy shares `t1` and the multi-copy right-hand side `b1`; the GPU rows of the right-hand side are `b2 = -(B1 @ t1)`. Both are then folded into the last column of `A_full`, which holds the negated right-hand side of the standard-form LP.

```text
total     = max(sum(global_counts), 1)
t1        = global_counts[log_single] / total
b1        = global_counts[log_replicated] / total
b2        = -(B1 @ t1)
b         = concat(b1, b2)
A_full[:, NV - 1] = b - A_base_row_sum
```

`A_full` is pre-allocated by the caller and its first `NV - 1` columns already hold the constant part of the constraint matrix; only the last column is written. `b` and `t1` are overwritten in place so that long-lived buffers can be reused across steps.

## Parameters and return value

All tensors must be on the same NPU device and contiguous.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `A_full` | Tensor | `Required` | Writable float32 [NC,NV]; only the last column is written. NC must equal `len(log_replicated) + B1.shape[0]`. |
| `b` | Tensor | `Required` | Writable float32 [NC] right-hand side buffer. |
| `t1` | Tensor | `Required` | Writable float32 [num_single] single-copy share buffer. |
| `global_counts` | Tensor | `Required` | Float32 [num_logical] token counts per logical expert. |
| `log_single` | Tensor | `Required` | Int32/int64 [num_single] logical ids that own exactly one physical copy. |
| `log_replicated` | Tensor | `Required` | Int32/int64 [num_red_log] logical ids that own several physical copies. |
| `B1` | Tensor | `Required` | Float32 [num_gpus,num_single] GPU assignment matrix of the single-copy experts. |
| `A_base_row_sum` | Tensor | `Required` | Float32 [NC] row sums of the constant part of the constraint matrix. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| A_full | Tensor | The same buffer, with the last column overwritten. |
| b | Tensor | The same buffer, overwritten with `concat(b1, b2)`. |
| t1 | Tensor | The same buffer, overwritten with the single-copy shares. |

## triton backend

- Single-program kernel (`grid=(1,)`): the count table is loaded into UB once and the per-expert counts are selected with an in-UB gather, so no random global reads are issued.
- `B1 @ t1` runs as one unrolled row-chunk tile sized to the GPU count, and the `t1` / `b` writes are fused into the same launch.
- Padding lanes of every tile contribute exactly zero to the reductions, so non-power-of-two expert counts need no special handling.

## torch_native backend

- Direct tensor expression of the formulas above (`sum`, `clamp`, gather by index, matvec), written independently of the kernel.
- Shares the same in-place contract, which makes it usable as the accuracy reference for the Triton path.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import lp_prep

num_gpus, num_single, num_red_log = 2, 2, 1
nc, nv = num_red_log + num_gpus, 2 * num_single + num_gpus + 1

A_full = torch.zeros(nc, nv, dtype=torch.float32, device="npu")
b = torch.empty(nc, dtype=torch.float32, device="npu")
t1 = torch.empty(num_single, dtype=torch.float32, device="npu")
counts = torch.tensor([20.0, 5.0, 7.0], device="npu")
log_single = torch.tensor([1, 2], dtype=torch.int32, device="npu")
log_replicated = torch.tensor([0], dtype=torch.int32, device="npu")
B1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu")

lp_prep(
    A_full,
    b,
    t1,
    counts,
    log_single,
    log_replicated,
    B1,
    A_full.sum(1).contiguous(),
    impl="triton",
)
```
