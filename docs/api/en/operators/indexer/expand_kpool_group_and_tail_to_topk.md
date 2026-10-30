# expand_kpool_group_and_tail_to_topk

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk

def expand_kpool_group_and_tail_to_topk(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

Let `K=pool_size`, `P=pool_indices.shape[1]`, and `L_r=seq_lens_per_token[r]`:

$$O_{r,pK+j}=K I_{r,p}+j\;(0\le j<K),\quad q_r=\lfloor L_r/K\rfloor K,\quad t_r=L_r\bmod K.$$

The suffix has `K-1` columns: entry `j<t_r` equals `q_r+j`, otherwise `-1`. The effective row count is `min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])`; later padded rows contain only `-1`. Negative pool IDs are expanded arithmetically, not normalized to `-1`. Inputs are read only, `index_topk=P*K`, and callers provide nonnegative row counts, valid nonnegative lengths, and positive K.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Selected pool positions. Row strides are passed to the kernel. |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Request length for each query row; use contiguous one-dimensional storage. |
| `n_real` | `int` | — | — | Required | Requested valid row count. The backend clamps it to the smaller input row count before launching. |
| `num_q_padded` | `int` | — | — | Required | Output row count; must be at least the effective `n_real`. |
| `pool_size` | `int` | — | — | Required | Number of tokens represented by one pool; use a valid positive pooling width. |
| `index_topk` | `int` | — | — | Required | Token-level top-k; must equal `N_POOL_TOPK * pool_size`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | Expanded indices followed by tail slots. Inputs are not modified. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range.

## triton backend

Processes four-row blocks with grid-stride scheduling. Pool-index strides are supported; lengths must be contiguous. Physical output width is aligned to eight int32 elements before slicing, so output need not be contiguous.

## torch_native backend

Uses broadcast arithmetic and length-based tail/padding construction; retains int32 input and contiguous-length requirements.

Importing the public registry requires NPU architecture detection, including for `torch_native`. Support is limited to accepted A3 cases; architecture registration does not establish coverage of every chip, shape or dtype.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk
pool_indices = torch.tensor([[0, 1], [0, 0]], device='npu', dtype=torch.int32)
seq_lens = torch.tensor([10, 0], device='npu', dtype=torch.int32)
out = expand_kpool_group_and_tail_to_topk(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=8,
    impl='torch_native',
)
print(out.shape)
```
