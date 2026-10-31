# expand_kpool_group_and_tail_to_topk

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

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

Expand selected pool indices into the token positions represented by those pools, preserving their order. Then append the current sequence's incomplete pool as tail tokens.

For example, pool index 2 with `pool_size=4` expands to `[8, 9, 10, 11]`. Unused tail slots and padded query rows are filled with `-1`. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Selected pool positions. Row strides are passed to the kernel. |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`, with enough rows for the effective `n_real` | `torch.int32` | Required | Request length for each query row; use contiguous one-dimensional storage. |
| `n_real` | `int` | — | — | Required | Requested valid row count. The backend clamps it to the smaller input row count before launching. |
| `num_q_padded` | `int` | — | — | Required | Output row count; must be at least the effective `n_real`. |
| `pool_size` | `int` | — | — | Required | Number of tokens represented by one pool; Triton requires `pool_size >= 4`. |
| `index_topk` | `int` | — | — | Required | Token-level top-k; must equal `N_POOL_TOPK * pool_size`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | Expanded indices followed by tail slots. Inputs are not modified. |

All tensors must be on the same NPU. Callers provide valid input values; the wrapper does not validate every value range.

## triton backend

Pool-index strides are supported; sequence lengths must be contiguous. The returned output may be noncontiguous.

## torch_native backend

Uses broadcast arithmetic and length-based tail/padding construction; retains int32 input and contiguous-length requirements.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk
pool_indices = torch.arange(512, device='npu', dtype=torch.int32).repeat(2, 1)
seq_lens = torch.tensor([2050, 0], device='npu', dtype=torch.int32)
out = expand_kpool_group_and_tail_to_topk(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=2048,
    impl='torch_native',
)
print(out.shape)
```
