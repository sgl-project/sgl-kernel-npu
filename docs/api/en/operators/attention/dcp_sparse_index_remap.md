# dcp_sparse_index_remap

## Hardware support

| Implementation | arch22 (A2) | arch22 (A3) | arch35 (A5) |
| --- | --- | --- | --- |
| `ascendc` | Unverified | Unverified | Unsupported |
| `torch_native` | Unverified | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

dcp_sparse_index_remap(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    *,
    impl: str = "auto",
) -> torch.Tensor
```

`impl` defaults to `"auto"`, which selects `"ascendc"` on arch22. Select `"torch_native"` explicitly to use the PyTorch implementation.

## Description

Convert global sparse token indices into compact rank-local KV positions for decode context parallelism (DCP). For an input index $i \geq 0$, let $P$ be the page size and $D$ the DCP group size. Its global page is $p = \lfloor i/P \rfloor$. Rank $r$ keeps the index if $p \bmod D = r$ and writes $\lfloor p/D \rfloor P + i \bmod P$.

Each row preserves the order and duplicates of retained indices. Negative inputs are ignored and unused output columns are filled with `-1`.

## Parameters and return value

| Parameter | Type | Description and constraints |
| --- | --- | --- |
| `indices` | `torch.Tensor` | Contiguous int32 tensor `[rows, 2048]`. Use an NPU tensor with `ascendc`. |
| `dcp_size` | `int` | DCP group size, 2 or 4. |
| `dcp_rank` | `int` | Zero-based rank in `[0, dcp_size)`. |
| `page_size` | `int` | Tokens per page. A power of two in `[1, 65536]`. |
| `impl` | `str` | Backend selector. Defaults to `"auto"`. |

| Return value | Type | Description |
| --- | --- | --- |
| output | `torch.Tensor` | Contiguous int32 tensor on the same device with the same shape as `indices`. |

## ascendc backend

An Ascend C vector kernel filters entries by page ownership, compacts them and fills the tail in one device launch.

## torch_native backend

Implements the same semantics using PyTorch tensor operations. This backend also accepts CPU tensors.

## Example

```python
import torch

from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

indices = torch.full((1, 2048), -1, dtype=torch.int32, device="npu")
indices[0, :3] = torch.tensor([128, 0, 130], dtype=torch.int32, device="npu")
local_indices = dcp_sparse_index_remap(indices, 4, 1, 128, impl="ascendc")
assert local_indices[0, :3].tolist() == [0, 2, -1]
```
