# dcp_verify_page_tables

## Hardware support

| Implementation | arch22 (A2) | arch22 (A3) | arch35 (A5) |
| --- | --- | --- | --- |
| `ascendc` | Unverified | Unverified | Unsupported |
| `torch_native` | Unverified | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

dcp_verify_page_tables(
    slots: torch.Tensor,
    requests: torch.Tensor,
    ordinary: torch.Tensor,
    local: torch.Tensor,
    active_cols: int,
    query_len: int,
    page_size: int,
    dcp_size: int,
    *,
    impl: str = "auto",
) -> None
```

`impl` defaults to `"auto"`, which selects `"ascendc"` on arch22. Select `"torch_native"` explicitly to use the PyTorch implementation.

## Description

Build the ordinary and rank-local page tables used by Target Verify. Let $s_i$ be the request-pool row ID for batch row $i$, $P$ the page size, $D$ the DCP group size, and $Q$ the query count per request. For active ordinary column $j$, write $\mathrm{ordinary}[i,j] = \lfloor \mathrm{slots}[s_i,jP]/P \rfloor$. Local column $k$ uses ordinary column $kD$ and writes $\mathrm{local}[iQ+q,k] = \lfloor \mathrm{ordinary}[i,kD]/D \rfloor$ for every $0 \leq q < Q$.

Both buffers are overwritten in place. Inactive columns and all output rows for invalid request IDs are set to zero, including when the active range shrinks between calls.

## Parameters and return value

| Parameter | Type | Description and constraints |
| --- | --- | --- |
| `slots` | `torch.Tensor` | Contiguous int32 tensor `[pool_rows, slot_cols]` mapping request tokens to physical KV slots. |
| `requests` | `torch.Tensor` | Contiguous int32 or int64 tensor `[rows]` of request-pool row IDs. IDs outside `[0, pool_rows)` produce zero rows. |
| `ordinary` | `torch.Tensor` | Caller-allocated contiguous int32 tensor `[rows, ordinary_cols]`. |
| `local` | `torch.Tensor` | Caller-allocated contiguous int32 tensor `[rows * query_len, local_cols]`. |
| `active_cols` | `int` | Active ordinary page count in `[0, min(ceil(slot_cols / page_size), ordinary_cols, local_cols * dcp_size)]`. |
| `query_len` | `int` | Positive query count per request. |
| `page_size` | `int` | Tokens per page. A power of two in `[1, 65536]`. |
| `dcp_size` | `int` | DCP group size, 2 or 4. |
| `impl` | `str` | Backend selector. Defaults to `"auto"`. |

| Return value | Type | Description |
| --- | --- | --- |
| — | `None` | Writes to `ordinary` and `local` in place. |

All tensors must be on the same device, and `ascendc` requires NPU tensors. Neither output buffer may overlap another argument.

## ascendc backend

An Ascend C vector kernel reads the slot mapping and writes both page tables into the caller-provided buffers in one device launch.

## torch_native backend

Implements the same semantics using PyTorch tensor operations. This backend also accepts CPU tensors.

## Example

```python
import torch

from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

slots = torch.tensor([[8, 9, 16, 17, 24, 25]], dtype=torch.int32, device="npu")
requests = torch.tensor([0], dtype=torch.int64, device="npu")
ordinary = torch.empty((1, 4), dtype=torch.int32, device="npu")
local = torch.empty((2, 2), dtype=torch.int32, device="npu")
dcp_verify_page_tables(slots, requests, ordinary, local, 3, 2, 2, 2, impl="ascendc")
assert ordinary.tolist() == [[4, 8, 12, 0]]
assert local.tolist() == [[2, 6], [2, 6]]
```
