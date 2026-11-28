# fused_prepare_lim_metadata

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| triton | Unverified | Unverified |
| torch_native | Unverified | Unverified |

## Signature

`auto` selects Triton.

```python
from sgl_kernel_npu.operators.indexer import fused_prepare_lim_metadata

def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    query_width: int = 1,
    positions=None,
    resident_safe_slots=None,
    request_to_resident=None,
    resident_rows=None,
    impl="auto",
) -> None:
    ...
```

## Description

Prepares LIM request metadata and page tables in place; invalid requests use sentinel slots and sparse state depends on capacity and prefix history.

## Parameters and return value

| Parameter | Shape, dtype and meaning |
| --- | --- |
| `seq_lens` | [B] int32 lengths |
| `request_slots` | [B] int32 request IDs |
| `last_prefix` | [resident_rows + 1] int32 prefix history including sentinel |
| `block_table` | [B, S] int32 source pages, arbitrary strides |
| `out_block_table` | [B, D] int32 output pages, arbitrary strides |
| `safe_slots` | [B] int32 output request IDs including sentinel |
| `lengths` | [B] int32 output visible lengths |
| `sparse_rows` | [B] bool output sparse flags |
| `valid_rows` | [B] bool output validity flags |
| `request_states` | [B] int32 output states |
| `next_prefixes` | [B] int32 output prefix lengths |
| `cache_sizes` | [B] int32 output cache sizes |
| `capacity` | int cache capacity |
| `request_rows` | int request row count, excluding sentinel |
| `query_width` | int number of queries per request |
| `positions` | optional contiguous integer positions, with B × query_width elements |
| `resident_safe_slots` | optional [B] int32 output resident IDs |
| `request_to_resident` | optional [request_rows] int32 mapping |
| `resident_rows` | optional int resident row count, excluding sentinel |
| `impl` | str backend: auto, triton or torch_native |

Returns `None`; output tensors are modified in place. Tensors are contiguous and on the same device unless otherwise stated.

The visible length is the last query position plus one when `positions` is
supplied. Otherwise the operator uses `seq_lens`. Valid sparse rows use state
`-1` when the historical prefix is nonnegative and no longer than the visible
length, or `-2` on first fill or rollback. Short and invalid rows use `-3`.
Invalid rows take the request and resident sentinel indices and a length of
zero. The operator copies the intersecting page table columns and zeros any
trailing destination columns.

## triton backend

Uses the Triton kernel.

## torch_native backend

Uses PyTorch operations as a reference implementation.

## Example

```python
import torch
from sgl_kernel_npu.operators.indexer import fused_prepare_lim_metadata

def ints(values):
    return torch.tensor(values, device="npu", dtype=torch.int32)

visible = torch.empty_like(ints([4096]))
state = torch.empty_like(visible)
fused_prepare_lim_metadata(
    seq_lens=ints([4096]),
    request_slots=ints([1]),
    last_prefix=ints([-1, 4095, -1]),
    block_table=ints([list(range(32))]),
    out_block_table=ints([[0] * 32]),
    safe_slots=torch.empty_like(visible),
    lengths=visible,
    sparse_rows=torch.empty(1, device="npu", dtype=torch.bool),
    valid_rows=torch.empty(1, device="npu", dtype=torch.bool),
    request_states=state,
    next_prefixes=torch.empty_like(visible),
    cache_sizes=torch.empty_like(visible),
    capacity=4096,
    request_rows=2,
    impl="auto",
)
# visible contains 4096; state contains -1.
```
