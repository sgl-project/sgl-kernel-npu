# prepare_copy_sfa_metadata

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| triton | Unverified | Unverified |
| torch_native | Unverified | Unverified |

## Signature

`auto` selects Triton.

```python
from sgl_kernel_npu.operators.attention import prepare_copy_sfa_metadata

def prepare_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    capacity: int,
    query_width: int = 1,
    impl="auto",
) -> None:
    ...
```

## Description

Prepares Copy-SFA lengths and HBM page tables in place, and sets invalid source IDs to -1.

## Parameters and return value

| Parameter | Shape, dtype and meaning |
| --- | --- |
| `source_ids` | [B * query_width, K] int32 source IDs, modified for invalid rows |
| `valid_rows` | [B] bool input validity flags |
| `sparse_rows` | [B] bool input sparse flags |
| `request_slots` | [B] int32 request IDs |
| `seq_lens` | [B] int32 lengths |
| `actual_seq_lengths_kv` | [B] int32 output lengths |
| `hbm_block_table` | [B, H] int32 output pages |
| `capacity` | int cache capacity |
| `query_width` | int number of queries per request |
| `impl` | str backend: auto, triton or torch_native |

Returns `None`; output tensors are modified in place. Tensors are contiguous and on the same device unless otherwise stated.

Valid source IDs remain unchanged and invalid rows become -1. HBM column j is request slot times H plus j for valid rows and j for invalid rows. The length is capacity + query_width - 1 for valid sparse rows, seq_lens for valid non-sparse rows and zero for invalid rows. B, Q, K and H must be positive; K must be a power of two for Triton arange and K >= H. Tensors must be contiguous.

## triton backend

Uses the Triton kernel.

## torch_native backend

Uses PyTorch operations as a reference implementation.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import prepare_copy_sfa_metadata

prepare_copy_sfa_metadata(
    source_ids=torch.zeros((2, 16), dtype=torch.int32, device="npu"),
    valid_rows=torch.tensor([True, False], device="npu"),
    sparse_rows=torch.tensor([True, False], device="npu"),
    request_slots=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([8, 3], dtype=torch.int32, device="npu"),
    actual_seq_lengths_kv=torch.empty(2, dtype=torch.int32, device="npu"),
    hbm_block_table=torch.empty((2, 8), dtype=torch.int32, device="npu"),
    capacity=4,
)
```
