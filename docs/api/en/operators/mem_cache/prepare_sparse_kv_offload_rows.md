# prepare_sparse_kv_offload_rows

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| triton | Unverified | Unverified |
| torch_native | Unverified | Unverified |

## Signature

`auto` selects Triton.

```python
prepare_sparse_kv_offload_rows(tokens, device, request_ids, seq_lens, positions, out_cache_loc, *, request_rows, context_len, host_capacity, query_width=1, extend_seq_lens=None, extend_prefix_lens=None, compact_prefill=False, impl="auto")
```

## Description

Prepare source row indices, host destination indices, token positions and validity flags for sparse KV offload. Decode maps each `query_width` tokens to a request. Prefill uses either compact variable-length rows or equally padded rows. Prefill is selected when `extend_seq_lens` is provided; missing prefix entries are treated as zero.

## Parameters and return value

| Parameter | Meaning |
| --- | --- |
| `tokens`, `device` | int output length and torch device |
| `request_ids`, `seq_lens` | One-dimensional integer tensors of length B containing request IDs and sequence lengths |
| `positions`, `out_cache_loc` | One-dimensional integer tensors of length tokens containing decode positions and destination indices; positions may be None for prefill |
| `request_rows`, `context_len`, `host_capacity` | int request, position and destination bounds |
| `query_width` | int decode tokens per request, default 1 |
| `extend_seq_lens`, `extend_prefix_lens` | One-dimensional integer tensors for prefill lengths and prefixes. Both default to None for decode. Pass a prefix tensor, which may be empty, when supplying extend lengths. |
| `compact_prefill` | bool compact prefill layout, default False |
| `impl` | Backend: auto, triton or torch_native |

Returns four tensors of shape [tokens]: source indices, destination indices and token positions (int64), then validity (bool). For padded prefill, tokens must be divisible by B. Inputs must be on the output device.

## triton backend

Uses the Triton kernel.

## torch_native backend

Uses PyTorch operations as an independent reference.

## Example

```python
import torch
from sgl_kernel_npu.operators.mem_cache import prepare_sparse_kv_offload_rows

device = torch.device("npu")
request_ids = torch.tensor([1], dtype=torch.int32, device=device)
seq_lens = torch.tensor([7], dtype=torch.int32, device=device)
query_positions = torch.tensor([6], dtype=torch.int32, device=device)
out_cache_loc = torch.tensor([12], dtype=torch.int32, device=device)
source, destination, positions, valid = prepare_sparse_kv_offload_rows(
    1, device, request_ids, seq_lens, query_positions, out_cache_loc,
    request_rows=2, context_len=32, host_capacity=64, impl="auto",
)
# [0], [12], [6], [True]
```
