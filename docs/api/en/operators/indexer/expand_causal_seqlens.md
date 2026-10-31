# expand_causal_seqlens

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Verified |
| `torch_native` | Supported | Verified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

def expand_causal_seqlens(
    cache_seqlens: torch.Tensor,
    query_lens: torch.Tensor,
    device: torch.device,
    total_q: int,
    max_query_len: int,
    *,
    seqlens_out: torch.Tensor | None = None,
    token_to_batch_out: torch.Tensor | None = None,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` and `triton` select the same entry point; `torch_native` selects an independent PyTorch implementation. Backends load lazily, with no automatic fallback.

## Description

Generate the visible KV length and request ID for each query token, using the total KV length and query count of each request. Results are packed in request order.

For example, a request with total KV length 10 and two query tokens produces visible lengths `[9, 10]`; both tokens receive that request's ID. Requests with no query tokens contribute no output. The causal starting length is clamped to zero.

## Parameters and return values

| Parameter | Type / Shape / Dtype | Default | Contract |
| --- | --- | --- | --- |
| `cache_seqlens` | Tensor `[B]`, int32 | Required | Total KV lengths including queries; contiguous and one-dimensional. |
| `query_lens` | Tensor `[B]`, int32 | Required | Nonnegative query counts; contiguous, one-dimensional, same batch as cache. |
| `device` | `torch.device` | Required | Must exactly match the input device; prefer `cache_seqlens.device`. |
| `total_q` | `int` | Required | Nonnegative host value equal to `sum(query_lens)`. |
| `max_query_len` | `int` | Required | Nonnegative upper bound on every query count; positive when `total_q>0`. It may exceed the actual maximum. |
| `seqlens_out` | Tensor `[total_q]`, int32 | `None` | Optional contiguous output buffer; overwritten and returned by identity. |
| `token_to_batch_out` | Tensor `[total_q]`, int64 | `None` | Optional contiguous output buffer; overwritten and returned by identity. |
| `impl` | `str` | `"auto"` | Keyword-only: `auto`, `triton`, `torch_native`. |

| Return value | Shape / Dtype | Meaning |
| --- | --- | --- |
| `seqlens` | `[total_q]`, int32 | Causal length visible to each packed query token. |
| `token_to_batch` | `[total_q]`, int64 | Request ID of each packed token. |

All tensors must share an NPU device; the return value is a tuple of two `torch.Tensor` objects. Either output buffer may be omitted independently; omitted outputs are allocated. Outputs must not overlap inputs or each other. Callers ensure nonnegative lengths and no int32 overflow in lengths, sums or indexing. The wrapper does not read device lengths to verify `total_q` or the maximum bound.

## triton backend

Input lengths must be contiguous int32 tensors on the requested device. Empty batches or zero query counts return empty outputs. Supplied output buffers are reused.

## torch_native backend

The independent reference lives in `implementation/torch_native/indexer/expand_causal_seqlens.py`, using `repeat_interleave`, `cumsum` and `index_select`. It does not call Triton. `max_query_len` controls only the Triton launch and is not used in native computation.

SGLang allocates outputs during metadata initialization and passes those same buffers as `seqlens_out` / `token_to_batch_out` during replay. `total_q` must match the existing output shape; device and dtype remain unchanged. The operator does not manage graphs or create host/device length mirrors.

## Example

Requires a version of sgl-kernel-npu exposing this API, torch_npu and Triton-Ascend. Allocate outputs once, then reuse their storage with another query distribution:

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

cache = torch.tensor([10, 8, 20], dtype=torch.int32, device="npu:0")
queries = torch.tensor([2, 0, 3], dtype=torch.int32, device=cache.device)
seqlens, requests = expand_causal_seqlens(cache, queries, cache.device, 5, 3, impl="triton")
# seqlens: [9, 10, 18, 19, 20]; requests: [0, 0, 2, 2, 2]

queries = torch.tensor([0, 3, 2], dtype=torch.int32, device=cache.device)
replayed = expand_causal_seqlens(
    cache, queries, cache.device, 5, 3,
    seqlens_out=seqlens, token_to_batch_out=requests, impl="triton",
)
assert replayed[0] is seqlens and replayed[1] is requests
# seqlens: [6, 7, 8, 19, 20]; requests: [1, 1, 1, 2, 2]
```
