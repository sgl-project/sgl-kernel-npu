# adler32_strided_checksum

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Verified on Ascend A3 | Unverified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

def adler32_strided_checksum(
    data_ptrs: list[int],
    strides: list[int],
    indices: list[torch.Tensor],
    *,
    impl="auto",
) -> int:
    ...
```

`auto` selects `triton`. The implementation is declared for both architectures;
arch35 still needs hardware validation. There is no automatic fallback.

## Description

Compute Adler-32 for PD KV cache verification. Concatenate descriptors in input
order, and for each descriptor concatenate the `stride` bytes at
`data_ptr + indices[j] * stride` in index order. Duplicate indices contribute
repeated bytes. Storage dtype does not change this raw-byte interpretation.

For the resulting byte stream `x[0:N]`, with `M = 65521`:

- `a = (1 + sum(x[j])) % M`
- `b = (N + sum((N - j) * x[j])) % M`
- `checksum = (b << 16) | a`

The empty byte stream returns 1. Inputs are not modified. Only reduced partial
sums are copied to the host; returning the Python integer synchronizes this
result. Checksum equality does not prove correctness of model computation.

## Parameters and return value

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data_ptrs` | `list[int]` | Required | NPU base addresses; one per descriptor. Nonempty, nonzero-stride descriptors require positive pointers. |
| `strides` | `list[int]` | Required | Nonnegative item sizes in bytes, equal to the contiguous byte range selected by each index. |
| `indices` | `list[torch.Tensor]` | Required | One contiguous 1D tensor `[num_items]` per descriptor, int32 or int64, all on the same NPU. |
| `impl` | `str` | `"auto"` | Keyword-only; `auto` or `triton`. |

| Return value | Type | Description |
| --- | --- | --- |
| Checksum | `int` | Unsigned 32-bit Adler-32 encoded in a Python integer. |

All three lists must have equal lengths. Callers keep referenced allocations
alive, provide valid nonnegative in-bounds indices and sufficient contiguous
storage, and complete or order all writes before the kernel reads them. Raw
pointers do not carry allocation bounds or synchronization metadata.

## triton backend

The kernel reads selected rows as bytes, computes weighted partial sums modulo
65521, and reduces them to a checksum. It uses the same descriptor and index
ordering as SGLang's CUDA checksum. This migration preserves the kernel math
and launch plan.

A `torch_native` reference is not provided: the API contains raw device addresses
and cannot recover their owning tensors through ordinary PyTorch operations.
Tests retain those tensors and use independent Python `zlib.adler32` over their
selected bytes. The public API tests preserve the six original test categories
and additionally cover representative Qwen3-32B TP2 page descriptors.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

data = torch.arange(32, dtype=torch.uint8, device="npu").reshape(4, 8)
indices = torch.tensor([2, 0], dtype=torch.int64, device="npu")
value = adler32_strided_checksum(
    [data.data_ptr()], [data[0].nbytes], [indices], impl="triton"
)
print(value)
```
