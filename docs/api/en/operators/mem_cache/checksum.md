# adler32_strided_checksum

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

Both implementations were verified on Ascend A3 (arch22) and Ascend A5 (arch35).

## Signature

```python
import torch
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

def adler32_strided_checksum(
    data_ptrs: list[int] | list[torch.Tensor],
    strides: list[int],
    indices: list[torch.Tensor],
    *,
    impl="auto",
) -> int:
    ...
```

`auto` selects `triton`. There is no automatic fallback.

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
| `data_ptrs` | `list[torch.Tensor]` or `list[int]` | Required | Contiguous NPU Tensor buffers, one per descriptor. `auto`/`triton` also accept integer NPU addresses; nonempty, nonzero-stride descriptors require positive pointers. `torch_native` requires Tensor buffers. |
| `strides` | `list[int]` | Required | Nonnegative item sizes in bytes, equal to the contiguous byte range selected by each index. |
| `indices` | `list[torch.Tensor]` | Required | One contiguous 1D tensor `[num_items]` per descriptor, int32 or int64, all on the same NPU. |
| `impl` | `str` | `"auto"` | Keyword-only; `auto`, `triton` or `torch_native`. |

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

## torch_native backend

Uses ordinary PyTorch to read selected raw bytes from Tensor buffers and CPU
`zlib.adler32` to compute an independent reference. It does not call the Triton
kernel. Both backends accept the same Tensor inputs for accuracy comparison.
Integer addresses remain supported by `auto`/`triton` for existing SGLang callers;
the reference requires the owning Tensors and rejects integer addresses.

The CPU copy makes this implementation suitable for correctness checks rather
than production KV transfer. No attempt is made to reconstruct device storage
from unowned raw pointers.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

data = torch.arange(32, dtype=torch.uint8, device="npu").reshape(4, 8)
indices = torch.tensor([2, 0], dtype=torch.int64, device="npu")
value = adler32_strided_checksum(
    [data], [data[0].nbytes], [indices], impl="triton"
)
reference = adler32_strided_checksum(
    [data], [data[0].nbytes], [indices], impl="torch_native"
)
assert value == reference
print(value)
```
