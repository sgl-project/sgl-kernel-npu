# unidex_copy_inplace

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ascendc`.

```python
from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace


def unidex_copy_inplace(
    src, dst, src_index, dst_index, valid_mask, src_address_ndims, dst_address_ndims,
    block_dim=8, src_ptr=None, dst_ptr=None, *, impl="auto"
): ...
```

## Description

Copies rows between KV buffers. For each nonzero mask entry with both indices in range, it copies the source row to the indexed destination row. The leading address axes are flattened into a row index, and all trailing axes form the copied byte range. The two row byte sizes must match and be at most 32 KiB. Use distinct destination rows for concurrently active mappings.

## Parameters and return value

All tensors must be contiguous. Index and mask tensors are on the same NPU. Endpoints without an address override must also be on that NPU.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `src` | Tensor | Required | Source buffer, with the same dtype and row byte size as `dst`. |
| `dst` | Tensor | Required | Destination buffer, written in place. |
| `src_index` | Tensor | Required | INT64 `[mappings]` source row indices. |
| `dst_index` | Tensor | Required | INT64 `[mappings]` destination row indices. |
| `valid_mask` | Tensor | Required | Bool or UINT8 `[mappings]` copy mask. |
| `src_address_ndims` | int | Required | Positive number of leading source axes used to address rows, less than the source rank. |
| `dst_address_ndims` | int | Required | Positive number of leading destination axes used to address rows, less than the destination rank. |
| `block_dim` | int | `8` | Positive vector core count for AscendC. |
| `src_ptr` | int or None | `None` | AscendC source address override visible to the NPU. |
| `dst_ptr` | int or None | `None` | AscendC destination address override visible to the NPU. |
| `impl` | str | `"auto"` | Implementation to use. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | Original `dst` tensor. An empty mapping leaves it unchanged. |

## ascendc backend

- Copies bytes directly between NPU buffers or between NPU and registered host memory.
- Keep overridden addresses valid until work on the current stream completes. Use [`create_shm_tensor`](shm_allocator.md) and `free_shm` from `sgl_kernel_npu.operators.mem_cache` to manage registered shared memory.

## torch_native backend

- Uses PyTorch indexing to copy valid rows between Tensor-backed buffers. This implementation uses the Tensor addresses.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace

src = torch.randn(4, 32, dtype=torch.float16, device="npu")
dst = torch.zeros_like(src)
src_index = torch.tensor([3, 0], dtype=torch.int64, device="npu")
dst_index = torch.tensor([0, 2], dtype=torch.int64, device="npu")
mask = torch.tensor([True, False], device="npu")
result = unidex_copy_inplace(src, dst, src_index, dst_index, mask, 1, 1, impl="ascendc")
assert result is dst
```
