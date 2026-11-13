# store_kv_cache_prefix_valid

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mem_cache import store_kv_cache_prefix_valid


def store_kv_cache_prefix_valid(
    dst_k, dst_v, src_k, src_v, loc_2d, commit_lens, *, impl="auto"
): ...
```

## Description

Commits only each request's accepted prefix from packed K/V source rows to flattened cache slots. The K and V head dimensions may differ. Unaccepted source locations are not dereferenced and other cache rows remain unchanged.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dst_k` | Tensor | `Required` | Writable [slots,heads,Kdim] cache, contiguous head/dim axes. |
| `dst_v` | Tensor | `Required` | Writable [slots,heads,Vdim] cache, same dtype as dst_k. |
| `src_k` | Tensor | `Required` | Packed [batch*width,heads,Kdim] keys, same dtype as caches. |
| `src_v` | Tensor | `Required` | Packed [batch*width,heads,Vdim] values, same dtype as caches. |
| `loc_2d` | Tensor | `Required` | Integer [batch,width] destination slots. Accepted positions must be valid; ignored positions may contain sentinels. |
| `commit_lens` | Tensor | `Required` | Integer [batch] accepted prefix sizes in [0,width]. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | None | Writes selected dst_k/dst_v rows in place. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import store_kv_cache_prefix_valid

dst_k = torch.zeros(16, 2, 64, device="npu", dtype=torch.bfloat16)
dst_v = torch.zeros_like(dst_k)
src_k = torch.randn(4, 2, 64, device="npu", dtype=dst_k.dtype)
src_v = torch.randn_like(src_k)
locations = torch.tensor([[3, 4, -1, -1]], device="npu", dtype=torch.int64)
lengths = torch.tensor([2], device="npu", dtype=torch.int32)
store_kv_cache_prefix_valid(
    dst_k, dst_v, src_k, src_v, locations, lengths, impl="triton"
)
```
