# move_kv_cache_rows

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`; both backends are registered without architecture restriction. All tensors use the same NPU.

```python
from sgl_kernel_npu.operators.mem_cache import move_kv_cache_rows

def move_kv_cache_rows(keys, values, tgt_loc, src_loc, *, impl="auto"): ...
```

## Description

Move MHA cache rows in place from a source snapshot, preserving 16-bit payloads even for overlapping moves.

## Parameters and return value

| Parameter | Type, shape and constraints |
| --- | --- |
| `keys` | FP16/BF16 `[layers,pages,slots,heads,dim]`; key cache with uniform row strides and contiguous row payloads. |
| `values` | Same layout as `keys`; value row width may differ. |
| `tgt_loc` | 1-D int32/int64; flattened destination slots. Non-identity targets are unique. |
| `src_loc` | 1-D int32/int64; paired flattened source slots. Sources may repeat or overlap destinations. |
| `impl` | str; `auto`, `triton`, or `torch_native` (default `auto`). |

The index tensors must have equal lengths. Repeated self-copies may serve as padding.

| Return value | Type and meaning |
| --- | --- |
| `None` | Output buffers are updated in place. |

## triton backend

Snapshots both payloads, then scatters them so overlapping rows do not change later reads.

## torch_native backend

Independent PyTorch reference verified on Ascend A3 (arch22).

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import move_kv_cache_rows

device = "npu"
keys = torch.zeros((1, 1, 4, 1, 16), dtype=torch.float16, device=device)
values = keys.clone()
src = torch.tensor([0], dtype=torch.int32, device=device)
dst = torch.tensor([1], dtype=torch.int32, device=device)
move_kv_cache_rows(keys, values, dst, src, impl="triton")
```
