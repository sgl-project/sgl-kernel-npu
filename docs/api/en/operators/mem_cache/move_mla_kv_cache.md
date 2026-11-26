# move_mla_kv_cache

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`; both backends are registered without architecture restriction. All tensors use the same NPU.

```python
from sgl_kernel_npu.operators.mem_cache import move_mla_kv_cache

def move_mla_kv_cache(k_cache, rope_cache, tgt_loc, src_loc, page_size, use_fia_nz=False, *, impl="auto"): ...
```

## Description

Move MLA latent and RoPE rows in place from a snapshot. NZ uses physical [layer,page,dim/16,slot,16] addressing.

## Parameters and return value

| Parameter | Type, shape and constraints |
| --- | --- |
| `k_cache` | FP16/BF16 `[layers,pages,slots,1,dim]`; latent cache. |
| `rope_cache` | Same layout as `k_cache`; RoPE row width may differ. |
| `tgt_loc` | int32/int64; flattened destination slots. Non-identity targets are unique. |
| `src_loc` | int32/int64; paired flattened source slots. Sources may repeat or overlap destinations. |
| `page_size` | int; number of logical slots in a page. |
| `use_fia_nz` | bool (default `False`); when true, both buffers have contiguous packed NZ storage and row widths divisible by 16. |
| `impl` | str; `auto`, `triton`, or `torch_native` (default `auto`). |

The index tensors must have equal element counts. Repeated self-copies may serve as padding; for NZ the padding slot may not also be a real destination.

| Return value | Type and meaning |
| --- | --- |
| `None` | Output buffers are updated in place. |

## triton backend

Uses snapshot/scatter for ND and fused physical index expansion for NZ.

## torch_native backend

Independent PyTorch reference verified on Ascend A3 (arch22).

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import move_mla_kv_cache

device = "npu"
k = torch.zeros((1, 1, 4, 1, 16), dtype=torch.float16, device=device)
r = k.clone()
src = torch.tensor([0], dtype=torch.int32, device=device)
dst = torch.tensor([1], dtype=torch.int32, device=device)
move_mla_kv_cache(k, r, dst, src, 4, impl="triton")
```
