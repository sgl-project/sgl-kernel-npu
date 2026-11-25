# slot_map_lookup

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Unsupported |

## Signature

`auto` selects `ascendc`.

```python
from sgl_kernel_npu.operators.mem_cache import slot_map_lookup


def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim=0, *, impl="auto"): ...
```

## Description

Finds the device cache slot for each logical token in a request's top-k list. For each pair of valid indices, it returns the slot from `slot_map[req_indices[b], topk_indices[b, k]]` and a hit flag. Slot `-1` means absent. Out-of-range and negative request or token indices also return flag 0 and position `-1`.

## Parameters and return value

The inputs are contiguous and on the same NPU.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `slot_map` | Tensor | Required | INT32 `[slots, context]` map with nonnegative slot IDs or `-1`. `context` is a positive multiple of 8. |
| `req_indices` | Tensor | Required | INT32 `[batch]` request row indices. |
| `topk_indices` | Tensor | Required | INT32 `[batch, 2048]` token indices. |
| `block_dim` | int | `0` | AscendC vector core count. Zero uses all device vector cores. |
| `impl` | str | `"auto"` | Implementation to use. |

| Return | Type | Description |
| --- | --- | --- |
| `token_on_device` | Tensor | INT32 `[batch, 2048]` flags, 1 for a hit and 0 for a miss. |
| `device_token_pos` | Tensor | INT32 `[batch, 2048]` device slots, with `-1` for a miss. |

## ascendc backend

- Resolves the request and token indices in tiles and writes both outputs to the current NPU.
- A positive `block_dim` is limited by the available vector cores.

## torch_native backend

- Uses masked PyTorch indexing to look up valid positions and to form the integer hit flags.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import slot_map_lookup

slot_map = torch.full((2, 64), -1, dtype=torch.int32, device="npu")
slot_map[0, 7] = 12
req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
topk_indices = torch.full((1, 2048), 7, dtype=torch.int32, device="npu")
hits, positions = slot_map_lookup(slot_map, req_indices, topk_indices, impl="ascendc")
```
