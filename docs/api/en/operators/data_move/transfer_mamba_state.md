# transfer_mamba_state

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `triton` | Not implemented | Not implemented |
| `tilelang` | Not implemented | Not implemented |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, transfer_mamba_state

def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Copy selected Mamba slots across all layers.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_buf` | Tensor | [L,D,...] | Matching payload dtype | Required | Device state [layers, slots, *state_shape]. |
| `host_buf` | Tensor | [H,L,1,...] | Matching payload dtype | Required | Pinned host state [slots, layers, 1, *state_shape]. |
| `device_indices` | Tensor | [N] | int64 | Required | Device slot indices (token indices for KV). |
| `host_indices` | Tensor | [N] | int64 | Required | Matching host slot or token indices. |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D or TransferDirection.D2H. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream.

## torch_native backend

Uses PyTorch operations on the input devices.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_mamba_state

dev = torch.zeros(2, 4, 8, device="npu")
host = torch.ones(4, 2, 1, 8, pin_memory=True)
slots = torch.tensor([1, 3])
transfer_mamba_state(dev, host, slots, slots)
torch.npu.synchronize()
print(dev[:, 1])
```
