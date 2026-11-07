# transfer_mamba_state

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, transfer_mamba_state

def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D, *, impl='auto'): ...
```

`auto` selects `aclrt`.

## Description

Transfer selected Mamba/SSM state slots between the NPU and pinned host memory for H2D restore
or D2H backup. Device states use `[layers, slots, *state_shape]`; host states use
`[slots,
layers, 1, *state_shape]`. Each paired device/host index transfers every layer of one
slot, including multidimensional state payloads. Unselected slots remain unchanged.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `device_buf` | Tensor | [L,D,...] | Matching payload dtype | Required | Device state [layers, slots, *state_shape]. |
| `host_buf` | Tensor | [H,L,1,...] | Matching payload dtype | Required | Pinned host state [slots, layers, 1, *state_shape]. |
| `device_indices` | Tensor | [N] | int64 | Required | Non-negative device slot indices, less than D; used directly without page-size conversion. |
| `host_indices` | Tensor | [N] | int64 | Required | Corresponding non-negative host slot indices, less than H. |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D or TransferDirection.D2H. |
| `impl` | str | — | — | 'auto' | auto (ACL Runtime), aclrt, or torch_native. |

| Return value | Type | Description |
| --- | --- | --- |
| `None` | NoneType | Updates the destination in place. Synchronize the submitting stream before reading it or reusing the source. |

## aclrt backend

Submits asynchronous 2D transfers to the current NPU stream. The host buffer must be pinned and contiguous; device slot payloads must be physically dense with packed slot/layer pitches. Dense payload transposes remain supported. Device/host dtypes and payload byte counts must match. All slot indices are validated before any copy, and ACL submission errors are propagated.

## torch_native backend

Adapts SGLang's `MambaPoolHost` PyTorch reference: H2D restores each layer with
`index_select/index_copy_`; D2H gathers selected slots and moves the layer dimension after the
slot dimension. Layout views preserve the physical storage order used by the `aclrt`
implementation, and repeated destination indices are written in input order.
The reference does not call the `aclrt` implementation.

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
