# format_cast

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.data_move import format_cast


def format_cast(
    input, acl_format, *, customize_dtype=None, input_dtype=None, impl="auto"
): ...
```

## Description

Converts the physical storage layout of an NPU tensor. With an ordinary dtype and format conversion, the logical shape and values are preserved.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | any supported shape and input dtype | Required | Tensor to convert. |
| `acl_format` | int or Tensor | - | Required | Target ACL format ID, such as 2 for ND, or a tensor whose format is used. |
| `customize_dtype` | torch.dtype or None | - | `None` | Optional logical dtype for the target packed representation. |
| `input_dtype` | torch.dtype or None | - | `None` | Optional logical dtype for the input packed representation. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | target storage format | Converted tensor on the input NPU. |

## torch_npu backend

Calls `torch_npu.npu_format_cast`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import format_cast

x = torch.randn((16, 32), dtype=torch.float16, device="npu")
y = format_cast(x, 2, impl="torch_npu")
```
