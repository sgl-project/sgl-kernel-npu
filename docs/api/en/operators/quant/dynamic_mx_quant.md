# dynamic_mx_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.quant import dynamic_mx_quant


def dynamic_mx_quant(input, *, impl="auto", **kwargs): ...
```

## Description

Finds a shared power-of-two scale $s=2^e$ for each block on the quantization axis, then converts $x/s$ to FP8 or packed FP4. The scale algorithm and destination dtype determine the exponent selection and rounding.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32, ND | Required | Floating activations. An FP4 destination requires an even last dimension. |
| `axis`, `block_size` | int | - | `-1` / `32` | Optional quantization axis and block size. Defaults are -1 and 32. |
| `dst_type` | torch.dtype or SDK dtype | - | `torch_npu.float4_e2m1fn_x2` | Optional destination, such as torch.float8_e4m3fn or torch_npu.float4_e2m1fn_x2. |
| `round_mode`, `scale_alg` | str, int | - | `"rint"` / `0` | Optional rounding and scale algorithm. Defaults are rint and 0. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `values` | Tensor | FP8 or packed FP4 bytes | FP8 keeps the input shape. FP4 packs two values per byte along the last dimension. |
| `scales` | Tensor | E8M0, stored as bytes | Scale groups are paired in a new final dimension of size 2. On the quantization axis there are ceil(ceil(size / block_size) / 2) pairs. |

## ops-nn backend

Calls `torch_npu.npu_dynamic_mx_quant`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/quant/dynamic_mx_quant)

## torch_native backend

Quantizes the last axis to MXFP8 or packed FP4 E2M1 with RINT and scale algorithm 0.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_mx_quant

x = torch.randn((16, 256), dtype=torch.bfloat16, device="npu")
y, s = dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, impl="ops-nn")
```
