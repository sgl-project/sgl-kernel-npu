# dynamic_dual_level_mx_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant


def dynamic_dual_level_mx_quant(input, *, impl="auto", **kwargs): ...
```

## Description

Quantizes the last input axis into FP4 values. A level-0 FP32 scale describes each 512-element group and level-1 E8M0 scales describe 32-element subgroups. Reconstruction multiplies the decoded value by both scales.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [..., K], FP16/BF16 | Required | ND activations. Packed output requires an even last dimension. |
| `smooth_scale` | Tensor or None | [K] | `None` | Optional SDK smoothing factors for input features. |
| `round_mode` | str | - | `"rint"` | Optional rounding control. The SDK defaults to rint. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `values` | Tensor | [..., K/2], packed FP4 | Two FP4 values per byte. |
| `level0_scale` | Tensor | [..., ceil(K/512)], FP32 | Scale for each 512-element group. |
| `level1_scale` | Tensor | [..., ceil(K/64), 2], E8M0 bytes | Paired scales for 32-element subgroups. |

## ops-nn backend

Calls `torch_npu.npu_dynamic_dual_level_mx_quant`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_dynamic_dual_level_mx_quant.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/quant/dynamic_dual_level_mx_quant)

## torch_native backend

Computes the 512-element level-0 scales and the 32-element E8M0 scales, then packs the RINT FP4 E2M1 values.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

x = torch.randn((1, 512), dtype=torch.bfloat16, device="npu")
y, s0, s1 = dynamic_dual_level_mx_quant(x, impl="ops-nn")
```
