# weight_quant_batchmatmul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul


def weight_quant_batchmatmul(
    x, weight, antiquant_scale, *args, impl="auto", **kwargs
): ...
```

## Description

Dequantizes the weight matrix using per-channel or per-group scales and offsets, then multiplies it by floating activations. For a dequantized weight $W'$, the prequantization result is $Y=XW'+b$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [..., K], FP16/BF16 | Required | Floating activation on NPU. |
| `weight` | Tensor | [K, N] logically, INT8 or SDK packed dtype | Required | Quantized weight. Packing, physical format and stride constraints depend on the mode. |
| `antiquant_scale` | Tensor | [N] or [ceil(K/group_size), N] for ordinary scales | Required | Dequantization scale matching the weight recipe. |
| `*args / **kwargs` | SDK arguments | - | SDK default | Optional antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, inner_precise and weight_dtype in vendor order. Group size defaults to 0. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | [..., N], dtype from SDK output configuration | Matmul output. Its dtype depends on the input and optional output quantization. |

## ops-nn backend

Calls `torch_npu.npu_weight_quant_batchmatmul`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/matmul/weight_quant_batch_matmul_v2)

## torch_native backend

Dequantizes per-channel INT8 weights with optional offsets and computes A16W8 matmul without output requantization.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul

x = torch.randn((8192, 320), dtype=torch.float16, device="npu")
weight = torch.randint(-8, 8, (320, 256), dtype=torch.int8, device=x.device)
s = torch.full((1, 256), 0.125, dtype=x.dtype, device=x.device)
y = weight_quant_batchmatmul(x, weight, s, impl="ops-nn")
```
