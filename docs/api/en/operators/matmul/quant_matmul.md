# quant_matmul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-nn` wherever it is registered.

```python
from sgl_kernel_npu.operators.matmul import quant_matmul


def quant_matmul(x1, x2, scale, *, impl="auto", **kwargs): ...
```

## Description

Computes a quantized matrix product. In the INT8 per-token and per-channel configuration, the core computation is $Y_{mn}=(\sum_k A_{mk}B_{kn})\,s_m\,s_n$ before conversion to the requested output dtype. Other SDK modes can use different offsets and groups.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, K], [K, N] for INT8 | Required | Quantized operands with matching contraction dimension. Packed and batched forms depend on the SDK. |
| `scale` | Tensor | [N], FP32 for the example | Required | Weight-channel or group scale in the representation required by the selected mode. |
| `pertoken_scale` | Tensor or None | [M], FP32 for the example | `None` | Optional scale for each input token. |
| `output_dtype`, `group_sizes` | torch.dtype, list[int] | - | `None` | Optional result dtype and SDK group dimensions. |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | [M, N] in the example | Quantized matmul result with the requested output dtype. |

## ops-nn backend

Calls `torch_npu.npu_quant_matmul`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md) [CANN ops-nn source](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/matmul/quant_batch_matmul_v4)

## torch_native backend

Computes INT8 accumulation with floating-point channel scales, optional token scales, and optional bias.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import quant_matmul

a = torch.randint(-3, 4, (16, 128), dtype=torch.int8, device="npu")
b = torch.randint(-3, 4, (128, 64), dtype=torch.int8, device=a.device)
channel_scale = torch.full((64,), 0.25, dtype=torch.float32, device=a.device)
token_scale = torch.full((16,), 0.125, dtype=torch.float32, device=a.device)
y = quant_matmul(
    a,
    b,
    channel_scale,
    pertoken_scale=token_scale,
    output_dtype=torch.bfloat16,
    impl="ops-nn",
)
```
