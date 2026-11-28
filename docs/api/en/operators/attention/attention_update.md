# attention_update

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import attention_update


def attention_update(lse, local_out, update_type, *, impl="auto"): ...
```

## Description

Combines attention over disjoint partitions. With local log-sum-exp values $L_i$ and normalized outputs $O_i$, the merged values are $L=\log\sum_i e^{L_i}$ and $O=\sum_i e^{L_i-L}O_i$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `lse` | sequence[Tensor] | P tensors of [N], FP32 | Required | Per-partition LSE, where N flattens token and head axes. |
| `local_out` | sequence[Tensor] | P tensors of [N, D], same floating dtype | Required | Normalized output per partition on the same NPU as its LSE. |
| `update_type` | int | - | Required | 0 computes the merged output. 1 also computes merged LSE. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output`, `lse` | tuple[Tensor, Tensor] | [N, D], [N] | Merged attention and SDK LSE buffer. Only mode 1 computes a usable merged LSE. |

## ops-transformer backend

Calls `torch_npu.npu_attention_update`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_attention_update.md) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/attention_update)

## torch_native backend

Merges partition outputs with PyTorch softmax weights and returns the merged LSE in mode 1.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import attention_update

lse_parts = [torch.zeros((64,), dtype=torch.float32, device="npu") for _ in range(2)]
values = [torch.ones((64, 512), dtype=torch.float32, device="npu") for _ in range(2)]
out, lse = attention_update(lse_parts, values, 1, impl="ops-transformer")
```
