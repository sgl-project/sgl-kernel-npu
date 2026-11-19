# top_k_top_p

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.sampler import top_k_top_p


def top_k_top_p(logits, p=None, k=None, *, impl="auto"): ...
```

## Description

Masks scores below the k-th largest logit, then removes low-probability entries using cumulative probability after softmax on the k-filtered logits. Ties at the top-k cutoff remain eligible. Each row retains its highest score.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `logits` | Tensor | [B, V], FP16/BF16/FP32 | Required | ND unnormalized token scores. |
| `p` | Tensor or None | [B], logits.dtype | `None` | Top-p threshold in [0, 1]. At least one of p and k is required. |
| `k` | Tensor or None | [B], INT32 | `None` | Number of highest-scoring positions to retain before nucleus filtering. Use values in [1, min(1024, V)]. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | same as logits | Filtered logits in original vocabulary order. Removed values are -inf. |

## torch_npu backend

Calls `torch_npu.npu_top_k_top_p`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_top_k_top_p.md)

## torch_native backend

Sorts scores, applies the two filters, then scatters them back to their vocabulary indices.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sampler import top_k_top_p

x = torch.randn((1, 8192), dtype=torch.float32, device="npu")
p = torch.tensor([0.85], dtype=x.dtype, device=x.device)
k = torch.tensor([32], dtype=torch.int32, device=x.device)
filtered = top_k_top_p(x, p, k, impl="torch_npu")
```
