# lightning_indexer

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer


def lightning_indexer(query, key, weights, *, impl="auto", **kwargs): ...
```

## Description

Ranks candidate keys by aggregating rectified query-key dot products across query heads. With head weights $w_h$, the score is $\sum_h w_h\max(0,q_h\cdot k_j)$. The SDK selects the requested top-k positions.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Sq, H, D] for BSND or [T, H, D] for TND, FP16/BF16 | Required | Query heads. D is 128 in the paged DSA workload. |
| `key` | Tensor | [P, L, Hkv, D] in PA_BSND, query dtype | Required | Key cache with matching head size. Dense layouts are also SDK options. |
| `weights` | Tensor | [B, Sq, H] or [T, H], query dtype as supported | Required | One score coefficient per query head. |
| `actual_seq_lengths_query`, `actual_seq_lengths_key` | Tensor or None | [B], INT32 | `None` | Optional lengths in SDK form for query/key layouts. |
| `block_table` | Tensor or None | [B, max_pages], INT32 | `None` | Optional logical-to-physical page mapping. Required in the paged example. |
| `layout_query`, `layout_key` | str | - | `"BSND"` | Optional layout tags, both BSND by default. The example uses PA_BSND for key. |
| `sparse_count`, `sparse_mode`, `return_value` | int, int, bool | - | `2048` / `3` / `False` | Optional SDK knobs. Defaults are 2048, 3 and False. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `indices`, `values` | tuple[Tensor, Tensor] | [B, Sq, Hkv, K] for BSND or [T, Hkv, K] for TND | Top-k positions as INT32. The score output is empty when return_value=False. |

## torch_npu backend

Calls `torch_npu.npu_lightning_indexer`. Shape, dtype and layout combinations follow the installed SDK. Current OpPlugin product documentation includes 950DT. Check the installed TorchNPU for the requested SoC and layouts. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import lightning_indexer

q = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16, device="npu")
key = torch.randn((32, 256, 1, 128), dtype=q.dtype, device=q.device)
weights = torch.ones((1, 1, 64), dtype=q.dtype, device=q.device)
q_lens = torch.tensor([1], dtype=torch.int32, device=q.device)
k_lens = torch.tensor([8192], dtype=torch.int32, device=q.device)
table = torch.arange(32, dtype=torch.int32, device=q.device).reshape(1, 32)
indices, values = lightning_indexer(
    q,
    key,
    weights,
    actual_seq_lengths_query=q_lens,
    actual_seq_lengths_key=k_lens,
    block_table=table,
    layout_key="PA_BSND",
    sparse_count=2048,
    impl="torch_npu",
)
```
