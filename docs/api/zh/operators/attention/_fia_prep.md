# fia_prep

## 硬件支持

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 未验证 | 未验证 |
| `torch_native` | 未验证 | 未验证 |

## Signature

```python
from sgl_kernel_npu.operators.attention import fia_prep
def fia_prep(ti, seq_lens, per_query_req, req_to_token, block_size, block_table_out=None, actual_kvlen_out=None, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

将当前逻辑块移至末尾，映射物理页并计算因果 KV 长度；可复用传入的输出缓冲区。

## 参数与返回值

ti: int32 [T,topk] with -1 padding; seq_lens and per_query_req: [T]; req_to_token: [requests,max_ctx]. Returns (int32 [T,topk], int32 [T]). Valid input must leave a slot for the own block.

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import fia_prep

ti = torch.tensor([[0, 1, -1, -1]], device="npu", dtype=torch.int32)
lengths = torch.tensor([17], device="npu", dtype=torch.int32)
req = torch.tensor([0], device="npu", dtype=torch.int32)
slots = torch.arange(64, device="npu", dtype=torch.int32).reshape(1, 64)
table, kvlen = fia_prep(ti, lengths, req, slots, 16, impl="triton")
```
