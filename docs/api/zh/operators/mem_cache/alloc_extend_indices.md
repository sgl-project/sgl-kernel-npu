# alloc_extend_indices

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices


def alloc_extend_indices(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
    *,
    impl="auto"
): ...
```

## 功能描述

为追加 token 填充打包的槽位索引，先利用原有未满页，再按批次顺序使用空闲页 ID。本接口不返回页数缓冲区，也不修改 free_pages，保留调用方索引张量的数据类型。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `pre_lens` | Tensor | `Required` | 连续 INT32/INT64 原序列长度 [batch]。 |
| `seq_lens` | Tensor | `Required` | 连续 INT32/INT64 目标长度 [batch]，不小于 pre_lens。 |
| `last_loc` | Tensor | `Required` | 连续整数末尾占用槽 [batch]，空前缀使用 -1。 |
| `free_pages` | Tensor | `Required` | 连续整数空闲页 ID [pages]，须满足所有追加需求。 |
| `page_size` | int | `Required` | 每页槽数，须为正的二次幂。 |
| `out_indices` | Tensor | `Required` | 可写整数打包槽 [sum(seq_lens-pre_lens)]。 |
| `max_num_extend_tokens` | int | `Required` | 每个请求追加 token 数的上界。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | None | 原地写入 out_indices。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices

pre = torch.tensor([1], device="npu", dtype=torch.int32)
seq = torch.tensor([3], device="npu", dtype=torch.int32)
last = torch.tensor([0], device="npu", dtype=torch.int32)
pages = torch.tensor([4], device="npu", dtype=torch.int32)
out = torch.empty(2, device="npu", dtype=torch.int32)
alloc_extend_indices(pre, seq, last, pages, 16, out, 2, impl="triton")
```
