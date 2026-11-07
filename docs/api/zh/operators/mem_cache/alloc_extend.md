# alloc_extend

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def alloc_extend(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values,
    *, impl="auto",
): ...
```

默认的 `auto` 选择 `ascendc`。

## 功能说明

为分页请求在 Prefill 或 Decode 阶段新增的 token 计算 KV Cache 槽位索引。每个请求先填满当前页的剩余位置，再按 free_pages 的顺序使用新页，必要时写入最后一个不完整的新页；结果按请求顺序紧凑写入 out_indices。该操作只计算索引，不搬运 KV 数据，空闲页列表的消费仍由调用方管理。函数原地更新输出缓冲区并返回 None。

## 参数与返回值

所有输入和输出 Tensor 均使用同一张 NPU。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pre_lens` | `torch.Tensor` | `[B]` | `int64` | 必填 | 每个请求已有的长度，B 须大于零。 |
| `seq_lens` | `torch.Tensor` | `[B]` | `int64` | 必填 | 每个请求的目标长度，应大于等于对应的已有长度。 |
| `last_loc` | `torch.Tensor` | `[B]` | `int64` | 必填 | 各请求已有的最后一个槽位；空前缀使用 -1。 |
| `free_pages` | `torch.Tensor` | `[P]` | `int64` | 必填 | 按分配顺序排列的空闲页 ID，数量须满足本批请求。 |
| `page_size` | `int` | — | — | 必填 | 每页的槽位数，须为 int32 可表示的正数。 |
| `out_indices` | `torch.Tensor` | `[sum(seq_lens - pre_lens)]` | `int64` | 必填 | 调用方分配的输出缓冲区，保存新增槽位编号。 |
| `values` | `torch.Tensor` | [N], N ≥ 1 | `int64` | 必填 | 仅当最后一个请求结束于新分配且未填满的页时，向第 0 项写入整批新增页数；其他情况下保留原值。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字，可选 `"auto"`, `"ascendc"`, `"torch_native"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | `NoneType` | 原地写入 out_indices，并按上述条件写入 values[0]。 |

## ascendc 后端

使用稠密的一维缓冲区。

## torch_native 后端

在输入设备上计算，结果写入相同的输出缓冲区。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def vector(items):
    return torch.tensor(items, device="npu", dtype=torch.int64)

out = vector([0, 0, 0])
values = vector([-1])
alloc_extend(vector([3]), vector([6]), vector([22]), vector([8]), 4,
             out, values, impl="torch_native")
print(out.tolist(), values.item())  # [23, 32, 33], 1
```
