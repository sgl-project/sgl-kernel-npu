# cache_loc_assign / cache_loc_update

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

def cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc, *, impl="auto"): ...
def cache_loc_update(req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc, *, impl="auto"): ...
```

两个函数的 `impl="auto"` 均选择 `ascendc`。使用 PyTorch 实现时请显式传入 `impl="torch_native"`。

## 功能说明

这两个函数在请求映射表和打包缓冲区之间复制缓存槽位编号。第 `i` 个请求对应映射表中的区间 `token_pool[req_pool_indices[i], start_offset[i]:end_offset[i]]`。各区间按请求顺序紧密排列在缓冲区中，空区间不占位置。

### cache_loc_assign

将缓冲区中的每段槽位编号写入对应的映射表行。目标区间以外的位置保留原值。

### cache_loc_update

从映射表中依次取出各区间的槽位编号，写入打包缓冲区。最后一个区间之后的缓冲区元素保留原值。

## 参数与返回值

所有 Tensor 均位于同一 NPU。映射表和索引使用稠密存储，行号与区间应在表内有效，映射表与打包缓冲区的存储不能重叠。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `req_pool_indices` | `torch.Tensor` | `[B]` | `int32 / int64` | 必填 | 每个请求对应的映射表行号，使用互异的行号。 |
| `token_pool` | `torch.Tensor` | `[pool_rows, row_width]` | `int32` | 必填 | 缓存槽位映射表。assign 写此表，update 读此表。 |
| `start_offset` | `torch.Tensor` | `[B]` | `int64` | 必填 | 每个请求的起始列，包含该列。 |
| `end_offset` | `torch.Tensor` | `[B]` | `int64` | 必填 | 每个请求的结束列，不包含该列，且不能小于起始列。 |
| `out_cache_loc` | `torch.Tensor` | 任意形状 | `int32` | 必填 | 连续缓冲区，至少有 `sum(end_offset - start_offset)` 个元素。多维缓冲区按一维访问。assign 从中读取，update 向其写入。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字。 `auto`, `ascendc`, `torch_native` |

| 函数 | 返回类型 | Shape 与 dtype | 说明 |
| --- | --- | --- | --- |
| `cache_loc_assign` | `torch.Tensor` | 同 `token_pool` | 与原地修改的 `token_pool` 共享存储。 |
| `cache_loc_update` | `torch.Tensor` | 同 `out_cache_loc` | 与原地修改的 `out_cache_loc` 共享存储。 |

## ascendc 后端

每个区间最多 16 个槽位编号。存储需满足对齐读取要求，每个实际访问的起始列之后应有足够空间访问 16 个元素。批量大小和打包缓冲区长度受设备片上缓冲容量限制。

## torch_native 后端

区间的打包顺序和原地写入的目标与 AscendC 相同。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

rows = torch.arange(8, device="npu", dtype=torch.int32)
pool = torch.zeros((8, 32), device="npu", dtype=torch.int32)
starts = torch.zeros(8, device="npu", dtype=torch.int64)
ends = starts + 1
slots = rows + 100

cache_loc_assign(rows, pool, starts, ends, slots, impl="ascendc")
result = torch.empty_like(slots)
cache_loc_update(rows, pool, starts, ends, result, impl="ascendc")
# result: [100, 101, 102, 103, 104, 105, 106, 107]
```
