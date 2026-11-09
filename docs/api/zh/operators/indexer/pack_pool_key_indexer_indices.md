# pack_pool_key_indexer_indices

## 芯片支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

def pack_pool_key_indexer_indices(
    indices, positions, query_ends, topk, pool_size, *, output_buffer=None, impl="auto"
):
    ...
```

`auto` 选择 Triton。

## 功能说明

保留 CANN 选出的完整历史候选，追加每个 query 尚未压缩的 tail，
将剩余列和图补齐行填为 -1。应先裁掉 sentinel 输出，再传入原始 query ends 和 positions。

有效 query 必须构成连续前缀。历史候选的因果性由 CANN 保证，本算子保留其顺序。
短历史只保留真实 tail，人工 pooled key 不会进入 attention。

## 参数与返回值

所有张量位于同一 NPU，末维连续。位置计算使用 int64，输出索引使用 int32。

| 参数 | 类型／形状／数据类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `indices` | Tensor `[T,C]`，int32 | 必填 | CANN 候选；列数容纳 topk 和最大 tail。 |
| `positions` | Tensor `[T]`，int64 | 必填 | 原始 token 位置。 |
| `query_ends` | Tensor `[R]`，int64 | 必填 | 原始真实 query 累积边界，最后一个值不计 padding。 |
| `topk` | int | 必填 | CANN 调用使用的 token 级历史预算。 |
| `pool_size` | int | 必填 | 一个完整压缩池包含的 token 数。 |
| `output_buffer` | Tensor 或 None，int32 | `None` | 可选输出；至少 T 行，列数须足够。 |
| `impl` | str | `"auto"` | `auto`、`triton`、`torch_native`。 |

| 返回值 | 类型／形状／数据类型 | 含义 |
| --- | --- | --- |
| 整理后的索引 | Tensor `[T,topk+pool_size-1]`，int32 | 默认是 indices 的视图，传入缓冲时为 output_buffer 的视图。 |

默认原地覆盖 indices。独立输出缓冲不会修改输入；其前 T 行的多余列填为 -1，后续行保持原值。

## triton 后端

每个 token 对应一个 program，沿用 vllm-ascend 原始 packing kernel。

## torch_native 后端

独立使用 PyTorch 选择历史和构造 tail，原地更新范围与 Triton 一致。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pack_pool_key_indexer_indices

indices = torch.full((2, 2051), -1, dtype=torch.int32, device="npu")
positions = torch.tensor([0, 1], dtype=torch.int64, device=indices.device)
ends = torch.tensor([2], dtype=torch.int64, device=indices.device)
packed = pack_pool_key_indexer_indices(indices, positions, ends, 2048, 4, impl="triton")
```
