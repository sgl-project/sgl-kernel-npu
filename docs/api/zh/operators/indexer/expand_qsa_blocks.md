# expand_qsa_blocks

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

def expand_qsa_blocks(blocks, positions, lengths, ratio, topk, *, impl="auto"):
    ...
```

## 说明

将 int32/int64 blocks[R,K] 展开为 token 索引，K=512/2048，topk=K*ratio。positions/lengths 为每行的查询位置和序列长度。输出连续 int32[R,topk+ratio-1]，依次为有效块、局部尾部、-1 补齐。块列表必须为有效前缀接 -1 后缀。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

每个块 b 展开为 `b*ratio + j`，其中 `j=0..ratio-1`。保留块顺序和重复项，追加未满一块的尾部 token，再用 -1 补齐。例如 ratio=4、position=12、length=13 时，块 2 展开为 `[8,9,10,11]`，随后追加尾部 token 12。尾部范围为 `[floor((position+1)/ratio)*ratio, min(position+1,length))`。

有效块必须放在所有 -1 前面，且每个块必须完整：`(b+1)*ratio <= min(position+1,length)`。这些数值要求由调用方保证；算子只检查形状、类型和步长。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `blocks` | Tensor | [R,K], int32/int64 | required | NPU，接受带步长视图；K=512/2048；有效前缀之后为 -1。 |
| `positions` | Tensor | [R], int32/int64 | required | 同设备；查询位置范围 [-1,2**31-2]，活动行必须小于 length；接受带步长视图。 |
| `lengths` | Tensor | [R], int32/int64 | required | 同设备；序列长度范围 [0,2**31-1]；接受带步长视图。空行可用 position=0、length=0。 |
| `ratio` | int | — | required | 至少为 2。 |
| `topk` | int | — | required | K*ratio；输出宽度 topk+ratio-1 不超过 2**31-1024。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk+ratio-1] | int32 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU 参考实现，逐块展开后追加尾部 token。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks

blocks = torch.full((1, 512), -1, device="npu", dtype=torch.int32)
position = torch.zeros(1, device="npu", dtype=torch.int32)
y = expand_qsa_blocks(blocks, position, position + 1, 4, 2048, impl="auto")
```
