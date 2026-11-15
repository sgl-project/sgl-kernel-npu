# qsa_mqa_paged

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_mqa_paged

def qsa_mqa_paged(q, cache, table, lengths, width, *, impl="auto"):
    ...
```

## 说明

分页 QSA 打分：BF16 Q[R,4,128]、cache[P,16,1,128]，int32 table[R,L]、lengths[R]，width=16*L。输出 FP32[R,width]，有效长度之外为负无穷。table 存储物理页编号。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

对每行有效的 j 读取 `cache[table[r,j//16],j%16,0]`，在 FP32 中计算 `sum_h ReLU(dot(q[r,h],key)) / sqrt(128)`。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,4,128], BF16 | required | 连续 NPU query。 |
| `cache` | Tensor | [P,16,1,128], BF16 | required | q 同设备连续缓存。 |
| `table` | Tensor | [R,L], int32 | required | 连续物理页表；仅有效长度内的页号需要合法。 |
| `lengths` | Tensor | [R], int32 | required | q 同设备连续张量；0 <= lengths <= width。 |
| `width` | int | — | required | 16*L。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,width] | FP32 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU FP64 参考实现，结果返回输入设备。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_mqa_paged

q = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
cache = torch.randn(1, 16, 1, 128, device="npu", dtype=q.dtype)
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
lengths = torch.full((1,), 16, device="npu", dtype=torch.int32)
y = qsa_mqa_paged(q, cache, table, lengths, 16, impl="auto")
```
