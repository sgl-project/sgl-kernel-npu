# assign_cache_op

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

def assign_cache_op(
    dst, src, dst_start, dst_end, src_start, src_end, *, impl="auto",
): ...
```

默认的 `auto` 选择 `ascendc`。

## 功能说明

将打包源缓冲区中的变长数据段，复制到各目标行指定的列区间。它是通用的分段复制操作，可用于在分配 KV Cache 槽位后更新请求到 token 的映射表，也可用于具有相同分段布局的其他稠密数据。复制按 1、2、4、8 字节元素逐位保留原数据，不进行 dtype 数值转换。函数原地修改目标缓冲区并返回布尔状态。

## 参数与返回值

所有输入和输出 Tensor 均使用同一张 NPU。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dst` | `torch.Tensor` | `[B, W]` | 按元素宽度 | 必填 | 稠密目标。元素宽度为 1、2、4 或 8 字节时按位复制。 |
| `src` | `torch.Tensor` | `[N]` | 按元素宽度 | 必填 | 一维稠密源数据；逐元素复制时请使用与目标相同的元素宽度。 |
| `dst_start` | `torch.Tensor` | [L], L ≥ B | `int64` | 必填 | 各行目标起始列，包含该列。 |
| `dst_end` | `torch.Tensor` | [L], L ≥ B | `int64` | 必填 | 各行目标结束列，不包含该列。 |
| `src_start` | `torch.Tensor` | [L], L ≥ B | `int64` | 必填 | 各行对应的源起始偏移，包含该位置。 |
| `src_end` | `torch.Tensor` | [L], L ≥ B | `int64` | 必填 | 各行对应的源结束偏移，不包含该位置。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字，可选 `"auto"`, `"ascendc"`, `"torch_native"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `ok` | `bool` | 输入缺失、为标量或索引向量不足 B 项时返回 False。其余情况下正常返回 True，包括元素宽度不支持而跳过复制的情况。 |

## ascendc 后端

使用正长度区间，并确保每段末尾按 32 字节向上取整后的访问仍在可用存储中。

## torch_native 后端

在输入设备上复制，直接更新传入的目标 Tensor。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

dst = torch.zeros((2, 16), device="npu", dtype=torch.int32)
src = torch.arange(16, device="npu", dtype=torch.int32)
starts = torch.tensor([0, 8], device="npu", dtype=torch.int64)
cols = torch.zeros_like(starts)
ok = assign_cache_op(dst, src, cols, cols + 8, starts, starts + 8, impl="ascendc")
print(ok, dst[:, 0].tolist())  # True, [0, 8]
```
