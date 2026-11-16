# prepare_chunk_indices

## 芯片支持

| 实现 | ARCH22 | ARCH35 |
| --- | --- | --- |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import prepare_chunk_indices

def prepare_chunk_indices(cu_seqlens, chunk_size, *, impl="auto"):
    ...
```

`auto` 选择 `torch_native`。

## 功能说明

为 packed GDN batch 生成分块坐标。长度为 `L` 的序列生成
`ceil(L / chunk_size)` 行，每行记录该序列在非空序列中的编号和从零开始的
序列内块编号。空序列不产生行。Ascend GDN prefill 使用输出行数确定
chunk 初始状态缓冲区的大小。

## 参数与返回值

| 参数 | 类型 / shape | dtype / 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `cu_seqlens` | Tensor `[B+1]` | int32/int64，必填 | 以 0 开头的非递减累计 token 边界，`B >= 1`。 |
| `chunk_size` | int | 必填 | 每块的最大 token 数，取正整数。 |
| `impl` | str | `"auto"` | `auto` 或 `torch_native`。 |

| 返回值 | 类型 / shape | dtype / 设备 | 说明 |
| --- | --- | --- | --- |
| 分块索引 | Tensor `[total_chunks, 2]` | 与 `cu_seqlens` 相同 | 按 packed 顺序排列的序列编号和序列内块编号。 |

## torch_native 后端

使用 PyTorch 构造整数坐标。分块数量在 Host 上读取，NPU 输入可能触发同步。
结果按输入张量对象和分块大小缓存。复用同一输入对象时，应保持其内容及返回的元数据不变。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import prepare_chunk_indices

cu_seqlens = torch.tensor([0, 70, 100], dtype=torch.int32, device="npu")
chunk_indices = prepare_chunk_indices(cu_seqlens, 64, impl="torch_native")
# [[0, 0], [0, 1], [1, 0]]
```
