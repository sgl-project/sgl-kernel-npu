# prepare_pool_key_indexer_inputs

## 芯片支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

def prepare_pool_key_indexer_inputs(
    query, weights, ends, lengths, tails, table, *, impl="auto"
):
    ...
```

`auto` 选择 Triton。

## 功能说明

让没有完整压缩池的请求，以及全 padding 的图回放，安全调用 CANN 池化索引算子。
追加一行零 query 和权重，让虚拟请求至少有一个 pooled key，并将空历史长度临时置为一。
返回六份连续输出，不修改输入。

CANN 调用后裁掉新增输出行，再用原始 positions 和 query ends 调用
`pack_pool_key_indexer_indices`，排除人工候选。

## 参数与返回值

所有张量位于同一 NPU。T 为静态 token 数，R 为请求数，M 为页表行数。

| 参数 | 类型／形状／数据类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `query` | Tensor `[T,H,D]`，BF16 | 必填 | 打包后的 query。 |
| `weights` | Tensor `[T,H]`，BF16 | 必填 | 每个 head 的权重。 |
| `ends` | Tensor `[R]`，int64 | 必填 | 真实 query 的累积结束位置，不计 padding。 |
| `lengths` | Tensor `[R]`，int64 | 必填 | 完整压缩池数量。 |
| `tails` | Tensor `[R]`，int64 | 必填 | 尚未压缩的尾部 token 数。 |
| `table` | Tensor `[M,C]`，int32 | 必填 | 物理池化页号；至少 R 行、一列。 |
| `impl` | str | `"auto"` | `auto`、`triton`、`torch_native`。 |

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| 六个张量 | tuple | 顺序和类型与输入相同；新增一个 token／请求／页表行，末尾 end 为 T+1。 |

缓存块零和每个空历史请求的首块必须可读；多余页表行也须包含有效页号。
每次调用独立分配输出，不同捕获图不共享可变工作区。

## triton 后端

一次 launch 完成拷贝和准备，非连续输入先转为连续布局。
Kernel 沿用 vllm-ascend 实现并保留原许可声明。

## torch_native 后端

使用独立的 PyTorch 拼接、填零和长度调整。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import prepare_pool_key_indexer_inputs

q = torch.zeros(2, 32, 128, dtype=torch.bfloat16, device="npu")
w = torch.ones(2, 32, dtype=q.dtype, device=q.device)
ends = torch.tensor([2], dtype=torch.int64, device=q.device)
lengths = torch.zeros_like(ends)
tails = torch.full_like(ends, 2)
table = torch.zeros(1, 1, dtype=torch.int32, device=q.device)
safe = prepare_pool_key_indexer_inputs(q, w, ends, lengths, tails, table, impl="triton")
```
