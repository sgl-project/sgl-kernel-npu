# kpool_indexer_prepare_decode_input

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input

def kpool_indexer_prepare_decode_input(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
    *,
    impl="auto",
):
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

为 decode 索引准备三项输入：将 hidden states 投影为各 head 的权重，将序列长度转换为完整 pool 数量，并提取 pooled 页表。

投影使用 FP32 计算，应用 `n_heads_pow` 和 `softmax_scale` 缩放后输出 BF16。页表按需在末尾补零，再选取每组的第一个页 ID。输入不被修改。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, HIDDEN_SIZE]` | `torch.bfloat16` | 必填 | decode hidden state。 |
| `weights_proj_weight` | `torch.Tensor` | `[N_HEADS, HIDDEN_SIZE]` | `torch.float32` | 必填 | ReplicatedLinear 投影权重。 |
| `seqlens_32` | `torch.Tensor` | `[N]` | `torch.int32` | 必填 | 每个请求的真实 KV 长度。 |
| `block_tables` | `torch.Tensor` | `[N, NUM_BLOCKS]` | `torch.int32` | 必填 | 连续的 token 页表。 |
| `pool_size` | `int` | — | — | `4` | pool 宽度；当前 kernel 要求严格等于 4。 |
| `n_heads_pow` | `float` | — | — | `32**-0.5` | 投影组合缩放系数的第一个因子。 |
| `softmax_scale` | `float` | — | — | `128**-0.5` | 投影组合缩放系数的第二个因子。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `w_bf16` | `torch.Tensor` | `[N, N_HEADS]` | `torch.bfloat16` | 缩放后的投影结果。 |
| `pool_seqlens` | `torch.Tensor` | `[N]` | `torch.int32` | 整除后的 pool 序列长度。 |
| `pool_block_tables` | `torch.Tensor` | `[N, N_POOL]` | `torch.int32` | 新建的 pool 页表。 |

所有 tensor 位于同一 NPU。调用方保证输入值有效，wrapper 不完整检查所有值域。

## triton 后端

输入必须连续，`N_HEADS`必须为[1, 128]内的二次幂，`pool_size`必须为4。hidden宽度无需整除256。

`n_heads_pow`按调用方传值使用，默认仍为`32**-0.5`；不同head数若采用head归一化，应显式传入`N_HEADS**-0.5`。

## torch_native 后端

独立 PyTorch FP32 矩阵乘法、补零、分组抽样与 BF16 转换。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import kpool_indexer_prepare_decode_input
n, hidden, heads = (2, 256, 32)
x = torch.randn(n, hidden, device='npu', dtype=torch.bfloat16)
weight = torch.randn(heads, hidden, device='npu', dtype=torch.float32)
seqlens = torch.tensor([8, 12], device='npu', dtype=torch.int32)
block_tables = torch.arange(16, device='npu', dtype=torch.int32).reshape(n, 8)
w_bf16, pool_seqlens, pool_block_tables = kpool_indexer_prepare_decode_input(
    x,
    weight,
    seqlens,
    block_tables,
    impl='torch_native',
)
print(w_bf16.shape, pool_seqlens.shape, pool_block_tables.shape)
```
