# fused_decode_prepare_indexer

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

def fused_decode_prepare_indexer(
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

## 功能说明

将 decode indexer 的输入准备融合为一个 Cube GEMM kernel 和一个 Vector kernel：

- `w_bf16 = (x @ weights_proj_weight.T) * (n_heads_pow * softmax_scale)`，结果转换为 BF16。
- `pool_seqlens = seqlens_32 // pool_size`。
- `pool_block_tables` 每隔 `pool_size` 列选取一个页表项；在后端内部补齐后，等价于按 `block_tables[:, ::pool_size]` 抽取。

该算子不生成 `actual_seq_lengths_q = arange(1, N + 1)`，这部分仍由调用方负责。

## 参数

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, HIDDEN_SIZE]` | `torch.bfloat16` | 必填 | decode hidden state。 |
| `weights_proj_weight` | `torch.Tensor` | `[N_HEADS, HIDDEN_SIZE]` | `torch.float32` | 必填 | ReplicatedLinear 投影权重。 |
| `seqlens_32` | `torch.Tensor` | `[N]` | `torch.int32` | 必填 | 每个请求的真实 KV 长度。 |
| `block_tables` | `torch.Tensor` | `[N, NUM_BLOCKS]` | `torch.int32` | 必填 | 连续的 token 页表。 |
| `pool_size` | `int` | — | — | `4` | pool 宽度；当前 kernel 要求严格等于 4。 |
| `n_heads_pow` | `float` | — | — | `32**-0.5` | 投影组合缩放系数的第一个因子。 |
| `softmax_scale` | `float` | — | — | `128**-0.5` | 投影组合缩放系数的第二个因子。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `w_bf16` | `torch.Tensor` | `[N, N_HEADS]` | `torch.bfloat16` | 缩放后的投影结果。 |
| `pool_seqlens` | `torch.Tensor` | `[N]` | `torch.int32` | 整除后的 pool 序列长度。 |
| `pool_block_tables` | `torch.Tensor` | `[N, N_POOL]` | `torch.int32` | 新建的 pool 页表。 |

令 `PADDED_BLOCKS = max(pool_size, ceil(NUM_BLOCKS / pool_size) * pool_size)`，则 `N_POOL = PADDED_BLOCKS // pool_size`。页表过窄或宽度不对齐时，后端会先内部补齐再抽取；每个请求超出 `pool_seqlens` 的表项不应被下游消费。输入不会被修改。

## triton 后端

- 要求 `x.shape[1] == weights_proj_weight.shape[1]`、`HIDDEN_SIZE % 256 == 0`、`N_HEADS <= 32` 且 `pool_size == 4`。GEMM 和向量布局要求兼容的连续输入；后端明确将 `block_tables` 约定为连续。
- `auto` 是 `triton` 的别名，不进行动态选择或 fallback。导入 registry 仍要求 NPU/ACL 架构检测可用。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import fused_decode_prepare_indexer

n, hidden, heads = 2, 256, 32
x = torch.randn(n, hidden, device="npu", dtype=torch.bfloat16)
weight = torch.randn(heads, hidden, device="npu", dtype=torch.float32)
seqlens = torch.tensor([8, 12], device="npu", dtype=torch.int32)
block_tables = torch.arange(16, device="npu", dtype=torch.int32).reshape(n, 8)

w_bf16, pool_seqlens, pool_block_tables = fused_decode_prepare_indexer(
    x, weight, seqlens, block_tables,
)
print(w_bf16.shape, pool_seqlens.shape, pool_block_tables.shape)
# torch.Size([2, 32]) torch.Size([2]) torch.Size([2, 2])
```
