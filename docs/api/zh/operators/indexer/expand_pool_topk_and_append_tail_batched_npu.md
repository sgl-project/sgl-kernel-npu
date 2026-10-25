# expand_pool_topk_and_append_tail_batched_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

公共名称没有前导下划线；后端实现名为 `_expand_pool_topk_and_append_tail_batched_npu`。

```python
from sgl_kernel_npu.operators.indexer import (
    expand_pool_topk_and_append_tail_batched_npu,
)

def expand_pool_topk_and_append_tail_batched_npu(
    pool_indices,
    seq_lens_per_token,
    n_real,
    num_q_padded,
    pool_size,
    index_topk,
    *,
    impl="auto",
):
    ...
```

## 功能说明

将每个 pool 级 top-k 索引 `p` 展开为 token 索引 `p * pool_size, ..., p * pool_size + pool_size - 1`，再为每个有效 query 行追加未压缩的 tail。

对 `seq_len = seq_lens_per_token[row]`，tail 从 `(seq_len // pool_size) * pool_size` 开始，包含 `seq_len % pool_size` 个索引。未使用的 tail 位置以及有效行之后的输出行均填入 `-1`。

## 参数

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 选中的 pool 位置；行 stride 会传给 kernel。 |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 每个 query 行的请求长度；使用连续的一维存储。 |
| `n_real` | `int` | — | — | 必填 | 请求的有效行数；后端启动前会将其截断到两个输入的较小行数。 |
| `num_q_padded` | `int` | — | — | 必填 | 输出行数，必须不小于有效 `n_real`。 |
| `pool_size` | `int` | — | — | 必填 | 一个 pool 表示的 token 数；使用有效的正 pool 宽度。 |
| `index_topk` | `int` | — | — | 必填 | token 级 top-k，必须等于 `N_POOL_TOPK * pool_size`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | 展开索引后接 tail slot；不修改输入。 |

后端按 8 个 int32 列的倍数分配物理行宽，再返回逻辑列数的切片。因此，调用方不能假定返回视图连续。

## triton 后端

- 检查二维/int32 的 `pool_indices`、一维/int32 的 `seq_lens_per_token`，以及 `index_topk == N_POOL_TOPK * pool_size`。使用 `BLOCK_M = 4`，并屏蔽有效 `n_real` 之后的行。
- `auto` 是 `triton` 的别名，不进行动态选择或 fallback。导入 registry 仍会执行 ACL 架构检测。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import (
    expand_pool_topk_and_append_tail_batched_npu,
)

pool_indices = torch.tensor(
    [[0, 1], [0, 0]], device="npu", dtype=torch.int32
)
seq_lens = torch.tensor([10, 0], device="npu", dtype=torch.int32)
out = expand_pool_topk_and_append_tail_batched_npu(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=8,
)

print(out.shape)  # torch.Size([2, 11])；第 1 行为 -1 填充行
```
