# expand_kpool_group_and_tail_to_topk

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk

def expand_kpool_group_and_tail_to_topk(
    pool_indices: torch.Tensor,
    seq_lens_per_token: torch.Tensor,
    n_real: int,
    num_q_padded: int,
    pool_size: int,
    index_topk: int,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

将选中的 pool 索引展开为该 pool 覆盖的 token 位置，保持原有顺序，再追加当前序列末尾尚未组成完整 pool 的 token。

例如，`pool_size=4` 时，pool 索引 2 展开为 `[8, 9, 10, 11]`。未使用的 tail 位置和补齐的 query 行填为 `-1`，输入不被修改。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 选中的 pool 位置；行 stride 会传给 kernel。 |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 每个 query 行的请求长度；使用连续的一维存储。 |
| `n_real` | `int` | — | — | 必填 | 请求的有效行数；后端启动前会将其截断到两个输入的较小行数。 |
| `num_q_padded` | `int` | — | — | 必填 | 输出行数，必须不小于有效 `n_real`。 |
| `pool_size` | `int` | — | — | 必填 | 一个 pool 表示的 token 数；Triton 要求 `pool_size >= 4`。 |
| `index_topk` | `int` | — | — | 必填 | token 级 top-k，必须等于 `N_POOL_TOPK * pool_size`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | 展开索引后接 tail slot；不修改输入。 |

所有 tensor 位于同一 NPU。调用方保证输入值有效，wrapper 不完整检查所有值域。

## triton 后端

pool索引支持非连续行布局，序列长度必须连续。返回结果可能不连续。

## torch_native 后端

使用广播算术展开和按长度生成 tail/padding；保持 int32 输入及长度连续性约束。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk
pool_indices = torch.arange(512, device='npu', dtype=torch.int32).repeat(2, 1)
seq_lens = torch.tensor([2050, 0], device='npu', dtype=torch.int32)
out = expand_kpool_group_and_tail_to_topk(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=2048,
    impl='torch_native',
)
print(out.shape)
```
