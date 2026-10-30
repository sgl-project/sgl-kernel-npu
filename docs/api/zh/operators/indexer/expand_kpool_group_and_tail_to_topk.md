# expand_kpool_group_and_tail_to_topk

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

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

令 `K=pool_size`、`P=pool_indices.shape[1]`、`L_r=seq_lens_per_token[r]`：

$$O_{r,pK+j}=K I_{r,p}+j\;(0\le j<K),\quad q_r=\lfloor L_r/K\rfloor K,\quad t_r=L_r\bmod K.$$

后缀共 `K-1` 列，第 `j<t_r` 项为 `q_r+j`，其余为 `-1`。有效行数为 `min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])`，之后的填充行全为 `-1`。负 pool ID 按算术展开，不会统一变成 `-1`。输入只读，`index_topk=P*K`；调用方提供非负行数、非负合法长度与正 K。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `pool_indices` | `torch.Tensor` | `[input_rows, N_POOL_TOPK]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 选中的 pool 位置；行 stride 会传给 kernel。 |
| `seq_lens_per_token` | `torch.Tensor` | `[input_rows]`，行数须覆盖有效 `n_real` | `torch.int32` | 必填 | 每个 query 行的请求长度；使用连续的一维存储。 |
| `n_real` | `int` | — | — | 必填 | 请求的有效行数；后端启动前会将其截断到两个输入的较小行数。 |
| `num_q_padded` | `int` | — | — | 必填 | 输出行数，必须不小于有效 `n_real`。 |
| `pool_size` | `int` | — | — | 必填 | 一个 pool 表示的 token 数；使用有效的正 pool 宽度。 |
| `index_topk` | `int` | — | — | 必填 | token 级 top-k，必须等于 `N_POOL_TOPK * pool_size`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[num_q_padded, index_topk + pool_size - 1]` | `torch.int32` | 展开索引后接 tail slot；不修改输入。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。

## triton 后端

按 4 行分块、grid-stride 调度；pool_indices 支持传入 stride，长度必须连续。物理输出宽度按 8 个 int32 对齐再返回切片，结果不保证整体连续。

## torch_native 后端

使用广播算术展开和按长度生成 tail/padding；保持 int32 输入及长度连续性约束。

公共 registry 导入时检测 NPU 架构，`torch_native` 入口也需要该环境。支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk
pool_indices = torch.tensor([[0, 1], [0, 0]], device='npu', dtype=torch.int32)
seq_lens = torch.tensor([10, 0], device='npu', dtype=torch.int32)
out = expand_kpool_group_and_tail_to_topk(
    pool_indices,
    seq_lens,
    n_real=1,
    num_q_padded=2,
    pool_size=4,
    index_topk=8,
    impl='torch_native',
)
print(out.shape)
```
