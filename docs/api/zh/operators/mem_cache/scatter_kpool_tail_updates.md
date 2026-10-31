# scatter_kpool_tail_updates

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates

def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
    *,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

将 chunk 中选定的连续 key 和 score 行复制到各请求的 tail 环形缓冲区。每条计划指定请求、源起点、目标起点和复制行数，写到环形缓冲区末尾后从头继续。

只修改选中的 tail 位置，直接复制数据，不进行汇聚或量化。调用方需避免多个计划写入相同位置产生冲突。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `pool` | 对象 | — | 必须提供整数 `index_kpool` 属性 | 必填 | pool 配置；wrapper 读取 `pool.index_kpool` 作为 `pool_size`。 |
| `chunk_k` | `torch.Tensor` | `[N, 128]` | 浮点；wrapper 未显式检查 | 必填 | 源 key；与 `tail_k` 使用相同 dtype。 |
| `chunk_score` | `torch.Tensor` | 与 `chunk_k` 相同 | 浮点；wrapper 未显式检查 | 必填 | 源 score；与 `tail_score` 使用相同 dtype。 |
| `tail_k` | `torch.Tensor` | `[R, T, 128]` | 浮点；必须与 `chunk_k` 兼容 | 必填 | 目标 key 环形缓冲；原地更新。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | 浮点；必须与 `chunk_score` 兼容 | 必填 | 目标 score 环形缓冲；原地更新。 |
| `req_pool_idx` | `torch.Tensor` | `[P]` | 整数；wrapper 未显式检查 | 必填 | 每个 plan 行对应的目标请求/环形缓冲行。 |
| `dst_logical_start` | `torch.Tensor` | `[P]` | 整数；wrapper 未显式检查 | 必填 | 每个 plan 行的第一个逻辑目标 slot。 |
| `chunk_src_start` | `torch.Tensor` | `[P]` | 整数；wrapper 未显式检查 | 必填 | 每个 plan 行的第一个源 chunk 行。 |
| `n_write` | `torch.Tensor` | `[P]` | 整数；wrapper 未显式检查 | 必填 | 每个 plan 行要复制的行数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。

## triton 后端

通道数固定为128。tail和一维计划必须连续，chunk key/score在内部转为连续布局。空计划直接返回。

## torch_native 后端

独立使用 PyTorch 索引和布尔掩码完成拷贝；源/目标成对使用兼容 dtype，避免隐式转换。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
from types import SimpleNamespace
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates
pool = SimpleNamespace(index_kpool=2)
chunk_k = torch.randn(2, 128, device='npu', dtype=torch.bfloat16)
chunk_score = torch.zeros(2, 128, device='npu', dtype=torch.bfloat16)
tail_k = torch.zeros(1, 4, 128, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device='npu', dtype=torch.bfloat16)
scatter_kpool_tail_updates(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device='npu', dtype=torch.int32),
    dst_logical_start=torch.tensor([0], device='npu', dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device='npu', dtype=torch.int32),
    n_write=torch.tensor([2], device='npu', dtype=torch.int32),
    impl='torch_native',
)
```
