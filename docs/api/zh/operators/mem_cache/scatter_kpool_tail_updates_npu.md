# scatter_kpool_tail_updates_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

def scatter_kpool_tail_updates_npu(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    dst_logical_start,
    chunk_src_start,
    n_write,
    *,
    impl="auto",
):
    ...
```

## 功能说明

将 chunk 中的 key/score 行散射写入请求本地 tail 环形缓冲。对 plan 行 `r` 和 `j < n_write[r]`，源行为 `chunk_src_start[r] + j`；目标请求为 `req_pool_idx[r]`，物理 tail slot 为 `(dst_logical_start[r] + j) % T`。

## 参数

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
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值与原地状态

返回 `None`，只修改 `tail_k` 和 `tail_score`。`P = req_pool_idx.shape[0]`，`P == 0` 时不执行任何操作。`pool`、chunk tensor 和所有 plan tensor 只读。

wrapper 不完整检查 shape、dtype、设备或取值范围。调用方必须提供非负写入计数以及有效的源、请求和 slot 范围。`chunk_k`、`chunk_score` 会在内部转换为连续布局；可变 tail 缓冲必须满足 kernel 布局要求，包括通道维连续，一维 plan tensor 也必须连续。所有 tensor 必须位于同一 NPU。

## triton 后端

- 使用固定 128 通道布局，并以 grid-stride 循环处理 plan 行。kernel 直接复制原生元素，不做 BF16 转换；key 两侧应使用相同 dtype，score 两侧也应使用相同 dtype。该算子后端不暴露 `num_programs` 参数。
- `auto` 是 `triton` 的别名，不是动态选择或 fallback。导入 registry 仍会执行 ACL 架构检测。

## 调用示例

```python
from types import SimpleNamespace

import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

pool = SimpleNamespace(index_kpool=2)
chunk_k = torch.randn(2, 128, device="npu", dtype=torch.bfloat16)
chunk_score = torch.zeros(2, 128, device="npu", dtype=torch.bfloat16)
tail_k = torch.zeros(1, 4, 128, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device="npu", dtype=torch.bfloat16)

scatter_kpool_tail_updates_npu(
    pool,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device="npu", dtype=torch.int32),
    dst_logical_start=torch.tensor([0], device="npu", dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device="npu", dtype=torch.int32),
    n_write=torch.tensor([2], device="npu", dtype=torch.int32),
)
```
