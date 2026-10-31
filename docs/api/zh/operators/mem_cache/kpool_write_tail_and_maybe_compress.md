# kpool_write_tail_and_maybe_compress

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

def kpool_write_tail_and_maybe_compress(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

将一批 draft token 的 key 和 score 写入各请求的 tail 环形缓冲区，对新补齐的 pool 逐通道进行 softmax 加权汇聚，并把 BF16 结果写入 `write_loc` 指定的 cache 位置。

汇聚使用 FP32 计算，并加入位置分数。有效 draft 数量只决定哪些 pool 可以压缩；对于有效 batch，传入的全部 draft 行仍会写入 tail。无效 batch 被跳过，cache、tail key 和 tail score 原地更新。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | 连续存储，可展平为 `[cache_rows, 128]` | `torch.bfloat16` | 必填 | pool 填满时原地更新的 cache。 |
| `key` | `torch.Tensor` | `[B * N, 128]` | `torch.bfloat16` | 必填 | draft key。 |
| `score` | `torch.Tensor` | 与 `key` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `key` 配对的 score。 |
| `tail_k` | `torch.Tensor` | `[R, pool_size + N, 128]` | `torch.bfloat16` | 必填 | 原地更新的 tail key 环形缓冲。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 原地更新的 tail score 环形缓冲。 |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | 必填 | pool 位置的加性 score；`pool_size > 0`。 |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | `torch.int64` | 必填 | 每个 batch 项对应的请求/环形缓冲行。 |
| `write_start` | `torch.Tensor` | `[>= B]` | `torch.int32` 或 `torch.int64` | 必填 | 每个 batch 开始写入的逻辑 tail 位置。 |
| `tail_logical_start` | `torch.Tensor` | `[>= B]` | `torch.int32` 或 `torch.int64` | 必填 | 候选待填满 pool 的逻辑起点。 |
| `write_loc` | `torch.Tensor` | `[B, ceil(N / pool_size)]` | `torch.int64` | 必填 | 候选 cache 行；只读，必须预先为连续张量。 |
| `out_cache_loc` | `torch.Tensor` | `[>= B * N]` | `torch.int32` 或 `torch.int64` | 必填 | token cache 位置；某 batch 的首项为 0 时跳过整个 batch。只读。 |
| `num_draft_tokens` | `int` | — | — | 必填 | 静态 `N`；必须为正，且整除 `key.shape[0]`。 |
| `effective_n_per_batch` | `torch.Tensor` 或 `None` | `[>= B]` 或 — | `torch.int32` 或 `torch.int64` | `None` | 可选有效 draft 数，只用于判断 pool 是否填满。 |
| `num_programs` | `int` 或 `None` | — | — | `None` | 可选执行并行度参数，显式提供时必须为正数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。

## triton 后端

通道数固定为128。cache、tail和write_loc必须连续，其余只读输入可在内部转为连续布局。

## torch_native 后端

独立按 batch 写 tail 并做 FP32 softmax；effective 为 0 仍写所有草稿的 tail。num_programs 不参与此后端运算。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress
b, n, pool_size, h = (1, 2, 2, 128)
cache = torch.zeros(1, h, device='npu', dtype=torch.bfloat16)
key = torch.randn(b * n, h, device='npu', dtype=torch.bfloat16)
score = torch.zeros(b * n, h, device='npu', dtype=torch.float32)
tail_k = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, pool_size + n, h, device='npu', dtype=torch.float32)
kpool_write_tail_and_maybe_compress(
    cache,
    key,
    score,
    tail_k,
    tail_score,
    ape=torch.zeros(pool_size, h, device='npu', dtype=torch.float32),
    req_pool_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    write_start=torch.tensor([0], device='npu', dtype=torch.int32),
    tail_logical_start=torch.tensor([0], device='npu', dtype=torch.int32),
    write_loc=torch.tensor([[0]], device='npu', dtype=torch.int64),
    out_cache_loc=torch.ones(b * n, device='npu', dtype=torch.int32),
    num_draft_tokens=n,
    impl='torch_native',
)
```
