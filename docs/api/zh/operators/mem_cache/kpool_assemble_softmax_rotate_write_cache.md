# kpool_assemble_softmax_rotate_write_cache

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache

def kpool_assemble_softmax_rotate_write_cache(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
    *,
    num_programs: Optional[int] = None,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

按照计划，从请求的 tail 环形缓冲区取出前段数据，再从当前 chunk 补齐一个 pool。将 score 与位置分数相加后，逐通道进行 softmax 加权汇聚，并把 BF16 结果写入指定 cache 行。

汇聚使用 FP32 计算。写入掩码可跳过指定 cache 行，tail 和 chunk 输入只读。写入的是未做旋转或量化的 pooled key。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `index_k_cache` | `torch.Tensor` | 连续存储，可展平为 `[cache_rows, 128]` | `torch.bfloat16` | 必填 | 原地更新的 index cache。 |
| `chunk_k` | `torch.Tensor` | `[N, 128]` | `torch.bfloat16` | 必填 | 当前 chunk 的 key。 |
| `chunk_score` | `torch.Tensor` | 与 `chunk_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `chunk_k` 配对的 score。 |
| `tail_k` | `torch.Tensor` | `[R, T, 128]` | `torch.bfloat16` | 必填 | 请求本地 key 环形缓冲；只读。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | `torch.float16`、`torch.bfloat16` 或 `torch.float32` | 必填 | 与 `tail_k` 配对的 score。 |
| `req_pool_idx` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | `P` 个 pool 各自对应的请求/环形缓冲行。 |
| `n_from_tail` | `torch.Tensor` | `[P]` | `torch.int32` 或 `torch.int64` | 必填 | 每个 pool 从 tail 读取的前导 slot 数。 |
| `chunk_src_start` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | 每个 pool 后缀在 chunk 中的起始行。 |
| `tail_logical_base` | `torch.Tensor` | `[P]` | `torch.int32` 或 `torch.int64` | 必填 | 逻辑 tail 起始 slot；物理位置按 `T` 取模。 |
| `ape` | `torch.Tensor` | `[pool_size, 128]` | `torch.float32` | 必填 | pool 位置的加性 score；`pool_size > 0`。 |
| `loc` | `torch.Tensor` | `[P]` | `torch.int64` | 必填 | 待写入的扁平 cache 行号。 |
| `write_mask` | `torch.Tensor` 或 `None` | `[P]` 或 — | `torch.bool` | `None` | 可选的逐 pool 写使能。 |
| `num_programs` | `int` 或 `None` | — | — | `None` | 可选执行并行度参数，显式提供时必须为正数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。要求 K>0、T>0，且 0<=n_from_tail<=K。

## triton 后端

通道数固定为128。cache、tail和计划字段必须连续；chunk key/score、APE、位置及掩码可在内部转为连续布局。

## torch_native 后端

独立 gather、FP32 softmax 与 cache 写回；只读取实际选中的源分支，支持全 tail/空 chunk。num_programs 不参与此后端运算。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_assemble_softmax_rotate_write_cache
cache = torch.zeros(1, 128, device='npu', dtype=torch.bfloat16)
chunk_k = torch.randn(4, 128, device='npu', dtype=torch.bfloat16)
chunk_score = torch.zeros(4, 128, device='npu', dtype=torch.float32)
tail_k = torch.randn(1, 4, 128, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device='npu', dtype=torch.float32)
ape = torch.zeros(4, 128, device='npu', dtype=torch.float32)
kpool_assemble_softmax_rotate_write_cache(
    cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device='npu', dtype=torch.int64),
    n_from_tail=torch.tensor([2], device='npu', dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device='npu', dtype=torch.int64),
    tail_logical_base=torch.tensor([0], device='npu', dtype=torch.int32),
    ape=ape,
    loc=torch.tensor([0], device='npu', dtype=torch.int64),
    impl='torch_native',
)
```
