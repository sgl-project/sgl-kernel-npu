# kpool_assemble_softmax_write_cache_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_assemble_softmax_write_cache_npu,
)

def kpool_assemble_softmax_write_cache_npu(
    index_k_cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    n_from_tail,
    chunk_src_start,
    tail_logical_base,
    ape,
    loc,
    write_mask=None,
    *,
    num_programs=None,
    impl="auto",
):
    ...
```

## 功能说明

为每个 plan 行组装一个 pool 后的 BF16 key。该行前 `n_from_tail[row]` 个 slot 从请求本地 tail 环形缓冲读取，逻辑起点为 `tail_logical_base[row]`，物理位置按 `T` 取模；其余 slot 从 `chunk_k` 的 `chunk_src_start[row]` 开始读取。

kernel 对 pool 内各 slot 的 `score + ape` 按通道做 softmax，计算加权 key，转换为 BF16 后写入扁平 cache 的 `loc[row]` 行。`write_mask[row]` 为 false 时，对应 cache 行保持不变。

## 参数

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
| `num_programs` | `int` 或 `None` | — | — | `None` | 仅限关键字参数；Triton grid 大小，显式提供时必须为正且会截断到 `P`。默认使用可用 Vector Core 数并截断到 `P`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值与原地状态

返回 `None`，只修改 `index_k_cache`。`P = req_pool_idx.numel()`，`P == 0` 时不执行任何操作。所有 plan tensor 必须有 `P` 项；源范围和每个 pool 的 slot 计数由上游保证有效。

所有 tensor 必须位于同一 NPU，最终传给 kernel 的输入必须连续。wrapper 会在内部转换 `chunk_k`、`chunk_score`、`ape`、`loc` 和可选 `write_mask`；`index_k_cache`、两个 tail tensor 及其余 plan tensor 必须预先满足连续性断言。

## triton 后端

- 固定通道宽度为 128，pool 归约使用 FP32，结果以 BF16 存储，不做 Hadamard 或 FP8 处理。
- `auto` 是 `triton` 的别名，不进行动态选择或 fallback。导入 registry 仍要求 NPU/ACL 架构检测可用。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_assemble_softmax_write_cache_npu,
)

cache = torch.zeros(1, 128, device="npu", dtype=torch.bfloat16)
chunk_k = torch.randn(4, 128, device="npu", dtype=torch.bfloat16)
chunk_score = torch.zeros(4, 128, device="npu", dtype=torch.float32)
tail_k = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, 4, 128, device="npu", dtype=torch.float32)
ape = torch.zeros(4, 128, device="npu", dtype=torch.float32)

kpool_assemble_softmax_write_cache_npu(
    cache,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx=torch.tensor([0], device="npu", dtype=torch.int64),
    n_from_tail=torch.tensor([2], device="npu", dtype=torch.int32),
    chunk_src_start=torch.tensor([0], device="npu", dtype=torch.int64),
    tail_logical_base=torch.tensor([0], device="npu", dtype=torch.int32),
    ape=ape,
    loc=torch.tensor([0], device="npu", dtype=torch.int64),
)
```
