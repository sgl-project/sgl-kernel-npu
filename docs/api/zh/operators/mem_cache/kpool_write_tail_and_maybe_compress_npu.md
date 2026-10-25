# kpool_write_tail_and_maybe_compress_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_write_tail_and_maybe_compress_npu,
)

def kpool_write_tail_and_maybe_compress_npu(
    index_k_cache,
    key,
    score,
    tail_k,
    tail_score,
    ape,
    req_pool_indices,
    write_start,
    tail_logical_start,
    write_loc,
    out_cache_loc,
    num_draft_tokens,
    effective_n_per_batch=None,
    *,
    num_programs=None,
    impl="auto",
):
    ...
```

## 功能说明

将 draft token 的 key/score 写入请求本地 tail 环形缓冲，并压缩新填满的 pool。输入行按 `[B, N, 128]` 分组，其中 `N = num_draft_tokens`。对每个未跳过的 batch 项，全部 `N` 行都会写入其 tail 环形缓冲。

对 `write_start + effective_n` 跨过的 pool 边界，按通道对 `tail_score + ape` 做 softmax 归约，并将所得 BF16 key 写入 `write_loc` 指定的 cache 行。未提供 `effective_n_per_batch` 时，`effective_n = N`；提供后，它只影响 pool 是否填满的判断，不改变实际写入 tail 的行数。

## 参数

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
| `num_programs` | `int` 或 `None` | — | — | `None` | 仅限关键字参数；Triton grid 大小，显式提供时必须为正且会截断到 `B`。默认使用可用 Vector Core 数并截断到 `B`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值与原地状态

返回 `None`。`index_k_cache`、`tail_k`、`tail_score` 可变，其余参数只读。`key` 为空时直接返回，不启动 kernel。

所有 tensor 必须位于同一 NPU，最终传给 kernel 的输入必须连续。可变状态必须预先连续；后端会在内部转换只读 forward 和 plan tensor。plan 取值、源位置和 cache 位置必须由上游保证有效。

## triton 后端

- 固定通道宽度为 128；pool 归约使用 FP32，结果以 BF16 写入，不做 Hadamard 或 FP8 处理。
- `auto` 是 `triton` 的别名，不进行动态选择或 fallback。导入 registry 仍会执行 ACL 架构检测。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_write_tail_and_maybe_compress_npu,
)

b, n, pool_size, h = 1, 2, 2, 128
cache = torch.zeros(1, h, device="npu", dtype=torch.bfloat16)
key = torch.randn(b * n, h, device="npu", dtype=torch.bfloat16)
score = torch.zeros(b * n, h, device="npu", dtype=torch.float32)
tail_k = torch.zeros(1, pool_size + n, h, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(1, pool_size + n, h, device="npu", dtype=torch.float32)

kpool_write_tail_and_maybe_compress_npu(
    cache,
    key,
    score,
    tail_k,
    tail_score,
    ape=torch.zeros(pool_size, h, device="npu", dtype=torch.float32),
    req_pool_indices=torch.tensor([0], device="npu", dtype=torch.int64),
    write_start=torch.tensor([0], device="npu", dtype=torch.int32),
    tail_logical_start=torch.tensor([0], device="npu", dtype=torch.int32),
    write_loc=torch.tensor([[0]], device="npu", dtype=torch.int64),
    out_cache_loc=torch.ones(b * n, device="npu", dtype=torch.int32),
    num_draft_tokens=n,
)
```
