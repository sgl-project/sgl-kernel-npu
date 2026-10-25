# kpool_decode_update_and_maybe_write_cache_bf16

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import (
    kpool_decode_update_and_maybe_write_cache_bf16,
)

def kpool_decode_update_and_maybe_write_cache_bf16(
    buf,
    tail_k,
    tail_score,
    key,
    slot_score,
    ape,
    block_tables,
    req_pool_indices,
    positions,
    seq_lens,
    out_cache_loc,
    *,
    pool_size,
    slots_per_page,
    head_dim=128,
    impl="auto",
):
    ...
```

## 功能说明

为有效 decode 行更新请求本地的 BF16 tail key 和 score 环形缓冲。有效行要求请求索引在范围内、`out_cache_loc[row] != 0`，且 `0 <= positions[row] < seq_lens[row]`。

当前 token 填满一个 pool 时，kernel 对该 pool 的 `score + ape` 按通道做 softmax，将加权得到的 BF16 key 写入扁平化 index cache。cache 页通过 `block_tables` 定位。该实现不做 Hadamard 变换、FP8 量化，也不输出逐 slot scale。

## 参数

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[num_pages, slots_per_page * head_dim]` | `torch.bfloat16` | 必填 | 扁平化 index cache；pool 填满时原地更新。 |
| `tail_k` | `torch.Tensor` | `[R, T, head_dim]` | `torch.bfloat16` | 必填 | 请求本地 key 环形缓冲；有效行原地更新。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | 按后端约定为 `torch.float32` | 必填 | 与 `tail_k` 配对的 score；有效行原地更新。 |
| `key` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` | 必填 | 当前 decode key。 |
| `slot_score` | `torch.Tensor` | `[B, head_dim]` | 按后端约定为 `torch.float32` | 必填 | 与 `key` 配对的 score。 |
| `ape` | `torch.Tensor` | `[pool_size, head_dim]` | `torch.float32` | 必填 | pool 位置的加性 score。 |
| `block_tables` | `torch.Tensor` | `[B, BLOCK_TABLE_COLS]` | `torch.int32` | 必填 | token 页表，用于定位已填满 pool 的 cache 页。 |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项对应的请求/环形缓冲行。 |
| `positions` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项的当前 token 位置。 |
| `seq_lens` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项的序列长度。 |
| `out_cache_loc` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | token cache 位置；某行对应项为 0 时跳过该行更新。只读。 |
| `pool_size` | `int` | — | — | 必填 | 仅限关键字参数；一个 pool 的 token 数。 |
| `slots_per_page` | `int` | — | — | 必填 | 仅限关键字参数；每个扁平 cache 页的 pool slot 数。 |
| `head_dim` | `int` | — | — | `128` | 仅限关键字参数；通道宽度，必须等于 tail 的最后一维。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

## 返回值与原地状态

返回 `None`。`buf`、`tail_k`、`tail_score` 是可变状态，其余 tensor 只读。空 batch 直接返回，不启动 kernel。

所有 tensor 应位于同一 NPU 且连续。后端会对输入调用 `.contiguous()`；非连续的可变输入可能被复制，从而无法让调用方看到原地更新。元数据取值和页位置必须由上游保证有效，wrapper 只检查部分 dtype 和 shape 条件。

## triton 后端

- 要求上述 BF16 cache/key 布局；pool 归约使用 FP32，结果转换为 BF16。
- `auto` 与 `triton` 注册到同一入口，不是动态选择或 fallback。导入 registry 仍会执行 ACL 架构检测。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import (
    kpool_decode_update_and_maybe_write_cache_bf16,
)

b, r, t, h = 1, 1, 4, 128
buf = torch.zeros(1, 8 * h, device="npu", dtype=torch.bfloat16)
tail_k = torch.zeros(r, t, h, device="npu", dtype=torch.bfloat16)
tail_score = torch.zeros(r, t, h, device="npu", dtype=torch.float32)
key = torch.randn(b, h, device="npu", dtype=torch.bfloat16)
slot_score = torch.zeros(b, h, device="npu", dtype=torch.float32)
ape = torch.zeros(4, h, device="npu", dtype=torch.float32)
block_tables = torch.zeros(b, 1, device="npu", dtype=torch.int32)
metadata = dict(
    req_pool_indices=torch.zeros(b, device="npu", dtype=torch.int32),
    positions=torch.tensor([3], device="npu", dtype=torch.int32),
    seq_lens=torch.tensor([4], device="npu", dtype=torch.int32),
    out_cache_loc=torch.ones(b, device="npu", dtype=torch.int32),
)

kpool_decode_update_and_maybe_write_cache_bf16(
    buf, tail_k, tail_score, key, slot_score, ape, block_tables,
    **metadata, pool_size=4, slots_per_page=8,
)
```
