# kpool_decode_update_and_maybe_write_cache

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 已验证 |
| `torch_native` | 支持 | 已验证 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.mem_cache import kpool_decode_update_and_maybe_write_cache

def kpool_decode_update_and_maybe_write_cache(
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    pool_size: int,
    slots_per_page: int,
    head_dim: int = 128,
    impl="auto",
) -> None:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

将当前 decode token 的 key 和 score 写入请求的 tail 环形缓冲区。如果该 token 恰好补齐一个 pool，则结合已有 tail 历史，逐通道进行 softmax 加权汇聚，并通过页表将 BF16 结果写入 pooled cache。

汇聚使用 FP32 计算，并加入位置分数。无效请求行不修改任何状态。`out_cache_loc` 用于判断该行是否有效，pooled cache 的实际写入位置另外通过页表确定。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[num_pages, S*D]` / `[num_pages, S, 1, D]` | `torch.bfloat16` | 必填 | 原地 cache；S=slots_per_page、D=head_dim，页 stride 必须等于 S*D。 |
| `tail_k` | `torch.Tensor` | `[R, T, head_dim]` | `torch.bfloat16` | 必填 | 请求本地 key 环形缓冲；有效行原地更新。 |
| `tail_score` | `torch.Tensor` | 与 `tail_k` 相同 | `torch.bfloat16` / `torch.float32` | 必填 | 配对 score；当前测试覆盖 BF16 与 FP32，归约用 FP32。 |
| `key` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` | 必填 | 当前 decode key。 |
| `slot_score` | `torch.Tensor` | `[B, head_dim]` | `torch.bfloat16` / `torch.float32` | 必填 | 配对 score；当前测试覆盖 BF16 与 FP32，归约用 FP32。 |
| `ape` | `torch.Tensor` | `[pool_size, head_dim]` | `torch.float32` | 必填 | pool 位置的加性 score。 |
| `block_tables` | `torch.Tensor` | `[B, BLOCK_TABLE_COLS]` | `torch.int32` | 必填 | token 页表，用于定位已填满 pool 的 cache 页。 |
| `req_pool_indices` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项对应的请求/环形缓冲行。 |
| `positions` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项的当前 token 位置。 |
| `seq_lens` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | 每个 batch 项的序列长度。 |
| `out_cache_loc` | `torch.Tensor` | `[>= B]` | 整数；wrapper 未显式检查 | 必填 | token cache 位置；某行对应项为 0 时跳过该行更新。只读。 |
| `pool_size` | `int` | — | — | 必填 | 仅限关键字参数；一个 pool 的 token 数。 |
| `slots_per_page` | `int` | — | — | 必填 | 仅限关键字参数；每个扁平 cache 页的 pool slot 数。 |
| `head_dim` | `int` | — | — | `128` | 仅限关键字参数；通道宽度，必须等于 tail 的最后一维。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | 原地更新范围见功能说明。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。要求 pool_size、T、slots_per_page 为正且页表非空；仅 buf、tail_k、tail_score 被修改。

## triton 后端

buf、tail_k和tail_score必须连续，不接受非连续的可变状态；只读输入可在内部转为连续布局。buf.stride(0)须等于slots_per_page*head_dim，支持满足约束的二维或四维布局。空batch直接返回。

## torch_native 后端

独立 PyTorch 按行处理、FP32 稳定 softmax 和页表地址写回。未复制所有 Triton 输入断言，调用方仍需满足公共布局约定。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import kpool_decode_update_and_maybe_write_cache
b, r, t, h = (1, 1, 4, 128)
buf = torch.zeros(1, 8 * h, device='npu', dtype=torch.bfloat16)
tail_k = torch.zeros(r, t, h, device='npu', dtype=torch.bfloat16)
tail_score = torch.zeros(r, t, h, device='npu', dtype=torch.float32)
key = torch.randn(b, h, device='npu', dtype=torch.bfloat16)
slot_score = torch.zeros(b, h, device='npu', dtype=torch.float32)
ape = torch.zeros(4, h, device='npu', dtype=torch.float32)
block_tables = torch.zeros(b, 1, device='npu', dtype=torch.int32)
metadata = dict(
    req_pool_indices=torch.zeros(b, device='npu', dtype=torch.int32),
    positions=torch.tensor([3], device='npu', dtype=torch.int32),
    seq_lens=torch.tensor([4], device='npu', dtype=torch.int32),
    out_cache_loc=torch.ones(b, device='npu', dtype=torch.int32),
)
kpool_decode_update_and_maybe_write_cache(
    buf,
    tail_k,
    tail_score,
    key,
    slot_score,
    ape,
    block_tables,
    **metadata,
    pool_size=4,
    slots_per_page=8,
    impl='torch_native',
)
```
