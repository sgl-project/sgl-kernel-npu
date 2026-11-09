# update_oe_token_table

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | 支持实现 | 不支持 |
| `torch_native` | 独立参考实现 | 不支持 |

构建和安装步骤见[公共构建指南](../../../../../python/sgl_kernel_npu/README.md)。
此表表示实现范围，不表示本次新增的硬件验证结果。

## 函数原型

```python
from sgl_kernel_npu.operators.embedding import update_oe_token_table

def update_oe_token_table(
    tokens, req_lens, row_indices, column_starts, ignore_tokens,
    batch_size, max_context_len, oe_token_table, *, impl="auto",
):
    ...
```

`auto` 选择 `ascendc`，调用 `torch.ops.npu.update_oe_token_table`。
不自动回退，也不转换输入 dtype。

## 功能说明

将 packed 当前 token **原地写入** LongCat2 over-embedding 历史表。
对请求 `r`，令 `s_r = sum(req_lens[:r])`。对 `0 <= j < req_lens[r]`：

```text
x = tokens[s_r + j]
oe_token_table[row_indices[r], column_starts[r] + j] = -1 if x in ignore_tokens else x
```

只修改本次写入区间。区间外的已有历史保持不变，即使其中包含 ignore token ID。
`ignore_tokens` 为空时不执行替换；长度为零的请求不写入。

负值 sentinel 在 [compute_n_gram_ids](compute_n_gram_ids.md) 中表示历史边界。
计算当前 token 的 ID 前应先执行本次更新。

## 参数与返回值

所有 Tensor 输入沿用连续、位于同一 NPU 的调用约定。`max_context_len` 使用历史表行宽，
原生 kernel 以此作为行跨度。调用方提供有效且互不冲突的请求写入区间，以及与 packed
tokens 一致的请求长度。这些是已有调用约定，不是公共包装层新增的检查。

| 参数 | 类型 / Shape / Dtype | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tokens` | Tensor `[T]`，int32 | 必填 | packed 当前 token，`T = sum(req_lens)`。 |
| `req_lens` | Tensor `[B]`，int32 | 必填 | 各请求 token 数，不是累计结束位置。 |
| `row_indices` | Tensor `[B]`，int64 | 必填 | 各请求的目标历史行。 |
| `column_starts` | Tensor `[B]`，int32 | 必填 | 各请求的写入起始列。 |
| `ignore_tokens` | Tensor `[I]`，int32 | 必填 | 本次写入时替换为 `-1` 的 token ID；允许 `I=0`。 |
| `batch_size` | int | 必填 | `B > 0`。 |
| `max_context_len` | int | 必填 | 原生寻址使用的正历史行宽。 |
| `oe_token_table` | Tensor `[R, max_context_len]`，int32 | 必填 | 可变历史表。 |
| `impl` | str | `"auto"` | 仅限关键字：`auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 / Shape / Dtype | 说明 |
| --- | --- | --- |
| 更新后的表 | Tensor `[R, max_context_len]`，int32 | 与输入 `oe_token_table` 共享存储、位于同一 NPU，不返回替代的新表。 |

## ascendc 后端

沿用官方独立 CANN 版本的 host、tiling 和 kernel，通过本仓库的 Torch adapter 调用
`aclnnSglUpdateOeTokenTableGetWorkspaceSize` 和 `aclnnSglUpdateOeTokenTable`。将原表传入
ACLNN 的输出参数，并返回同一表；Torch schema 以同一 `Tensor(a!)` alias 标记
`oe_token_table` 和返回值。CANN 管理 tiling 与 kernel 调度，不再使用旧手写 tiling cache。
使用根目录 REFACTOR 构建；arch22 的 `--ops all` 包含此算子。公共包装层不增加
输入归一化或校验。

## torch_native 后端

使用 PyTorch 独立计算 packed token 位置、替换 ignore token 并更新输入表，
保持相同 alias 与写入区间语义。精度对比时，两种实现应使用各自独立的初始表。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.embedding import update_oe_token_table

device = "npu:0"
torch.npu.set_device(device)
table = torch.full((2, 8), -1, device=device, dtype=torch.int32)
tokens = torch.tensor([5, 9, 6], device=device, dtype=torch.int32)
lengths = torch.tensor([2, 1], device=device, dtype=torch.int32)
rows = torch.tensor([1, 0], device=device, dtype=torch.int64)
starts = torch.tensor([0, 2], device=device, dtype=torch.int32)
ignored = torch.tensor([9], device=device, dtype=torch.int32)
updated = update_oe_token_table(
    tokens, lengths, rows, starts, ignored, 2, 8, table, impl="ascendc",
)
assert updated.data_ptr() == table.data_ptr()
# 第 1 行第 0:2 列写入 [5, -1]，第 0 行第 2 列写入 6。
# 使用 impl="torch_native" 选择独立参考实现。
```
