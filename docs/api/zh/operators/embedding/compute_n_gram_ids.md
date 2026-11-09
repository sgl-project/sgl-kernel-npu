# compute_n_gram_ids

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | 支持实现 | 不支持 |
| `torch_native` | 独立参考实现 | 不支持 |

构建和安装步骤见[公共构建指南](../../../../../python/sgl_kernel_npu/README.md)。
此表表示实现范围，不表示本次新增的硬件验证结果。

## 函数原型

```python
from sgl_kernel_npu.operators.embedding import compute_n_gram_ids

def compute_n_gram_ids(
    oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
    tokens, exclusive_req_len_sums, oe_token_table, row_indices, column_starts,
    batch_size, oe_n, oe_k, max_context_len, *, impl="auto",
):
    ...
```

`auto` 选择 `ascendc`，调用 `torch.ops.npu.compute_n_gram_ids`。
不自动回退，也不转换输入 dtype。

## 功能说明

从**已经更新的** token table 计算 LongCat2 over-embedding ID。
`tokens` 提供输出的 token 数，其内容不会替代历史表内容。
写入新 token 时，应先调用 [update_oe_token_table](update_oe_token_table.md)。

令 `C[r] = sum(req_lens[:r+1])`，`C[-1] = 0`。虽然保留了历史名称，
`exclusive_req_len_sums` 实际存储**各请求的包含当前项的累计结束位置**，长度为 `B`，
不包含前导零。请求 `r` 的 packed token `i` 对应历史行 `row_indices[r]`，
列 `p = column_starts[r] + i - C[r-1]`。
对 n-gram 阶数 `n = 2..oe_n` 和组 `k = 0..oe_k-1`：

```text
f = (n - 2) * oe_k + k
id[i, f] = offset[f] + (sum_j ((history[p-j] * weight[n-2,k,j]) % mod[n-2,k])) % mod[n-2,k]
```

按 `j = 0..n-1` 回溯，遇到请求行起点或首个负 token 即停止。
负 token 是边界 sentinel，不是跳过后继续回溯的 token。空和对应组的 offset。

## 参数与返回值

所有 Tensor 输入沿用连续、位于同一 NPU 的调用约定。下表描述既有约定，公共包装层
只做分发。调用方负责一致地提供写入区间与 packed 请求长度。

| 参数 | 类型 / Shape / Dtype | 默认值 | 说明 |
| --- | --- | --- | --- |
| `oe_weights` | Tensor `[oe_n-1, oe_k, oe_n]`，int32 | 必填 | 模位置权重；现有模型表使用非负权重。 |
| `oe_mods` | Tensor `[oe_n-1, oe_k]`，int32 | 必填 | 每组的正模数。 |
| `exclusive_oe_embeder_size_sums` | Tensor `[(oe_n-1)*oe_k+1]`，int32 | 必填 | 含前导零与末尾总数的 embedding 表 offset；每组读取第 `f` 项。 |
| `tokens` | Tensor `[T]`，int32 | 必填 | packed 当前 token，决定输出行数。 |
| `exclusive_req_len_sums` | Tensor `[B]`，int32 | 必填 | 各请求累计结束位置；最后一项为 `T`。 |
| `oe_token_table` | Tensor `[R, max_context_len]`，int32 | 必填 | 更新后的历史；负值终止 n-gram。原生寻址的行跨度为 `max_context_len`。 |
| `row_indices` | Tensor `[B]`，int64 | 必填 | 每个请求对应的历史行。 |
| `column_starts` | Tensor `[B]`，int32 | 必填 | 每行本次首个 token 的列位置。 |
| `batch_size` | int | 必填 | `B > 0`。 |
| `oe_n` | int | 必填 | 最大 n-gram 阶数，大于 1。 |
| `oe_k` | int | 必填 | 每阶的组数，大于 0。 |
| `max_context_len` | int | 必填 | 原生寻址使用的正历史行宽。 |
| `impl` | str | `"auto"` | 仅限关键字：`auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 / Shape / Dtype | 说明 |
| --- | --- | --- |
| IDs | Tensor `[T, (oe_n-1)*oe_k]`，int32 | 同一 NPU 上的新 Tensor，按阶数、组编号排列；不修改输入。 |

## ascendc 后端

沿用官方独立 CANN 版本的 host、tiling 和 kernel，通过本仓库的 Torch adapter 调用
`aclnnSglComputeNGramIdsGetWorkspaceSize` 和 `aclnnSglComputeNGramIds`。CANN 管理 tiling
与 kernel 调度，不再使用旧直接启动路径的手写 tiling cache。公共 API 不增加输入归一化
或新的输入校验。使用根目录 REFACTOR 构建，arch22 的 `--ops all` 包含此算子。

## torch_native 后端

通过 PyTorch 独立收集历史窗口并计算加权和，用于精度参考，
保留 int32 输出和现有 sentinel 行为。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.embedding import compute_n_gram_ids

device = "npu:0"
torch.npu.set_device(device)
weights = torch.tensor([[[1, 2]]], device=device, dtype=torch.int32)
mods = torch.tensor([[7]], device=device, dtype=torch.int32)
offsets = torch.tensor([0, 7], device=device, dtype=torch.int32)
tokens = torch.tensor([2, 3], device=device, dtype=torch.int32)
ends = torch.tensor([2], device=device, dtype=torch.int32)
history = torch.tensor([[2, 3, -1, -1]], device=device, dtype=torch.int32)
rows = torch.tensor([0], device=device, dtype=torch.int64)
starts = torch.tensor([0], device=device, dtype=torch.int32)
ids = compute_n_gram_ids(
    weights, mods, offsets, tokens, ends, history, rows, starts,
    batch_size=1, oe_n=2, oe_k=1, max_context_len=4, impl="ascendc",
)
# 数学结果为 [[2], [0]]；参考实现使用 impl="torch_native"。
```
