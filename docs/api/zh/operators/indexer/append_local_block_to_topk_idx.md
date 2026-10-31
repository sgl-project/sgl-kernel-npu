# append_local_block_to_topk_idx

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

本算子只提供 `triton` 与 `torch_native` 两种实现，两者在 arch22、arch35 上均支持。

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

def append_local_block_to_topk_idx(
    topk_idx,
    seq_lens,
    block_size,
    num_blocks,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 当前指向同一入口，不会在执行失败后自动切换实现；`torch_native` 是独立的
PyTorch 参考实现，用于精度对比。

## 功能说明

一句话：把当前 query 归属的 block id（下称 local）追加到稀疏注意力已经选出的 topk 个 block 后面。

`topk_idx` 是稀疏注意力已经选好的块号，形状 `[N, B, topk]`；本算子给每一行再补一个“query 自己
所在的块”，输出 `[N, B, topk + 1]`。对每个 `(n, b)`，记 `seq_len = seq_lens[b]`：

1. query 在序列里的位置：`q_pos = max(seq_len - 1, 0)`；
2. query 归属的块：`local = min(q_pos // block_size, num_blocks - 1)`；`num_blocks` 是块号上界（不含），
   超过就夹到最后一个块。保底要求 `num_blocks >= ceil(max_seq_len / block_size)`，否则 local 会被夹小；
3. 过滤输入候选：块号 `c` 同时满足 `0 <= c < num_blocks` 和 `c * block_size <= q_pos` 才保留，
   否则写 `-1`；这一步去掉越界的块和排在 query 后面的块；
4. 把 `local` 放到最后一列：过滤后的候选里已经有它，就写 `-1`（避免重复），否则写 `local`。

补充说明：

- 前 `topk` 列保持输入顺序；重复的候选原样保留，只有追加的 `local` 会去重；
- `num_blocks` 是块号上界（不含），不是 page table 容量，也不是每行实际的块数；SGLang 调用点传的是由
  `max_seqlen` 推导的 `max_blocks = ceil(max_seqlen / block_size)`；
- `seq_len = 0` 时 `q_pos = 0`、`local = 0`，只有块 0 能通过过滤；
- `topk` 必须大于 0；
- 输入不会被修改，返回新张量。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `topk_idx` | `torch.Tensor` | `[N, B, topk]` | `torch.int32` | 必填 | 待校验的块号候选；`N` 为头数（块稀疏注意力调用点传 `num_kv_heads`）。要求连续；Triton 要求位于 NPU，`torch_native` 不限设备；且 `N > 0`、`B > 0`、`topk > 0`。 |
| `seq_lens` | `torch.Tensor` | `[B]` | `torch.int32`、`torch.int64` | 必填 | 每行的 `seq_len`（该 query 能看到多少 KV token，含自身），取值应在 `[0, INT32_MAX]`；要求连续、与 `topk_idx` 同设备，长度严格等于 `B`。 |
| `block_size` | `int` | — | — | 必填 | 每个 KV block 的 token 数，必须为正。 |
| `num_blocks` | `int` | — | — | 必填 | 逻辑块号上界（不含），必须为正，且不小于 `ceil(max_seq_len / block_size)`；不是每行实际块数，也不是 page table 容量。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, B, topk + 1]` | `torch.int32` | 新建输出，与输入位于同一设备，不修改输入。 |

入口校验 9 条断言：`topk_idx` 三维/`int32`/连续/`topk > 0`、`seq_lens` 一维/长度等于 `B`/连续、
`block_size`/`num_blocks` 为正；违反时抛 `AssertionError`，不会回退到其他实现。
`seq_lens` 的 dtype（`int32`/`int64`）、两输入同设备、空 batch 属于调用约定，入口不额外校验，
由调用方保证。

## triton 后端

- 使用 `_append_local_block_to_topk_idx_kernel`，每个 program 处理 `BSQ = 16` 行，`grid` 为
  `(cdiv(B, 16), N)`，`BLOCK_SIZE_T` 由 `triton.next_power_of_2(topk)` 通过 heuristics 注入，
  `num_warps = 4`、`num_stages = 1`。
- 按传入的 stride 读取候选并写连续输出；要求输入连续。
- 候选合法性中的 `candidate * block_size` 为 int32 运算（`block_size` 为 constexpr）；
  `(num_blocks - 1) * block_size >= 2^31` 时可能溢出（例如 `block_size = 128` 需要
  `num_blocks >= 16,777,217`），该范围不在支持范围内。
- `seq_len` 按 int32 读取，超出 `[0, INT32_MAX]` 的值会被截断，不在支持范围。

## torch_native 后端

- 使用 PyTorch 张量运算整段向量化实现，校验算术用 int64，支持同样的输入约束，同样不修改输入。
- 与 Triton 的差异都在越界输入上：候选乘积 int32 溢出、`seq_len` 超出 int32 范围；
  支持范围内两者逐位一致。

## 调用示例

示例需要已安装包含本接口的 wheel 及 NPU 运行依赖。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.indexer import append_local_block_to_topk_idx

# seq_len = 513 -> q_pos = 512, local block = min(512 // 128, 7) = 4.
topk_idx = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32, device="npu")
seq_lens = torch.tensor([513], dtype=torch.int32, device="npu")

print(list_impls(append_local_block_to_topk_idx))  # ['auto', 'torch_native', 'triton']
y_default = append_local_block_to_topk_idx(topk_idx, seq_lens, 128, 8)
y_native = append_local_block_to_topk_idx(topk_idx, seq_lens, 128, 8, impl="torch_native")
print(y_default.tolist())  # [[[0, 1, 2, 3, 4]]]
torch.testing.assert_close(y_default, y_native, rtol=0, atol=0)
```
