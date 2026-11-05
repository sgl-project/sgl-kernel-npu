# flash_prefill_bnsd_indexer

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 默认选择 `triton`；省略 `impl` 与此相同。运行时不会自动回退到其他后端。

```python
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

def flash_prefill_bnsd_indexer(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    topk,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    per_query_seq_lens=None,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在 prefill 阶段用一次调用完成三件事：为每个 query 计算各 KV 逻辑块的分数，计算每个索引头的
causal 索引注意力输出 `idx_o`，并按块分数选出每个 query 的 top-k 块号。

request `r` 的打包 query 行范围为 `cu_seqlens[r]:cu_seqlens[r + 1]`。令
`P = block_size_k`、`L_r = seq_lens[r]`。逻辑块号满足 `0 <= b < ceil(L_r / P)`，块内位置为
`p = b * P + j`（`0 <= j < P`）。索引头 `h` 使用 KV 头 `floor(h / gqa_group_size)`，其中
`gqa_group_size = num_idx_heads / num_kv_heads`。

**可见范围。** 当前实现只对 `0 <= p < L_r` 且 `p <= t` 的 key 打分，
其中 `t` 是打包后的全局 query 行号，不是 request 内行号，也不会加上已有 KV 前缀长度。
`per_query_seq_lens` 只用于后续选块，不改变这一掩码；`idx_o` 使用同一掩码。
因此，有 KV 前缀或多个 request 时，不能将其视为按 request 内位置计算的标准 causal attention。

对 query 打包行 `t` 与 key 位置 `p`，先算 log2 域 logits：

```math
x_{t,p} = (q_t \cdot k_p)\,\text{sm\_scale}\,\log_2 e
```

块分数按 `score_type` 归约，以下公式中的 `p` 仅遍历该块内的可见 key（省略头下标）：

```math
\text{score}[h,t,b] =
\begin{cases}
\max_{p \in b} x_{t,p}, & \text{score\_type} = \text{"max"} \\
\log_2 \sum_{p \in b} 2^{x_{t,p}}, & \text{score\_type} = \text{"lse"}
\end{cases}
```

块内没有可见 key 时该块分数为 `-inf`。分数仅用于选块，不单独返回。
`idx_o` 使用所有可见 key 的 `QKᵀ * sm_scale` 做 softmax，再对 V 加权求和；它不受
`score_type` 和 `topk` 影响。

**选块规则。** 令 `K = ceil(max(seq_lens) / P)`，块号是各 request 内从 0 开始的逻辑块号，
不是物理页号。有两条路径：

- **融合分支**：`per_query_seq_lens` 非 `None` 且 `num_idx_heads == num_kv_heads`。
  对 query 行 `t`，令 `query_pos = max(per_query_seq_lens[t] - 1, 0)`，
  `local_t = min(query_pos // P, K - 1)`；local block 指 query 所在的逻辑 KV 块。
  每轮从尚未选择的全部块中取最大分数，同分取更小块号。选中块的分数大于 `-inf` 且
  `块号 * P <= query_pos` 时写入块号，否则该槽写 `-1`，不会为该槽另选一个块。
  共执行 `topk` 轮，随后在末槽追加 `local_t`；若它已经选中，末槽写 `-1`。
- **回退分支**：其余情况。对 `score` 的全部 `K` 列调用 `torch.topk` 取前
  `min(topk, K)` 个块号；`topk > K` 时尾部补 `-1`。该分支不剔除分数为 `-inf` 的列，也
  **不**追加 local block。

因此 `topk_idx` 的宽度由分支决定：融合分支为 `topk + 1`，回退分支为 `topk`。
回退分支同分时的块号顺序由 `torch.topk` 决定，不保证取较小块号。

## 参数与返回值

`bs` 为 request 数，`total_q` 为打包后的 query 总行数。以下 Tensor shape 均按实际存储顺序列出。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`、`bfloat16` | 必填 | 索引 query。Triton 后端要求位于 NPU。`num_idx_heads` 必须能被 KV 头数整除（被断言）。 |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | 与 `q` 相同 | 必填 | 分页索引 K cache。`page_size == block_size_k`；物理页号必须有效（调用方约定）。 |
| `v_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | 与 `k_cache_bnsd` 相同 | 必填 | 分页索引 V cache；与 K 使用同一 block table 和 KV 头。 |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` 或 `int64` | 必填 | 各 request 打包 query 行的前缀和；从 0 开始、单调不减，末项等于 `total_q`（调用方约定）。 |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | 每个 request 的 KV 总长度，决定块数与分数宽度。`bs > 0` 由调用方保证；有 query 的 request 应有正 KV 长度。 |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` 或 `int64` | 必填 | request token 位置到 token slot 的映射；仅在 `qblock_mappings` 为 `None` 时使用。逻辑块 `b` 的物理页号为 `req_to_token[req_pool_indices[r], b * block_size_k] // block_size_k`；列宽需覆盖读取的块起始列。 |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` 或 `int64` | 必填 | `req_to_token` 中的行号；仅在 `qblock_mappings` 为 `None` 时使用。 |
| `block_size_q` | `int` | — | — | 必填 | 正数 query 分组大小；允许非 2 的幂。 |
| `block_size_k` | `int` | — | — | 必填 | 每个 KV 逻辑块包含的 token 数，必须为正数、等于 `page_size`，且为 2 的幂（调用方约定）。 |
| `topk` | `int` | — | — | 必填 | 选块槽数。融合分支要求 `topk > 0`（被断言）；回退分支允许 `topk == 0`，返回零宽结果。 |
| `sm_scale` | `float` | — | — | 必填 | logits 缩放系数，通常为 `head_dim ** -0.5`。 |
| `score_type` | `str` | — | — | `"max"` | 块分数归约方式：`"max"` 或 `"lse"`；其他取值触发 `AssertionError`。只影响分数，不影响 `idx_o`。 |
| `qblock_mappings` | `tuple` 或 `None` | — | — | `None` | 可选预计算六元组：`qb_to_qstart`、`qb_to_qblock`、`qb_seq_lens`、`qb_qend`（均为长度 `all_seqblock_q` 的 `int32` 数组）、`block_table`（`int32`、`[all_seqblock_q, max_blocks]`，物理页号）和 Python `int` 类型的 `all_seqblock_q`。省略时由算子构建。 |
| `per_query_seq_lens` | `torch.Tensor` 或 `None` | `[total_q]` | `int32` | `None` | 逐 query 的 KV 长度计数，包含当前 token（即位置加 1）；只用于选块边界和 local block，不控制打分掩码。头数相等时使用，否则忽略。融合分支断言 dtype 为 `int32` 且长度等于 `total_q`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `idx_o` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | 与 `q` 相同 | 在 `q` 所在设备新建。不修改输入。 |
| `topk_idx`（融合分支） | `torch.Tensor` | `[num_idx_heads, total_q, topk + 1]` | `int32` | 在 `q` 所在设备新建。无效槽为 `-1`；末槽为 local block（已选中则为 `-1`）。 |
| `topk_idx`（回退分支） | `torch.Tensor` | `[num_idx_heads, total_q, topk]` | `int32` | 在 `q` 所在设备新建。`topk > 块数` 时尾部为 `-1`；不剔除 `-inf` 列。 |

预计算映射需与本次输入一致：`qb_to_qstart`、`qb_qend` 是 request 的打包 query 起止行
（结束行不包含），`qb_to_qblock` 是 request 内的 query 分组编号，`qb_seq_lens` 是 request 的
KV 总长度。`all_seqblock_q` 是 query 分组总数，`max_blocks >= ceil(max(seq_lens) / block_size_k)`。
只有这些元数据和页映射不变时，`qblock_mappings` 才能跨调用复用。

表中注明的断言由后端检查；其余 shape、dtype、块大小及映射有效性由调用方保证，
所有 Tensor 应位于同一设备。
Triton 要求其使用的 `per_query_seq_lens` 和预计算映射中的四个一维 Tensor 连续存储。
`max(seq_lens) == 0` 时仅使用无 query 的空输入，不将其用于非空 query 的选块。

## triton 后端

块打分和索引注意力在同一个 Triton kernel 中完成；选块另行执行。
融合选块分支使用 Triton，回退分支使用 `torch.topk`。
softmax 的未归一化权重先转换为 V 的 dtype 再参与矩阵乘，最终 `idx_o` 转为 `q.dtype`。

公开入口会读取设备标量并产生同步，不能直接用于图捕获；提供 `qblock_mappings` 也不能消除全部同步。

## torch_native 后端

独立的 PyTorch 正确性参考实现，不调用 Triton；按 request 和 KV 块计算，沿用相同的
掩码与选块规则。Q/K 点积及 softmax 累积使用 FP32，权重与 V 相乘前转换为 V 的 dtype，输出为 `q.dtype`。浮点舍入差异可能使两后端对接近同分的块给出不同排序。

## 调用示例

需要安装包含此算子的包及 NPU 运行环境。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

page_size = 128
num_pages = 1  # ceil(4 / page_size)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")
v_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# 单个 request 使用物理页 0。
req_to_token = torch.zeros(1, num_pages * page_size, dtype=torch.int32, device="npu")

idx_o, topk_idx = flash_prefill_bnsd_indexer(
    q,
    k_cache,
    v_cache,
    cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([4], dtype=torch.int32, device="npu"),
    req_to_token=req_to_token,
    req_pool_indices=torch.tensor([0], dtype=torch.int32, device="npu"),
    block_size_q=4,
    block_size_k=page_size,
    topk=2,
    sm_scale=128**-0.5,
    per_query_seq_lens=torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="npu"),
    impl="triton",  # 可改为 "torch_native" 运行参考实现。
)
print(idx_o.shape, idx_o.dtype)        # torch.Size([4, 1, 128]), torch.bfloat16
print(topk_idx.shape, topk_idx.dtype)  # torch.Size([1, 4, 3]), torch.int32
print(topk_idx[0, -1].tolist())        # [0, -1, -1]
```
