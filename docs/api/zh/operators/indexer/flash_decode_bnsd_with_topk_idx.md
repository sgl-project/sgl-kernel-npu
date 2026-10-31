# flash_decode_bnsd_with_topk_idx

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

`ascendc` 未实现：本算子只提供 `triton` 与 `torch_native` 两种实现。

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数；不传 `impl` 或显式传入 `"auto"` 时均使用
Triton。

```python
from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

def flash_decode_bnsd_with_topk_idx(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    max_seqlen,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
    disable_index_value=False,
    num_kv_chunks=None,
    max_num_kv_chunks=8,
    use_triton_topk=None,
    num_topk_chunks=None,
    use_chunked_score=True,
    score_blocks_per_chunk=16,
    score_max_chunks=32,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    packed_seq_lens=False,
    runtime_fill_only=False,
    runtime_score_short_max_blocks=0,
    runtime_score_short_chunks=0,
    fused_append_local=False,
    use_native=True,
    *,
    impl="auto",
):
    ...
```

## 功能说明

对 paged KV cache 做 decode：按块（block）计算分数并返回块级 topk 索引，或在全量注意力路径上
同时返回注意力输出与 topk 索引。三条互斥分支：

- **① 空 topk**：`disable_index_value=True` 且 `topk<=0`，直接返回 `(None, [QH, B, 0])`。
- **② 只选块**：`disable_index_value=True` 且 `topk>0`，逐行逐头计算每个 KV 块的分数，返回
  `(None, topk_idx)`；生产 decode/verify 走该路径。
- **③ 全量注意力**：`disable_index_value=False`，在一次 KV 扫描中同时产出注意力输出与块分数，
  返回 `(o, topk_idx)`。

受支持范围：分支 ①②（score-only）为完整支持范围，包括 `score_type` 的 `max`/`lse`、
`init_blocks`/`local_blocks` 强制块、直连与 `block_table` 两种 page source、`packed_seq_lens`、
`sanitize_page_ids` 与 `fused_append_local`。分支 ③ 只保留在 Triton 实现中，不在支持与验证范围，
`torch_native` 会拒绝该分支。

块分数（`max` 与 `lse` 两种归约，$D$ 为 head_dim，$\sigma$ 为 `sm_scale`，默认 $D^{-1/2}$）：

$$
s_{i,h,b}=\max_{j\in b}\left(\sigma\,q_i k_j^{T}\right),
\qquad
s_{i,h,b}=\log\sum_{j\in b}\exp\left(\sigma\,q_i k_j^{T}\right)
$$

返回每行每头分数最高的 `topk` 个块。典型使用场景：

- **decode**：`disable_index_value=True`、`topk>0`，输入当前 token 的 `q` 与 paged KV cache，返回
  `[QH, B, topk]` 的块索引；常配合 `fused_append_local=True` 把因果 local block 追加到末槽。
- **verify / packed**：`packed_seq_lens=True`，`seq_lens` 为 `[B*G]` 的逐行长度，一次处理同一 request 的
  多个 draft token；`block_table` 与直连 page source 都可用。
- **全量注意力（仅 Triton）**：`disable_index_value=False`，在一次 KV 扫描中同时返回注意力输出 `o` 与
  `topk_idx`。

## 参数与返回值

公开 operator 直接透传参数，不执行输入检查；dtype、page source 组合、capacity、packed 的
`num_kv_heads`、`seq_lens`/`block_table` 维数等关键约束由各实现校验，其余 shape 约束由调用方保证。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[B, QH, D]` | `float16`、`bfloat16` | 必填 | 查询张量，位于 NPU；`QH` 必须是 `KVH` 的整数倍（GQA）。 |
| `sink` | `torch.Tensor` 或 `None` | `[QH, D]` | 同 `q` | 必填（可传 `None`） | 注意力 sink，仅 Triton 的全量注意力路径使用；生产场景恒为 `None`，torch_native 不接受该分支。 |
| `k_cache_bnsd` | `torch.Tensor` | `[num_blocks, block_size, KVH, D]` | 同 `q` | 必填 | Key cache；按块读取。 |
| `v_cache_bnsd` | `torch.Tensor` 或 `None` | 同 `k_cache_bnsd` | 同 `q` | 必填（只选块可传 `None`，全量注意力必填） | 只选块路径忽略此参数；仅 Triton 的全量注意力路径使用。 |
| `block_table` | `torch.Tensor` 或 `None` | `[B, max_num_blocks]` | `int32`、`int64` | 必填（可传 `None`） | 逻辑块到物理块映射；与直连 page source 二选一，全量注意力（仅 Triton）只支持此模式；列数需覆盖实际块数，否则越界读表（当前未断言） |
| `seq_lens` | `torch.Tensor` | `[B]` 或 `[B*QH//KVH]` | `int32`、`int64` | 必填 | 因果 KV 长度；`packed_seq_lens=True` 时为逐行长度。 |
| `max_seqlen` | `int` | — | — | 必填 | KV 长度上界，用于确定分数分块数；分数互不相同时不改变选中块；仅 triton 生效。 |
| `block_size` | `int` | — | — | 必填 | 每块 token 数，须与 cache 一致，且必须为 2 的幂（当前未断言，生产为 128）。 |
| `topk` | `int` | — | — | 必填 | 返回的块索引数量。 |
| `init_blocks` | `int` | — | — | 必填 | 前若干块候选分数改写为 `1e30`；同类强制块 ≥2 时彼此同分，次序与超出 `topk` 的取舍未定义 |
| `local_blocks` | `int` | — | — | 必填 | 尾部若干块候选分数改写为 `1e29`，优先于 `init_blocks`；同类强制块 ≥2 时彼此同分，次序与取舍未定义 |
| `sm_scale` | `float` 或 `None` | — | — | `None` | softmax 缩放；`None` 时取 `D ** -0.5`。 |
| `score_type` | `str` | — | — | `"max"` | `"max"` 或 `"lse"`，块内 token 归约方式 |
| `disable_index_value` | `bool` | — | — | `False` | `True` 为只选块路径，`False` 为全量注意力路径；torch_native 仅支持 `True`。 |
| `num_kv_chunks` | `int` 或 `None` | — | — | `None` | 全量注意力的 KV 分块数；`None` 时按 vector core 数自动选择；仅 triton 生效。 |
| `max_num_kv_chunks` | `int` | — | — | `8` | 自动分块数的 2 的幂上界；仅 triton 生效。 |
| `use_triton_topk` | `bool` 或 `None` | — | — | `None` | 兼容参数，无作用。 |
| `num_topk_chunks` | `int` 或 `None` | — | — | `None` | 兼容参数，无作用。 |
| `use_chunked_score` | `bool` | — | — | `True` | 兼容参数，无作用。 |
| `score_blocks_per_chunk` | `int` | — | — | `16` | 每个 score 程序的目标块数；仅 triton 生效。 |
| `score_max_chunks` | `int` | — | — | `32` | 分数分块数的上界；仅 triton 生效。 |
| `req_to_token` | `torch.Tensor` 或 `None` | `[num_requests, max_context]` | `int32`、`int64` | `None` | 直连 page source，按位置给出 token slot；需同时提供 `req_pool_indices`。 |
| `req_pool_indices` | `torch.Tensor` 或 `None` | `[B]` | `int32`、`int64` | `None` | `req_to_token` 的 batch 行号。 |
| `max_num_blocks` | `int` 或 `None` | — | — | `None` | page table 容量。直连模式必填且 `>0`，并要求 `max_num_blocks * block_size <= req_to_token.shape[1]`（空 topk 分支也会先校验）；`block_table` 模式仅在只选块且 `fused_append_local=True` 时必填。 |
| `num_pages` | `int` 或 `None` | — | — | `None` | 物理页数；直连模式必填。 |
| `sanitize_page_ids` | `bool` | — | — | `False` | 仅直连模式生效，把物理块号 clamp 到 `[0, num_pages-1]`。 |
| `packed_seq_lens` | `bool` | — | — | `False` | `seq_lens` 为逐行长度（verify 打包）；仅只选块路径，且要求 `num_kv_heads == 1`。 |
| `runtime_fill_only` | `bool` | — | — | `False` | 按运行期块数选择直接填充候选路径；仅 triton 生效。 |
| `runtime_score_short_max_blocks` | `int` | — | — | `0` | 短上下文自适应分块的块数阈值；启用时（非 0）必须 `>0` 且与下一参数同时给出，默认 0 表示不启用；仅 triton 生效。 |
| `runtime_score_short_chunks` | `int` | — | — | `0` | 低于阈值时使用的分块数；启用时（非 0）须为 2 的幂，超过分块总数时按总数截断，默认 0 表示不启用；仅 triton 生效。 |
| `fused_append_local` | `bool` | — | — | `False` | 融合候选合并与因果 local block 追加；仅 ② 且 `topk>0` 时输出末槽。 |
| `use_native` | `bool` | — | — | `True` | 兼容参数，无作用（无原生后端）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `o` | `torch.Tensor` 或 `None` | `[B, QH, D]` | 同 `q` | 只选块路径为 `None`；全量注意力路径（仅 Triton）为新建输出。 |
| `topk_idx` | `torch.Tensor` | `[QH, B, topk]`，或 ② 且 `topk>0` 且 `fused_append_local` 时 `[QH, B, topk+1]` | `int32` | 无效、越界或被去重的槽为 `-1`；有效候选耗尽后重复块 0。同分次序未定义。 |

附加说明：空 topk 分支返回 `[QH, B, 0]`；直连 page source 只支持只选块路径；`sink` 只影响全量注意力的
softmax 分母。

## triton 后端

- 使用 CANN Triton kernel，覆盖 ①②③ 三条分支；其中 ③ 仅保留兼容、不在支持与验证范围。按 `max_seqlen` 与运行期长度选择分数分块数。
- `init_blocks`/`local_blocks` 改写只作用于 topk 候选分数。
- 直连 page source 在 kernel 内完成逻辑块到物理块的 gather，可按需 clamp。
- `packed_seq_lens=True` 且 `num_kv_heads>1` 时，分数 kernel 与融合 merge 的行映射不一致，不支持该组合（生产
  verify 为 `num_kv_heads=1`）。

## torch_native 后端

- 使用 PyTorch 运算独立实现，不调用 Triton，**只实现 score-only 分支**（`disable_index_value=True`）：
  支持 `score_type` 的 `max`/`lse`、`init_blocks`/`local_blocks` 强制块、直连与 `block_table` 两种
  page source、`packed_seq_lens`、`sanitize_page_ids` 与 `fused_append_local`；`topk<=0` 返回空索引。
- 全量注意力分支（`disable_index_value=False`）没有参考实现，会以 `AssertionError` 拒绝。
- 忽略 12 个调度/兼容参数：`max_seqlen`、`num_kv_chunks`、`max_num_kv_chunks`、
  `score_blocks_per_chunk`、`score_max_chunks`、`runtime_fill_only`、
  `runtime_score_short_max_blocks`、`runtime_score_short_chunks`、`use_native`、`use_triton_topk`、
  `num_topk_chunks`、`use_chunked_score`。
- 按整段 KV 一次计算，显存与耗时随 KV 长度线性增长，仅用于正确性对比；`o` 恒为 `None`。
- `topk_idx` 在分数无并列时与 Triton 一致；同分次序未定义。
- `packed_seq_lens` 与 `runtime_score_short_max_blocks` 同时使用、且行组跨越该阈值时，候选列表可能被截断，
  不保证与 Triton 等价。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

batch, q_heads, kv_heads, head_dim = 1, 1, 1, 128
block_size, topk = 128, 16
seq_len = 4096
num_blocks = seq_len // block_size

# 构造互不相同的块分数，保证两个实现的 int32 索引可以逐位比较。
q = torch.zeros(batch, q_heads, head_dim, device="npu", dtype=torch.bfloat16)
q[:, :, 0] = 1.0
k_cache = torch.zeros(
    num_blocks, block_size, kv_heads, head_dim, device="npu", dtype=torch.bfloat16
)
coefficients = torch.arange(num_blocks, device="npu", dtype=torch.float32)
k_cache[:, :, :, 0] = coefficients.view(-1, 1, 1)
block_table = torch.arange(num_blocks, device="npu", dtype=torch.int32).view(1, -1)
seq_lens = torch.tensor([seq_len], device="npu", dtype=torch.int32)

print(list_impls(flash_decode_bnsd_with_topk_idx))  # ['auto', 'torch_native', 'triton']
_, topk_idx = flash_decode_bnsd_with_topk_idx(
    q, None, k_cache, None, block_table, seq_lens,
    max_seqlen=seq_len, block_size=block_size, topk=topk,
    init_blocks=0, local_blocks=0,
    disable_index_value=True, fused_append_local=True, max_num_blocks=num_blocks,
)
print(topk_idx.shape)  # [1, 1, 17]

_, topk_ref = flash_decode_bnsd_with_topk_idx(
    q, None, k_cache, None, block_table, seq_lens,
    max_seqlen=seq_len, block_size=block_size, topk=topk,
    init_blocks=0, local_blocks=0,
    disable_index_value=True, fused_append_local=True, max_num_blocks=num_blocks,
    impl="torch_native",
)
torch.testing.assert_close(topk_idx, topk_ref)
```
