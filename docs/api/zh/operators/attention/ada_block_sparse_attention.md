# ada_block_sparse_attention

分块稀疏注意力算子。

## 硬件支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | FP16 / BF16 | — |
| `torch_native` | FP16 / BF16 | FP16 / BF16 |

## 签名

```python
from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

def ada_block_sparse_attention(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    *,
    input_layout="BNSD",
    sparse_size=128,
    num_heads=1,
    num_key_value_heads=1,
    scale_value=1.0,
    causal=True,
    inner_precise=1,
    pre_tokens=214748647,
    next_tokens=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    impl="auto",
):
    ...
```

`auto` 在 arch22 上选择 `ascendc`；`torch_native` 使用 PyTorch 实现。
在 arch35 上请求 `ascendc` 会直接报错，不会回退。

## 功能说明

对 BNSD 布局的 Q/K/V 计算分块稀疏注意力。稀疏粒度由 `sparse_size` 指定：
`sparse_mask[b, h, i, j] != 0` 表示查询块 `i` 可以关注 KV 块 `j` 的全部位置。
`causal=True` 时还会在最后一个对角块内做逐元素下三角屏蔽。

```text
scores = scale_value * (query @ key^T)
mask   = 块稀疏掩码 [+ 对角块 causal]
out    = softmax(scores, mask) @ value
```

支持 GQA：`key`/`value` 的头数可以少于 `query`，每个 KV 头服务
`query_heads / kv_heads` 个查询头。若某个查询块没有任何可用 KV 块，该块输出全零。

## 参数与返回值

`B` 为 batch，`Nq`/`Nkv` 为查询/KV 头数，`S1`/`S2` 为查询/KV 序列长度，
`D` 为 head 宽度。

| 参数 | 形状 | 类型 / 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | `[B, Nq, S1, D]` | FP16 / BF16 | 必须为 4 维。 |
| `key` | `[B, Nkv, S2, D]` | 同 `query` | `Nq % Nkv == 0`。 |
| `value` | `[B, Nkv, S2, D]` | 同 `query` | value 张量。 |
| `sparse_mask` | `[B, Nq, ceil(S1/bs), ceil(S2/bs)]` | int8 / bool | 块稀疏掩码；最后一维允许按 32 补齐。 |
| `sparse_count_table` | `[B, Nq, ceil(S1/bs)]` | int32 | 每个查询块的可用 KV 块数量。 |
| `input_layout` | — | str，`"BNSD"` | 仅支持 BNSD，TND 会被拒绝。 |
| `sparse_size` | — | int，128 | 稀疏块大小。 |
| `num_heads` | — | int，1 | 查询头数，应等于 `Nq`。 |
| `num_key_value_heads` | — | int，1 | KV 头数；传 0 时按 `num_heads` 处理。 |
| `scale_value` | — | float，1.0 | 分数缩放系数。 |
| `causal` | — | bool，`True` | 是否启用因果屏蔽。 |
| `inner_precise` | — | int，1 | 高精度累加模式。 |
| `pre_tokens` / `next_tokens` | — | int | 滑窗边界；默认值表示不限制。 |
| `actual_seq_lengths` / `actual_seq_lengths_kv` | — | `None` | 变长序列，当前实现按定长处理。 |
| `impl` | — | `"auto"` | 仅关键字：`auto`、`ascendc`、`torch_native`。 |

| 返回 | 形状 | 类型 |
| --- | --- | --- |
| `attention_out` | `[B, Nq, S1, D]` | 同 `query` |

## ascendc 后端

- 要求所有张量位于同一 NPU；非连续输入会在内部转为连续工作张量。
- 掩码最后一维的补齐区域必须为 0，`sparse_count_table` 必须等于掩码中有效块的数量。

## torch_native 后端

- 仅支持 BNSD、定长序列和不限制的窗口。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

batch, heads, seq, dim = 1, 2, 256, 128
blocks = seq // 128
query = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
mask = torch.zeros(batch, heads, blocks, 32, device="npu", dtype=torch.int8)
mask[..., :blocks] = 1
count = mask[..., :blocks].sum(dim=-1, dtype=torch.int32)

out = ada_block_sparse_attention(
    query, key, value, mask, count,
    sparse_size=128,
    num_heads=heads,
    num_key_value_heads=heads,
    scale_value=dim**-0.5,
    causal=False,
)
print(out.shape)  # [1, 2, 256, 128]
```
