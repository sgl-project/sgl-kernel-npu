# sparse_attn_sharedkv

## 芯片支持

| 实现 | arch22（2201） | arch35（3510） |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import sparse_attn_sharedkv

def sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=0.0,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=128,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` 默认使用 `ascendc`；`torch_native` 使用 PyTorch 运算。
原生实现不可用时报错，不自动切换实现。

## 功能说明

该算子计算共享 KV 的稀疏注意力。当前实现覆盖滑动窗口（SWA）模板与分页 KV：

```text
aligned = seqused_kv[b] - q_len + s
left    = max(aligned - ori_win_left, 0)
right   = min(aligned + ori_win_right, seqused_kv[b] - 1)
logits  = [sink_h, q[b, s, h] @ k[left:right+1].T * softmax_scale]
attn_out[b, s, h] = softmax(logits)[1:] @ k[left:right+1]
```

其中 `k` 由 `ori_kv` 按 `ori_block_table` 还原为逻辑序列，
`h // g` 为对应的 KV 头，`g = n1 / n2`。`sinks` 作为额外的一个
零值虚拟 token 参与 softmax。

`layout_kv` 仅支持 `PA_ND`，`ori_mask_mode` 仅支持 `4`，
`return_softmax_lse` 仅支持 `False`。

## 参数与返回值

`B` 为 batch 数，`S1` 为 query 长度，`T = B * S1` 为 TND 下的 token 数，
`N1`、`N2` 分别为 query 和 KV 头数，`D` 为 head dim。

| 参数 | Shape | Dtype / 默认值 | 说明 |
| --- | --- | --- | --- |
| `q` | `[B, S1, N1, D]` 或 `[T, N1, D]` | FP16 / BF16 | BSND 或 TND 布局。 |
| `ori_kv` | `[blockNum, blockSize, N2, D]` | 同 `q` | 分页 KV 缓存。 |
| `cmp_kv` | 同 `ori_kv` | 同 `q` | 压缩 KV，当前参考实现不支持。 |
| `ori_sparse_indices` | 任意 | INT32 | 原生实现可选输入。 |
| `cmp_sparse_indices` | 任意 | INT32 | 压缩 KV 的稀疏索引。 |
| `ori_block_table` | `[B, maxBlocks]` | INT32 | 分页 KV 的页表。 |
| `cmp_block_table` | `[B, maxBlocks]` | INT32 | 压缩 KV 的页表。 |
| `cu_seqlens_q` | `[B + 1]` | INT32 / `None` | TND 布局必填，BSND 布局必须为 `None`。 |
| `cu_seqlens_ori_kv` | `[B + 1]` | INT32 / `None` | 原生实现可选输入。 |
| `cu_seqlens_cmp_kv` | `[B + 1]` | INT32 / `None` | 压缩 KV 的累计长度。 |
| `seqused_q` | `[B]` | INT32 / `None` | 原生实现可选输入。 |
| `seqused_kv` | `[B]` | INT32，必填 | 每个 batch 的有效 KV 长度。 |
| `sinks` | `[N1]` | FP32，必填 | attention sink 值。 |
| `metadata` | `[1024]` | INT32，必填 | 核间任务划分元数据。 |
| `softmax_scale` | — | float / `0.0` | softmax 缩放。 |
| `cmp_ratio` | — | int / `0` | 压缩比例。 |
| `ori_mask_mode` | — | int / `4` | 仅支持 `4`（滑动窗口）。 |
| `cmp_mask_mode` | — | int / `3` | 压缩 KV mask 模式。 |
| `ori_win_left`、`ori_win_right` | — | int / `128`、`0` | 滑动窗口左右范围。 |
| `layout_q` | — | `"BSND"` | 可选 `BSND`、`TND`。 |
| `layout_kv` | — | `"PA_ND"` | 仅支持 `PA_ND`。 |
| `return_softmax_lse` | — | `False` | 仅支持 `False`。 |
| `impl` | — | `"auto"` | 可选 `auto`、`ascendc`、`torch_native`，仅限关键字。 |

| 返回值 | Shape | Dtype |
| --- | --- | --- |
| `attn_out` | 与 `q` 相同 | 与 `q` 相同 |
| `softmax_lse` | `[0]` | FP32 |

使用时请注意：

- 所有张量必须位于同一 NPU；当前原生实现要求 `N1 = 64` 且 `N2 = 1`（仅支持单个 KV 头）。
- `ori_kv`、`sinks`、`metadata` 必填；`ori_block_table` 与 `seqused_kv` 在
  `PA_ND` 下必填。
- `torch_native` 仅实现 SWA + `PA_ND`，不实现压缩 KV 模板。
