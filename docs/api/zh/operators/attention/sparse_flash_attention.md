# sparse_flash_attention

## 硬件支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `cce` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

```python
from sgl_kernel_npu.operators.attention import sparse_flash_attention

def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` 选择 `ascendc`；指定 `impl="cce"` 使用手写 CCE 实现；
`torch_native` 使用 PyTorch 计算。CCE 仅支持 arch35（3510，Ascend 950），
不支持 A3/arch22。
原实现不可用时直接报错，不会回退到其他实现。

AscendC 注册名为 `torch.ops.npu.sgl_sparse_flash_attention`，CCE 注册名为
`torch.ops.cce.sparse_flash_attention`。建议使用上述公开函数，保留代际检查与实现选择。

## 功能说明

本算子实现 DeepSeek 稀疏注意力（MLA），只对 `sparse_indices` 选中的 KV
分块做计算。每个 query token 携带 top-k 索引，Kernel 从 `key` 中取 `D`
个非 rope 维度、从 `key_rope` 中取 `D_rope` 个维度，在选中位置上做 softmax
后再与 `value` 相乘：

```text
positions = pad_truncate(sparse_indices * sparse_block_size, sparse_block_size)
logits    = q @ k[positions].T * scale_value
attn_out  = softmax(logits) @ v[positions]
lse       = logmax(logits) + log(sum(exp(logits - logmax(logits))))
```

索引中的 `-1` 表示填充，遇到即停止选择；超出有效 KV 长度的位置会被忽略。

## 参数与返回值

`B` 为 batch，`S1` 为 query 长度，`T` 为 TND 下的打包 token 数，`N1`/`N2`
为 query/KV head 数，`D` 为非 rope head 维度，`D_rope` 为 rope 维度。
AscendC Kernel 仅支持 `D = 512`、`D_rope = 64`、`N2 = 1`，且 `N1` 取值
`1..64` 或 `128`。

| 参数 | 形状 | 类型/默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | `[T, N1, D]` 或 `[B, S1, N1, D]` | FP16 / BF16 | TND 或 BSND query。 |
| `key` | `[Bn, Bs, N2, D]`、`[T, N2, D]` 或 `[B, S, N2, D]` | 同 `query` | 分页、TND 或 BSND KV。 |
| `value` | 同 `key` | 同 `query` | KV value。 |
| `sparse_indices` | `[T, N2, K]` 或 `[B, S1, N2, K]` | INT32 | 选中的 KV 分块索引，`-1` 填充。 |
| `scale_value` | — | float | softmax 缩放系数。 |
| `block_table` | `[B, maxBlocks]` | INT32 / `None` | `PA_BSND` 必填，其余布局必须为 `None`。 |
| `actual_seq_lengths_query` | `[B]` | INT32 / `None` | TND query 前缀和，`TND` 必填。 |
| `actual_seq_lengths_kv` | `[B]` | INT32 / `None` | 有效 KV 长度（TND KV 为前缀和）。 |
| `query_rope` | `[T, N1, D_rope]` 或 `[B, S1, N1, D_rope]` | 同 `query` | `attention_mode=2` 必填。 |
| `key_rope` | `[Bn, Bs, N2, D_rope]`、`[T, N2, D_rope]` 或 `[B, S, N2, D_rope]` | 同 `query` | `attention_mode=2` 必填。 |
| `sparse_block_size` | — | int / `1` | 每个索引对应的 KV 位置数。 |
| `layout_query` | — | `"BSND"` | `BSND` 或 `TND`。 |
| `layout_kv` | — | `"BSND"` | `BSND`、`TND` 或 `PA_BSND`。 |
| `sparse_mode` | — | int / `3` | 稀疏选择模式；`0` 表示仅使用给定 top-k。 |
| `pre_tokens`、`next_tokens` | — | int / `INT64_MAX` | 因果窗口边界。 |
| `attention_mode` | — | int / `2` | 仅支持 `2`（MLA）。 |
| `return_softmax_lse` | — | `False` | 是否返回 softmax max/sum。 |
| `impl` | — | `"auto"` | 仅关键字参数：`auto`、`ascendc`、`cce`、`torch_native`。 |

| 返回值 | 形状 | 类型 |
| --- | --- | --- |
| `attention_out` | 同 `query` | 同 `query` |
| `softmax_max` | `[N2, T, N1/N2]`（TND）或 `[B, N2, S1, N1/N2]`（BSND） | FP32 |
| `softmax_sum` | 同 `softmax_max` | FP32 |

使用说明：

- 所有张量必须在同一张 NPU 上且连续。
- 对 `ascendc` 和 `torch_native`，`return_softmax_lse` 为 false 时 `softmax_max`/`softmax_sum` 为空张量；
  `softmax_max + log(softmax_sum)` 即自然对数 LSE。
- `torch_native` 参考实现覆盖 `sparse_mode=0`、`attention_mode=2`，即 DSA
  生产链路使用的组合。

## CCE 后端（仅 arch35）

CCE prefill 实现沿用公开函数签名，但输入范围更窄。输出为 BF16，softmax
统计量为 FP32。

| 输入/设置 | CCE 要求 |
| --- | --- |
| `query` | BF16 `[B, SQ, H, 512]`，`B >= 1`，`1 <= SQ <= 8192`。 |
| `query_rope` | BF16 `[B, SQ, H, 64]`。 |
| `H` | `16`、`32`、`64` 或 `128`。 |
| `key`、`value` | 同一个张量对象（`key is value`），BF16 `[1, 8192, 1, 512]`，整个 batch 共享。 |
| `key_rope` | BF16 `[1, 8192, 1, 64]`，整个 batch 共享。 |
| `sparse_indices` | INT32 `[B, SQ, 1, 2048]`；使用 `0..8191` 范围内的索引。 |
| 布局/缩放 | `layout_query=layout_kv="BSND"`，`scale_value=1/24`。 |
| 存储 | 同一 NPU，连续基础格式（ND/NCHW）；BF16 首地址按 32 字节对齐。输入不得要求梯度。 |

使用显式 token 索引：`sparse_mode=0`、`sparse_block_size=1`、`attention_mode=2`。
直接提供有效索引，不要依赖通用接口中“遇到第一个 `-1` 即停止”的填充约定；
CCE 会逐项屏蔽无效索引，而不会在第一个无效索引处停止。

CCE wrapper 不会把 `block_table`、序列有效长度、`sparse_mode`、
`sparse_block_size`、`attention_mode` 或 `pre_tokens`/`next_tokens` 传给内核。
这些参数不提供分页、长度裁剪或因果窗口处理；需要将预期选择直接编码在
`sparse_indices` 中。

即使 `return_softmax_lse=False`，CCE 也始终返回
`(attention_out, softmax_max, softmax_sum)`，形状依次为 `[B, SQ, H, 512]`、
`[B, 1, SQ, H]`、`[B, 1, SQ, H]`。不支持的配置不会自动回退到其他实现。

### 调用示例

在 arch35 上运行，安装的 wheel 需包含 CCE 算子：

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import sparse_flash_attention

batch, q_len, heads = 1, 2, 16
device = "npu"
query = torch.randn(batch, q_len, heads, 512, device=device, dtype=torch.bfloat16)
query_rope = torch.randn(batch, q_len, heads, 64, device=device, dtype=torch.bfloat16)
kv = torch.randn(1, 8192, 1, 512, device=device, dtype=torch.bfloat16)
key_rope = torch.randn(1, 8192, 1, 64, device=device, dtype=torch.bfloat16)
indices = torch.arange(2048, device=device, dtype=torch.int32)
indices = indices.view(1, 1, 1, 2048).expand(batch, q_len, 1, 2048).contiguous()

out, softmax_max, softmax_sum = sparse_flash_attention(
    query, kv, kv, indices, 1.0 / 24.0,
    query_rope=query_rope, key_rope=key_rope,
    layout_query="BSND", layout_kv="BSND",
    sparse_mode=0, sparse_block_size=1, attention_mode=2,
    return_softmax_lse=True, impl="cce",
)
```

跨代际 CI 使用 `check_accuracy`，或在直接调用 CCE 前使用 `require_test_impl` /
`require_distinct_impls`。这些测试辅助函数会跳过不支持的代际；直接调用公开 API
则会报错。
