# kv_quant_sparse_flash_attention

## 芯片支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

def kv_quant_sparse_flash_attention(
    query,
    query_rope,
    packed_kv,
    sparse_indices,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value,
    *,
    return_softmax_lse=True,
    impl="auto",
):
    ...
```

在 arch22 和 arch35 上，`auto` 选择 `ascendc`。SGLang 使用
`torch.ops.npu.kv_quant_sparse_flash_attention` 原生绑定，使用相同的 Tensor 与注意力参数。

## 功能说明

使用显式的 Top-K 索引计算 MLA Attention。
RoPE64 的每条 KV 记录依次存放 512 字节 INT8 或 E4M3 FP8 latent、128 字节 BF16 RoPE，
以及从第 640 字节开始的四个 FP32 scale。每个 scale 对应连续的 128 个
latent 元素。latent 在 FP32 中乘 scale，再舍入到 BF16 后参与 Attention。
RoPE0 的 KV 行长为 528 字节，四个 scale 从字节偏移 512 开始，query_rope 的末维为空。
计算因果注意力时，传入已经按可见范围筛选的索引。

```text
latent[d] = BF16(FP32(quantized_latent[d]) * scale[d // 128])
logit[i] = (dot(query, latent[i]) + dot(query_rope, rope[i])) * scale_value
maximum = max(logit)
total = sum(exp(logit - maximum))
attention_out = sum(exp(logit[i] - maximum) * latent[i]) / total
LSE = maximum + log(total)
```

尾部的 `-1` 表示选取结束。空行的三个输出均为零，对应的 LSE 为负无穷。

## 参数与返回值

所有输入 Tensor 位于同一张 NPU。`T` 为查询 token 数，`H` 为查询头数，
`B` 为请求数，`K` 为 Top-K 容量。

| 参数 | 类型与形状 | 说明 |
| --- | --- | --- |
| `query` | BF16 `[T, H, 512]` | TND 查询 latent。`H` 为 1～64 或 128。 |
| `query_rope` | BF16 `[T, H, 64]` 或 `[T, H, 0]` | 查询 RoPE。 |
| `packed_kv` | INT8 或 UINT8 `[pages, page_size, 1, 656 或 528]` | PA_BSND 打包 KV，只有一个 KV 头。页大小为不超过 1024 的正 16 倍数。 |
| `sparse_indices` | INT32 `[T, 1, K]` | 每个请求中选中的逻辑位置，尾部用 `-1` 填充。 |
| `block_table` | INT32 `[B, max_pages]` | 从逻辑页映射到物理页。 |
| `actual_seq_lengths_query` | INT32 `[B]` | 查询长度的累加值，无开头的零。 |
| `actual_seq_lengths_kv` | INT32 `[B]` | 每个请求在本 rank 的有效 KV token 数。 |
| `scale_value` | float | Softmax 前的分数缩放系数。 |
| `return_softmax_lse` | bool，默认 `True` | 仅限关键字，用于控制是否返回 softmax 统计量。 |
| `impl` | str，默认 `"auto"` | 仅限关键字的实现选择。 |

| 返回值 | 类型与形状 | 说明 |
| --- | --- | --- |
| `attention_out` | BF16 `[T, H, 512]` | 本 rank 的 Attention 输出。 |
| `softmax_max` | FP32 `[1, T, H]` 或空 Tensor | 每个查询头的最大分数。`return_softmax_lse=False` 时为空。 |
| `softmax_sum` | FP32 `[1, T, H]` 或空 Tensor | 分数减去最大值后，取指数并求和。`return_softmax_lse=False` 时为空。 |

## Ascend C 后端

A2/A3（arch22）使用 `torch.int8` 打包 KV。A5（arch35）用 `torch.uint8` 承载 E4M3 FP8 的字节。
两个架构的内核都直接读取分页打包 KV，并可返回 softmax 统计量供框架做跨 rank LSE 合并。

## torch_native 后端

精度参考在 CPU 解码缓存并用 PyTorch 计算 Attention。`torch.int8` 按 INT8 解码，
`torch.uint8` 按 E4M3 FP8 解码。softmax 统计量保持 FP32，
指数权重在乘 V 前舍入为 BF16。结果放回查询所在设备。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

device = "npu"
packed_dtype = torch.uint8  # A5 使用 E4M3 FP8，A2/A3 改为 torch.int8。
query = torch.zeros((1, 32, 512), dtype=torch.bfloat16, device=device)
query_rope = torch.zeros((1, 32, 64), dtype=torch.bfloat16, device=device)
packed = torch.cat(
    (
        torch.zeros((1, 128, 1, 512), dtype=torch.uint8, device="cpu"),
        torch.zeros((1, 128, 1, 64), dtype=torch.bfloat16, device="cpu").view(torch.uint8),
        torch.ones((1, 128, 1, 4), dtype=torch.float32, device="cpu").view(torch.uint8),
    ),
    dim=-1,
).view(packed_dtype).to(device)
indices = torch.tensor([[[0]]], dtype=torch.int32, device=device)
table = torch.tensor([[0]], dtype=torch.int32, device=device)
lengths = torch.tensor([1], dtype=torch.int32, device=device)
out, maximum, total = kv_quant_sparse_flash_attention(
    query, query_rope, packed, indices, table, lengths, lengths, 1 / 24, impl="ascendc"
)
```
