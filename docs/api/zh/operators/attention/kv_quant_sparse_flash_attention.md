# kv_quant_sparse_flash_attention

## 芯片支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | 不支持 | 已验证 |
| `torch_native` | 不支持 | 已验证 |

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
    impl="auto",
):
    ...
```

在 arch35 上，`auto` 选择 `ascendc`。SGLang 使用
`torch.ops.npu.sgl_kv_quant_sparse_flash_attention` 原生绑定，入参为相同的八项。

## 功能说明

使用显式的 rank-local Top-K 索引，计算单个 DCP rank 的 MLA Attention。
每条 KV 记录依次存放 512 字节 FP8 E4M3FN latent、128 字节 BF16 RoPE，
以及从第 640 字节开始的四个 FP32 scale。每个 scale 对应连续的 128 个
latent 元素。latent 在 FP32 中乘 scale，再舍入到 BF16 后参与 Attention。
RoPE 已经是 BF16。

```text
latent[d] = BF16(FP32(fp8_latent[d]) * scale[d // 128])
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
| `query_rope` | BF16 `[T, H, 64]` | 查询 RoPE。 |
| `packed_kv` | UINT8 `[pages, page_size, 1, 656]` | PA_BSND 打包 KV，只有一个 KV 头。页大小为不超过 1024 的正 16 倍数。 |
| `sparse_indices` | INT32 `[T, 1, K]` | 每个请求中选中的逻辑位置，尾部用 `-1` 填充。 |
| `block_table` | INT32 `[B, max_pages]` | 从逻辑页映射到物理页。 |
| `actual_seq_lengths_query` | INT32 `[B]` | 查询长度的累加值，无开头的零。 |
| `actual_seq_lengths_kv` | INT32 `[B]` | 每个请求在本 rank 的有效 KV token 数。 |
| `scale_value` | float | Softmax 前的分数缩放系数。 |
| `impl` | str，默认 `"auto"` | 仅限关键字的实现选择。 |

| 返回值 | 类型与形状 | 说明 |
| --- | --- | --- |
| `attention_out` | BF16 `[T, H, 512]` | 本 rank 的 Attention 输出。 |
| `softmax_max` | FP32 `[1, T, H]` | 每个查询头的最大分数。 |
| `softmax_sum` | FP32 `[1, T, H]` | 分数减去最大值后，取指数并求和。 |

## Ascend C 后端

arch35 kernel 直接读取分页打包 KV，返回两项 softmax 统计量供框架做跨 rank LSE 合并。

## torch_native 后端

精度参考在 CPU 解码缓存并用 PyTorch 计算 Attention。softmax 统计量保持 FP32，
指数权重在乘 V 前舍入为 BF16。结果放回查询所在设备。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

device = "npu"
query = torch.zeros((1, 32, 512), dtype=torch.bfloat16, device=device)
query_rope = torch.zeros((1, 32, 64), dtype=torch.bfloat16, device=device)
packed = torch.cat(
    (
        torch.zeros((1, 128, 1, 512), dtype=torch.uint8, device="cpu"),
        torch.zeros((1, 128, 1, 64), dtype=torch.bfloat16, device="cpu").view(torch.uint8),
        torch.ones((1, 128, 1, 4), dtype=torch.float32, device="cpu").view(torch.uint8),
    ),
    dim=-1,
).to(device)
indices = torch.tensor([[[0]]], dtype=torch.int32, device=device)
table = torch.tensor([[0]], dtype=torch.int32, device=device)
lengths = torch.tensor([1], dtype=torch.int32, device=device)
out, maximum, total = kv_quant_sparse_flash_attention(
    query, query_rope, packed, indices, table, lengths, lengths, 1 / 24, impl="ascendc"
)
```
