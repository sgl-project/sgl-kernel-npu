# attention_sinks

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention import attention_sinks


def attention_sinks(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
    *,
    impl="auto"
): ...
```

## 功能描述

计算分页分组查询解码注意力，每个查询头包含一个可学习的 sink logit。sink 参与 softmax 分母但不贡献输出值；滑动窗口参数限制可见 KV token。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | Tensor | `Required` | 连续浮点查询 [tokens,QH*D]。 |
| `k_cache` | Tensor | `Required` | 连续浮点 key 缓存 [pages,page_size,KVH,D]。 |
| `v_cache` | Tensor | `Required` | 与 k_cache 同形状、同类型的 value 缓存。 |
| `sinks` | Tensor | `Required` | 浮点 sink logits [QH]。 |
| `block_tables` | Tensor | `Required` | INT32/INT64 物理页 ID [batch,max_pages]。 |
| `context_lens` | Tensor | `Required` | INT32/INT64 总 KV 长度 [batch]，包含新查询。 |
| `scale` | float or None | `Required` | QK 缩放；KDA 默认 key_dim**-0.5，要求为正。 |
| `sliding_window_size` | int | `Required` | 可见 token 窗口，-1 关闭窗口限制。 |
| `q_head_num` | int | `Required` | 查询头数，须能被 k_head_num 整除。 |
| `k_head_num` | int | `Required` | KV 头数。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 注意力输出 [tokens,QH*D]，与 query 同类型、同设备。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import attention_sinks

q = torch.randn(1, 4096, device="npu", dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 64, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
sinks = torch.zeros(64, device="npu")
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
length = torch.tensor([1], device="npu", dtype=torch.int32)
y = attention_sinks(q, k, v, sinks, table, length, 0.125, -1, 64, 8, impl="triton")
```
