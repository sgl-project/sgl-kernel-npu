# attention_sinks_prefill

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention import attention_sinks_prefill


def attention_sinks_prefill(
    query,
    k_cache,
    v_cache,
    sinks,
    seq_lens,
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

对拼接请求和分页 KV 计算因果 prefill 注意力，每个头的 softmax 分母包含零输出贡献的 sink。context_lens 包含新加入的查询，各查询只能看到已有前缀和之前的查询；支持滑动窗口。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | Tensor | `Required` | 连续浮点查询 [tokens,QH*D]。 |
| `k_cache` | Tensor | `Required` | 连续浮点 key 缓存 [pages,page_size,KVH,D]。 |
| `v_cache` | Tensor | `Required` | 与 k_cache 同形状、同类型的 value 缓存。 |
| `sinks` | Tensor | `Required` | 浮点 sink logits [QH]。 |
| `seq_lens` | Tensor | `Required` | 整数请求查询长度；FIA 接口中表示逐查询的因果 KV 长度。 |
| `block_tables` | Tensor | `Required` | INT32/INT64 物理页 ID [batch,max_pages]。 |
| `context_lens` | Tensor | `Required` | INT32/INT64 总 KV 长度 [batch]，包含新查询。 |
| `scale` | float | `Required` | 查询与键的注意力分数缩放系数。 |
| `sliding_window_size` | int | `Required` | 可见 token 窗口，-1 关闭窗口限制。 |
| `q_head_num` | int | `Required` | 查询头数，须能被 k_head_num 整除。 |
| `k_head_num` | int | `Required` | KV 头数。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 注意力输出 [tokens,QH*D]，与 query 同类型、同设备。 |

## triton 后端

- 按请求和查询头处理变长 prefill，利用累计查询长度确定各 token 的因果 KV 范围。
- 以 FP32 累加 attention，并把 sink logit 纳入 softmax 分母；支持滑动窗口，输出使用 query dtype。

## torch_native 后端

- 按 block table 收集各请求的 K/V，展开共享 KV 头，并通过 PyTorch 矩阵运算计算 attention。
- 将 sink 作为只参与归一化的额外 logit，按有效窗口取值；FP32 计算结果转回 query dtype。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import attention_sinks_prefill

q = torch.randn(1, 4096, device="npu", dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 64, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
sinks = torch.zeros(64, device="npu")
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
length = torch.tensor([1], device="npu", dtype=torch.int32)
y = attention_sinks_prefill(
    q, k, v, sinks, length, table, length, 0.125, -1, 64, 8, impl="triton"
)
```
