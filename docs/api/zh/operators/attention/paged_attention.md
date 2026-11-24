# paged_attention

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascend-transformer-boost` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

两个代际的默认实现均通过 TorchNPU 调用 ATB 接口，并使用相同的主机侧长度参数。

## 函数原型

`auto` 在对应代际选择已注册的 `ascend-transformer-boost`。

```python
from sgl_kernel_npu.operators.attention import paged_attention


def paged_attention(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_kv_heads,
    scale_value,
    block_table,
    context_lens,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

通过页表读取键值缓存，对每个请求的一个解码 token 执行分组查询注意力。每个头在有效缓存前缀上计算 $Y=\mathrm{softmax}(sQK^\top)V$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [T, H, D], FP16/BF16 | 必传 | NPU 上每个 query token 对应一行。 |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, Dk] 与 [P, L, Hkv, Dv] | 必传 | 分页键值缓存。P 是物理页数，L 是页大小。 |
| `num_heads`, `num_kv_heads` | int | - | 必传 | query 与 KV 的头数。H 须能被 Hkv 整除。 |
| `scale_value` | float | - | 必传 | 注意力 logit 尺度，通常取 Dk**-0.5。 |
| `block_table` | Tensor | [B, max_pages], INT32, NPU | 必传 | 将每个请求的逻辑页映射到物理页。 |
| `context_lens` | Tensor | [B], INT32 CPU 在示例中 | 必传 | 每个请求可见的缓存 token 数。 |
| `out` | Tensor | [T, H, Dv], same dtype 与 NPU as query | 必传 | 由调用方分配，原地写入。 |
| `impl` | str | - | `"auto"` | `auto`, `ascend-transformer-boost`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 结果写入 out。 |

## ascend-transformer-boost 后端

调用 `torch_npu._npu_paged_attention`。形状、dtype 和布局组合以已安装的 SDK 为准。 要求 SDK 提供 ATB 私有分页注意力入口。示例使用主机侧长度和设备侧页表。workspace 等可选后端参数由 SDK 决定。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_paged_attention.py) [Ascend Transformer Boost 源码](https://gitcode.com/cann/ascend-transformer-boost)

## torch_native 后端

根据每个解码请求的页表读出 ND 布局的 KV，结果写入 `out`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention

heads, kv_heads, dk, dv = 32, 16, 288, 96
q = torch.randn((2, heads, dk), dtype=torch.float16, device="npu")
k = torch.randn((64, 128, kv_heads, dk), dtype=q.dtype, device=q.device)
v = torch.randn((64, 128, kv_heads, dv), dtype=q.dtype, device=q.device)
table = torch.arange(16, dtype=torch.int32, device=q.device).reshape(2, 8)
lengths = torch.full((2,), 1024, dtype=torch.int32, device="cpu")
out = torch.empty((2, heads, dv), dtype=q.dtype, device=q.device)
paged_attention(
    q, k, v, heads, kv_heads, dk**-0.5, table, lengths, out, impl="ascend-transformer-boost"
)
```
