# flash_attention_qlens

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascend-transformer-boost` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

旧版 ATB 入口在 A2/A3 上注册，采用主机侧长度参数。

## 函数原型

`auto` 在对应代际选择已注册的 `ascend-transformer-boost`。

```python
from sgl_kernel_npu.operators.attention import flash_attention_qlens


def flash_attention_qlens(
    query,
    key_cache,
    value_cache,
    mask,
    block_table,
    seq_len,
    context_lens,
    scale_value,
    num_heads,
    num_kv_heads,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

执行可变长因果预填充。页表用于读取 K/V，查询长度与上下文总长度用于定位各请求的可见键前缀。每个头计算 $\mathrm{softmax}(sQK^\top+M)V$，掩码采用 SDK 格式。

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
| `mask` | Tensor | [128, 128], BF16 在示例中 | 必传 | 厂商因果路径使用的压缩上三角掩码。 |
| `seq_len` | Tensor | [B], INT32 CPU 在示例中 | 必传 | 每个请求打包的 query token 数。 |
| `impl` | str | - | `"auto"` | `auto`, `ascend-transformer-boost`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 结果写入 out。 |

## ascend-transformer-boost 后端

调用 `torch_npu._npu_flash_attention_qlens`。形状、dtype 和布局组合以已安装的 SDK 为准。 要求 SDK 提供 ATB 私有 Flash Attention 入口。示例采用 ATB BF16 测试配置，缓存已包含当前查询后缀。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_flash_attention_qlens.py) [Ascend Transformer Boost 源码](https://gitcode.com/cann/ascend-transformer-boost)

## torch_native 后端

读取 ND 布局的分页 KV 缓存，应用输入的二维预填充 mask，结果写入 `out`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_attention_qlens

heads, kv_heads, d = 32, 16, 128
q = torch.randn((384, heads, d), dtype=torch.bfloat16, device="npu")
k = torch.randn((64, 128, kv_heads, d), dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
mask = torch.ones((128, 128), dtype=q.dtype, device=q.device).triu(1)
table = torch.arange(16, dtype=torch.int32, device=q.device).reshape(2, 8)
qlens = torch.tensor([128, 256], dtype=torch.int32, device="cpu")
lengths = torch.tensor([1024, 1024], dtype=torch.int32, device="cpu")
out = torch.empty_like(q)
flash_attention_qlens(
    q,
    k,
    v,
    mask,
    table,
    qlens,
    lengths,
    d**-0.5,
    heads,
    kv_heads,
    out,
    impl="ascend-transformer-boost",
)
```
