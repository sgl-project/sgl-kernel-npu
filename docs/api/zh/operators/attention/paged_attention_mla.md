# paged_attention_mla

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascend-transformer-boost` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

在 Ascend 950 上，ATB 实现适用于 SGLang 的 eager decode 路径，每个本地 KV 头对应的 Q 头数小于 8。例如，DeepSeek-V2 在 attention TP32 下使用 4 个 Q 头和 1 个 KV 头。头数比为 8 或更高时请使用 FIA。

## 函数原型

`auto` 在对应代际选择已注册的 `ascend-transformer-boost`。

```python
from sgl_kernel_npu.operators.attention import paged_attention_mla


def paged_attention_mla(
    query,
    key_cache,
    num_kv_heads,
    num_heads,
    scale_value,
    block_table,
    context_lens,
    mla_vheadsize,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

从 MLA 分页缓存执行解码。完整的潜变量和旋转键参与 $\mathrm{softmax}(sQK^\top)$ 计算，缓存中前 mla_vheadsize 个潜变量提供 $V$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [T, H, D], FP16/BF16 | 必传 | NPU 上每个 query token 对应一行。 |
| `key_cache` | Tensor | [P, L, Hkv, Dk], 输入 dtype | 必传 | 分页潜变量与 RoPE 缓存。Dk 包含两个部分。 |
| `num_heads`, `num_kv_heads` | int | - | 必传 | query 与 KV 的头数。H 须能被 Hkv 整除。 |
| `scale_value` | float | - | 必传 | 注意力 logit 尺度，通常取 Dk**-0.5。 |
| `block_table` | Tensor | [B, max_pages], INT32, NPU | 必传 | 将每个请求的逻辑页映射到物理页。 |
| `context_lens` | Tensor | [B], INT32 CPU 在示例中 | 必传 | 每个请求可见的缓存 token 数。 |
| `out` | Tensor | [T, H, Dv], same dtype 与 NPU as query | 必传 | 由调用方分配，原地写入。 |
| `mla_vheadsize` | int | - | 必传 | 缓存行的潜变量前缀宽度，也是输出头宽度。 |
| `impl` | str | - | `"auto"` | `auto`, `ascend-transformer-boost`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 结果写入 out。 |

## ascend-transformer-boost 后端

调用 `torch_npu._npu_paged_attention_mla`。形状、dtype 和布局组合以已安装的 SDK 为准。 要求 SDK 提供 ATB 私有分页 MLA 入口。示例采用 512 维潜变量前缀和 64 维旋转尾部。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_paged_attention_mla.py) [Ascend Transformer Boost 源码](https://gitcode.com/cann/ascend-transformer-boost)

## torch_native 后端

读取 ND 布局的 MLA 缓存页，以完整缓存行作为 key，以该行的 latent 前缀作为 value。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_mla

heads, dk, dv = 4, 576, 512
q = torch.randn((4, heads, dk), dtype=torch.float16, device="npu")
cache = torch.randn((64, 16, 1, dk), dtype=q.dtype, device=q.device)
table = torch.arange(4, dtype=torch.int32, device=q.device).reshape(4, 1)
lengths = torch.full((4,), 4, dtype=torch.int32, device="cpu")
out = torch.empty((4, heads, dv), dtype=q.dtype, device=q.device)
paged_attention_mla(
    q, cache, 1, heads, dk**-0.5, table, lengths, dv, out, impl="ascend-transformer-boost"
)
```
