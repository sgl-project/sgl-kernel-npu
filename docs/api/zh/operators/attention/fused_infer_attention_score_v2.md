# fused_infer_attention_score_v2

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.attention import (
    fused_infer_attention_score_v2,
    fused_infer_attention_score_v2_out,
    fused_infer_attention_score_v2_get_max_workspace,
)


def fused_infer_attention_score_v2(query, key, value, *, impl="auto", **kwargs): ...


def fused_infer_attention_score_v2_out(
    query, key, value, *, out, impl="auto", **kwargs
): ...


def fused_infer_attention_score_v2_get_max_workspace(
    query, key, value, *, impl="auto", **kwargs
): ...
```

## 功能说明

执行融合解码或预填充注意力。对稠密非量化输入，计算 $Y=\mathrm{softmax}(sQK^\top+M)V$，其中 $M$ 为可选加性掩码或偏置。按需返回逐行 log-sum-exp。out 重载写入调用方提供的缓冲区。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Hq, Sq, D]，用于 BNSD, FP16/BF16，用于 example | 必传 | query Tensor。其他布局和量化模式按 SDK 定义。 |
| `key`, `value` | Tensor | [B, Hkv, Skv, D]，用于 BNSD, query 的 dtype，用于 example | 必传 | KV 长度匹配的键值 Tensor。SDK 还支持分页布局。 |
| `num_query_heads`, `num_key_value_heads` | int | - | `1` / `0` | 可选 query 和 KV 头数，默认为 1 和 0，0 表示推断 KV 头数。 |
| `softmax_scale` | float | - | `1.0` | 可选 logit 尺度，默认 1.0。 |
| `input_layout` | str | - | `"BSH"` | 可选 SDK 布局标签，默认 BSH。示例采用 BNSD。 |
| `actual_seq_qlen`, `actual_seq_kvlen` | sequence[int] or None | [B] 普通分批布局 | `None` | 可选有效 query 和 KV 长度，解释方式取决于输入布局。 |
| `block_table`, `atten_mask` | Tensor or None | INT32 page table 与 SDK-compatible mask | `None` | 可选逻辑页映射和掩码。 |
| `return_softmax_lse` | bool | - | `False` | 可选，设为 true 时返回 LSE，默认 false。 |
| `out` | sequence[Tensor] | two buffers matching the output shape 与 dtype | 必传 | out 重载必传，依次为注意力输出和 LSE 缓冲区。 |
| `workspace` | Tensor or None | SDK workspace | SDK 默认值 | out 重载的可选工作区，由对应辅助函数以相同参数获取。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `attention_output`, `softmax_lse` | tuple[Tensor, Tensor] | [B, Hq, Sq, D] 在示例中; FP32 LSE | 主调用返回两个 Tensor。BNSD 下启用 LSE 时形状为 [B, Hq, Sq, 1]。out 重载使用传入的缓冲区。 |
| `workspace` | Tensor | SDK workspace shape 与 dtype | 工作区辅助函数为相同配置返回此缓冲区。 |

## torch_npu 后端

调用 `torch_npu.npu_fused_infer_attention_score_v2`。形状、dtype 和布局组合以已安装的 SDK 为准。 可传递已安装 SDK 支持的选项。V2 使用 softmax_scale 和 return_softmax_lse。主调用、工作区辅助函数和 out 重载的布局、可选输入及标志应一致。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    fused_infer_attention_score_v2,
    fused_infer_attention_score_v2_out,
    fused_infer_attention_score_v2_get_max_workspace,
)

q = torch.randn((1, 32, 1, 128), dtype=torch.float16, device="npu")
k = torch.randn((1, 32, 512, 128), dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
kwargs = dict(
    num_query_heads=32,
    softmax_scale=128**-0.5,
    input_layout="BNSD",
    return_softmax_lse=True,
    impl="torch_npu",
)
y, lse = fused_infer_attention_score_v2(q, k, v, **kwargs)
workspace = fused_infer_attention_score_v2_get_max_workspace(q, k, v, **kwargs)
out = [torch.empty_like(y), torch.empty_like(lse)]
fused_infer_attention_score_v2_out(q, k, v, out=out, workspace=workspace, **kwargs)
```
