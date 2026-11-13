# split_qkv_rmsnorm_mrope

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_mrope


def split_qkv_rmsnorm_mrope(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
    *,
    impl="auto"
): ...
```

## 功能描述

拆分打包的 Q/K/V 及可选逐头 gate，对 Q/K 应用带权重及可选偏置的 RMSNorm，再应用时间、高度、宽度多轴旋转位置编码。支持全维/部分旋转及交错/连续 MRoPE 分段；V 和 gate 不做归一化。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `qkv` | Tensor | `Required` | 连续 BF16/FP16 [tokens,QH*H*(2 if has_gate else 1)+2*KVH*H]。 |
| `q_weight` | Tensor | `Required` | 浮点 query RMSNorm 权重 [head_size]。 |
| `k_weight` | Tensor | `Required` | 浮点 key RMSNorm 权重 [head_size]。 |
| `cos_sin` | Tensor | `Required` | 浮点 [3,tokens,rope_dim]，每个轴先 cosine 后 sine。 |
| `num_q_heads` | int | `Required` | 张量并行切分后的本地 query 头数。 |
| `num_kv_heads` | int | `Required` | 本地 KV 头数。 |
| `head_size` | int | `Required` | Q/K/V 每头通道数。 |
| `eps` | float | `Required` | RMSNorm 稳定常数。 |
| `mrope_section` | list[int] | `Required` | 时间、高度、宽度三个旋转分段。 |
| `is_interleaved` | bool | `Required` | 交错排列旋转轴，而非连续分段。 |
| `rope_dim` | int or None | `None` | 偶数旋转宽度，<=head_size；None 旋转整个头。 |
| `q_bias` | Tensor or None | `None` | 可选浮点 query 偏置 [head_size]。 |
| `k_bias` | Tensor or None | `None` | 与 q_bias 配对的同形状 key 偏置。 |
| `has_gate` | bool | `False` | 输入包含每个 query 头的 gate 通道。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor,Tensor,Tensor] | Q [T,QH*H]，K/V [T,KVH*H]，gate [T,QH*H] 或 [T,0]，类型均与 qkv 相同。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_mrope

qkv = torch.randn(7, 1024, device="npu", dtype=torch.bfloat16)
w = torch.ones(256, device="npu", dtype=qkv.dtype)
cs = torch.randn(3, 7, 256, device="npu", dtype=qkv.dtype)
q, k, v, gate = split_qkv_rmsnorm_mrope(
    qkv, w, w, cs, 2, 1, 256, 1e-6, [32, 48, 48], True, impl="triton"
)
```
