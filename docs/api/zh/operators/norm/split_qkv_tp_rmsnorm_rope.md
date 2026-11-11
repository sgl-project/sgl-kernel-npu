# split_qkv_tp_rmsnorm_rope

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 支持 | 支持 |
| `triton` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

def split_qkv_tp_rmsnorm_rope(
    input, cos, sin, q_hidden_size, kv_hidden_size, head_dim, eps, q_weight,
    k_weight, rotary_dim, tp_world, tp_group, *, impl="auto",
): ...
```

默认的 `auto` 选择 `triton`。

## 功能说明

从拼接的本地投影中拆分 Q、K、V，完成 Attention 前的数据准备，包括 MiniMax-M2 的 QK 归一化路径。算子为每个 token 分别计算 Q、K 的均方统计量；tp_world > 1 时先在 TP 组内归约，再执行带权重的 RMSNorm。归一化后的 Q/K 先舍入到输入 dtype，再对前 rotary_dim 个通道执行 NeoX RoPE，其余通道不旋转。V 只复制，不做归一化或旋转；函数在输入设备返回新的 Q、K、V Tensor。

## 参数与返回值

所有输入和输出 Tensor 均使用同一张 NPU。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, Q + 2*K]` | `float32` / `float16` / `bfloat16` | 必填 | 稠密的本地投影，依次排列 Q、K、V；Q 和 K 表示本地通道宽度。 |
| `cos` | `torch.Tensor` | `[B, R]` | `floating point` | 必填 | 稠密余弦表。每行前 R/2 个值用于该行的所有 head。 |
| `sin` | `torch.Tensor` | `[B, R]` | `floating point` | 必填 | 与 cos 配对的稠密正弦表。 |
| `q_hidden_size` | `int` | — | — | 必填 | Q 的本地宽度，须为 K 本地宽度的整数倍。 |
| `kv_hidden_size` | `int` | — | — | 必填 | K、V 各自的本地宽度，分片须包含完整 head。 |
| `head_dim` | `int` | — | — | 必填 | 每个 head 的通道数，须为 2 的幂。 |
| `eps` | `float` | — | — | 必填 | 归一化时加到均方值上的稳定项。 |
| `q_weight` | `torch.Tensor` | `[Q]` | `floating point` | 必填 | Q 各本地通道的权重。 |
| `k_weight` | `torch.Tensor` | `[K]` | `floating point` | 必填 | K 各本地通道的权重。 |
| `rotary_dim` | `int` | — | — | 必填 | 每个 head 参与旋转的前 R 个通道，R 为偶数且不大于 head_dim。 |
| `tp_world` | `int` | — | — | 必填 | TP 组的 rank 数。单卡使用 1。 |
| `tp_group` | `ProcessGroup / None` | — | — | 必填 | 单卡传 None；多卡时组内各 rank 必须一起调用，并使用相同的批量大小和分片宽度。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字，可选 `"auto"`, `"triton"`, `"torch_native"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `(q, k, v)` | `tuple[Tensor, Tensor, Tensor]` | 同输入设备及 dtype 的三个新 Tensor，形状依次为 [B, Q]、[B, K]、[B, K]。 |

## triton 后端

当 tp_world 大于 1 时，使用指定 TP 组合并 Q/K 统计量。

## torch_native 后端

在输入设备上计算，使用相同的 TP 组和输出布局。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

device = "npu"
qkv = torch.randn((2, 256), device=device, dtype=torch.bfloat16)
cos = torch.ones((2, 64), device=device, dtype=torch.bfloat16)
sin = torch.zeros_like(cos)
qw = torch.ones(128, device=device, dtype=qkv.dtype)
kw = torch.ones(64, device=device, dtype=qkv.dtype)
q, k, v = split_qkv_tp_rmsnorm_rope(
    qkv, cos, sin, 128, 64, 64, 1e-6, qw, kw, 64, 1, None, impl="triton",
)
print(q.shape, k.shape, v.shape)  # [2, 128], [2, 64], [2, 64]
```
