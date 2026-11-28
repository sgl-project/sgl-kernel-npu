# fused_conv1d_decode_gating

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 支持 |
| `torch_native` | 待验证 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一入口函数。

```python
from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_decode_gating

def fused_conv1d_decode_gating(
    projected_states_qkvz,
    projected_states_ba,
    conv_weights_t,
    conv_state,
    cache_indices,
    A_log,
    dt_bias,
    num_v_heads,
    *,
    impl="auto",
):
    ...
```

## 功能说明

直接从打包投影张量 `projected_states_qkvz` 与 `projected_states_ba` 中融合执行单步解码深度因果卷积（卷积核宽 4 + SiLU）、原地滚动 `conv_state` 移位更新及 GDN 门控计算：

对每个有效样本 $b$（槽位 $s = \text{cache\_indices}[b] \ge 0$，$x_0 = \text{qkvz}[b, :D_{\text{qkv}}]$）：

$$y_b = \operatorname{SiLU}\!\left(S_{s,0} w_0 + S_{s,1} w_1 + S_{s,2} w_2 + x_0 w_3\right), \quad (S_{s,0}, S_{s,1}, S_{s,2}) \leftarrow (S_{s,1}, S_{s,2}, x_0)$$

$$g_b = -\exp(A_{\log}) \cdot \operatorname{softplus}(a_b + \text{dt\_bias}), \quad \beta_b = b_{\text{raw}}$$

当 $s < 0$（填充槽位）时直接透传 $y_b = x_0$ 且不修改 `conv_state`；`beta` 输出未经过 sigmoid 的原始 logit 以匹配 `recurrent_gated_delta_rule`。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `projected_states_qkvz` | `torch.Tensor` | `[B, QKVZ_DIM]` | `torch.bfloat16`、`torch.float16` | 必填 | 连续打包投影张量，前 `QKV_DIM` 列为 `qkv`。 |
| `projected_states_ba` | `torch.Tensor` | `[B, 2*NV]` | `torch.bfloat16`、`torch.float16` | 必填 | 连续打包的 `[b_raw, a_raw]` 张量。 |
| `conv_weights_t` | `torch.Tensor` | `[4, QKV_DIM]` | `torch.bfloat16`、`torch.float16` | 必填 | 连续转置深度卷积权重（`QKV_DIM % 512 == 0`）。 |
| `conv_state` | `torch.Tensor` | `[num_slots, 3, QKV_DIM]` | `torch.bfloat16`、`torch.float16` | 必填 | 连续可写滚动卷积状态池，原地更新。 |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32`、`torch.int64` | 必填 | 每个样本对应的槽位索引（`< 0` 表示填充槽位）。 |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | 必填 | 每头对数衰减缩放参数。 |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`、`torch.float16`、`torch.float32` | 必填 | 每头衰减偏置参数。 |
| `num_v_heads` | `int` | — | — | 必填 | V 头数 `NV`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[B, QKV_DIM]` | 同 `projected_states_qkvz` | 有效槽位的 Conv1d+SiLU 输出，填充槽位直接透传。 |
| `g` | `torch.Tensor` | `[B, NV]` | `torch.float32` | 对数衰减门控值。 |
| `beta` | `torch.Tensor` | `[B, NV]` | 同 `projected_states_qkvz` | 拆包后的 pre-sigmoid beta logit。 |

## triton 后端

- 当 `B >= 2` 且 `QKV_DIM % 2048 == 0` 时采用 `BLOCK_D = 2048`（否则 `1024` 或 `512`），并使用 `tl.int64` 槽位与批次索引。
- 兼容 NPU Decode Graph 捕获与回放。

## torch_native 后端

- 独立的 PyTorch 参考实现，使用 float32 累加完成卷积、SiLU、softplus 门控与原地状态移位。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_decode_gating

bs, nk, nv, d = 8, 16, 48, 128
qkv_dim = (2 * nk + nv) * d
qkvz = torch.randn(bs, qkv_dim + nv * d, device="npu", dtype=torch.bfloat16)
ba = torch.randn(bs, 2 * nv, device="npu", dtype=torch.bfloat16)
w_t = torch.randn(4, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
state = torch.randn(16, 3, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
cache_idx = torch.arange(bs, device="npu", dtype=torch.int32)
A_log = torch.randn(nv, device="npu", dtype=torch.float32)
dt_bias = torch.randn(nv, device="npu", dtype=torch.bfloat16)

mixed_qkv, g, beta = fused_conv1d_decode_gating(
    qkvz, ba, w_t, state, cache_idx, A_log, dt_bias, nv, impl="auto"
)
```
