# fused_conv1d_gdn_verify

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 不支持 | 支持 |
| `torch_native` | 待验证 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

默认选项为 `auto`，在 `Generation.ARCH35` 上与 `triton` 注册到同一入口函数。

```python
from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_gdn_verify


def fused_conv1d_gdn_verify(
    qkvz,
    ba,
    conv_weights_t,
    conv_states,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    batch_size,
    k_ver,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    query_start_loc=None,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
): ...
```

## 功能说明

在 Ascend 950（`arch35`）上将投机解码验证阶段的深度因果卷积（`kernel_size=4` + SiLU + 原地 `conv_states` 滑动更新）、`Q/K` 行级 L2 归一化、GDN 门控与 Cube WY 分块验证完全融合：直接从跨步打包输入 `qkvz[:, :QKV_DIM]` 与 `ba` 读取数据，省去 `Slice`、独立 `causal_conv1d` 及中间 `mixed_qkv` 写回。

对于有效槽位（`cache_indices[b] >= 0` 且在提供 `query_start_loc` 时满足 `query_start_loc[b+1] > query_start_loc[b]`），以 `[old[W-3], old[W-2], old[W-1]]` 为初始 3-tap 历史并原地更新为 `[old[W-2], old[W-1], x_0, ..., x_{K-1}]`；ACLGraph 填充行自动跳过状态写回，避免污染槽位 0。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `qkvz` | `torch.Tensor` | `[B * k_ver, qkvz_dim]` | `torch.bfloat16` | 必填 | 连续打包投影输出，需满足 `qkvz_dim >= (2*NK + NV)*128`。 |
| `ba` | `torch.Tensor` | `[B * k_ver, 2 * NV]` | `torch.bfloat16` | 必填 | 连续打包 `[b \| a]` 门控投影输出。 |
| `conv_weights_t` | `torch.Tensor` | `[4, QKV_DIM]` | `torch.bfloat16` | 必填 | 转置后的连续 depthwise Conv1D 权重。 |
| `conv_states` | `torch.Tensor` | `[num_slots, k_ver + 2, QKV_DIM]` | `torch.bfloat16` | 必填 | 连续卷积历史状态池，原地更新。 |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | 必填 | 连续每头对数衰减参数。 |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续每头衰减偏置参数。 |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16` | 必填 | 连续只读 SSM 状态池。 |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | 必填 | 每条请求的槽位索引。 |
| `batch_size` | `int` | — | — | 必填 | 请求数 `B`。 |
| `k_ver` | `int` | — | — | 必填 | 验证宽度（`2 <= k_ver <= 16`）。 |
| `num_k_heads` | `int` | — | — | 必填 | Q/K 头数（整除 `num_v_heads`）。 |
| `num_v_heads` | `int` | — | — | 必填 | V 头数。 |
| `head_dim` | `int` | — | — | `128` | 单头维度（仅支持 `128`）。 |
| `eps` | `float` | — | — | `1e-6` | L2 归一化数值稳定项。 |
| `query_start_loc` | `torch.Tensor` 或 `None` | `[B + 1]` | `torch.int32` | `None` | 可选累积偏移；`qsl[b+1] <= qsl[b]` 的填充行跳过 `conv_states` 写回。 |
| `workspace` | `FusedGDNVerifyWorkspace` 或 `None` | — | — | `None` | 可选静态工作区。 |
| `linear_layer_idx` | `int` | — | — | `0` | 当前层在 `workspace` 中的索引。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`（仅 `ARCH35`）、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B, k_ver, NV, 128]` | `torch.bfloat16` | 验证阶段注意力输出。 |
| `rec_k` | `torch.Tensor` | `[B, k_ver, NK, 128]` | `torch.bfloat16` | L2 归一化 K 提交记录。 |
| `rec_v` | `torch.Tensor` | `[B, k_ver, NV, 128]` | `torch.bfloat16` | WY 创新向量提交记录。 |
| `rec_g` | `torch.Tensor` | `[B, k_ver, NV]` | `torch.float32` | 累积对数衰减提交记录。 |

## triton 后端

- 仅注册于 `Generation.ARCH35`（Ascend 950），单次 56-AIV 融合核完成卷积、归一化与门控，随后进入与 `fused_gdn_verify` 相同的 Cube WY 验证流水线。

## torch_native 后端

- 独立的 FP32 PyTorch 因果 Conv1D + SiLU + 逐步递推 GDN 参考实现，原地更新 `conv_states`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_conv1d_gdn_verify

B, K, NK, NV, D = 4, 4, 16, 48, 128
qkv_dim = (2 * NK + NV) * D
qkvz = torch.randn(B * K, qkv_dim + NV * D, device="npu", dtype=torch.bfloat16)
ba = torch.randn(B * K, 2 * NV, device="npu", dtype=torch.bfloat16)
conv_w_t = torch.randn(4, qkv_dim, device="npu", dtype=torch.bfloat16) * 0.1
conv_states = torch.zeros(8, K + 2, qkv_dim, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(NV, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(NV, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(8, NV, D, D, device="npu", dtype=torch.bfloat16)
cache_indices = torch.arange(B, device="npu", dtype=torch.int32)

out, rec_k, rec_v, rec_g = fused_conv1d_gdn_verify(
    qkvz, ba, conv_w_t, conv_states, A_log, dt_bias, ssm_states,
    cache_indices, B, K, NK, NV, impl="auto",
)
```
