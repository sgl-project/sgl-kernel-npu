# fused_gdn_commit

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
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_commit


def fused_gdn_commit(
    ssm_states_all,
    rec_k_all,
    rec_v_all,
    rec_g_all,
    dst_slots,
    last_steps,
    *,
    track_slots=None,
    track_steps=None,
    impl="auto",
): ...
```

## 功能说明

在 Ascend 950（`arch35`）上通过单次 28-AIC 持久化 Cube kernel 启动，跨所有线性注意力层原地提交投机解码接受后的 SSM 状态：
$$S_{\text{last}} = \exp(G_{\text{last}}) \cdot S_0 + \sum_{t=0}^{\text{last}} \exp(G_{\text{last}} - G_t)\, v_t k_t^\top$$
当同时传入 `track_slots` 与 `track_steps` 时，在覆盖 `dst_slots[b]` 之前直接基于初态 $S_0$ 计算并写出 Radix Cache 页边界检查点 $S_{\text{track}}$ 到 `ssm_states_all[:, track_slots[b]]`。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `ssm_states_all` | `torch.Tensor` | `[L, num_slots, NV, 128, 128]` | `torch.bfloat16` | 必填 | 全层 SSM 状态池，原地更新。 |
| `rec_k_all` | `torch.Tensor` | `[L, max_bs, K_VER, NK, 128]` | `torch.bfloat16` | 必填 | 验证阶段产出的归一化 K 记录。 |
| `rec_v_all` | `torch.Tensor` | `[L, max_bs, K_VER, NV, 128]` | `torch.bfloat16` | 必填 | 验证阶段产出的 WY 创新向量记录。 |
| `rec_g_all` | `torch.Tensor` | `[L, max_bs, K_VER, NV]` | `torch.float32` | 必填 | 验证阶段产出的累积对数衰减记录。 |
| `dst_slots` | `torch.Tensor` | `[B]` | 整型 | 必填 | 目标状态槽位索引（`< 0` 跳过）。 |
| `last_steps` | `torch.Tensor` | `[B]` | 整型 | 必填 | 每条请求的接受步索引，范围 `[0, K_VER - 1]`（`< 0` 跳过）。 |
| `track_slots` | `torch.Tensor` 或 `None` | `[B]` | 整型 | `None` | 可选的 Radix 检查点槽位（`< 0` 跳过）。 |
| `track_steps` | `torch.Tensor` 或 `None` | `[B]` | 整型 | `None` | 可选的 Radix 检查点步索引（`< 0` 跳过）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`（仅 `ARCH35`）、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `result` | `torch.Tensor` | `[L, num_slots, NV, 128, 128]` | `torch.bfloat16` | 原地更新后的 `ssm_states_all` 别名。 |

## triton 后端

- 仅注册于 `Generation.ARCH35`（Ascend 950），使用 `min(L * B * NV, 28)` 持久化 Cube 核全层一次提交，采用 `tl.int64` 偏移支持超大规模状态池，并兼容任意整数 `GQA = NV // NK >= 1` 与 `2 <= K_VER <= 16`。

## torch_native 后端

- 独立的 FP32 PyTorch 矩阵求和参考实现，原地更新 `ssm_states_all` 与可选检查点槽位。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_commit

L, S, B, K, NK, NV, D = 4, 16, 4, 4, 16, 48, 128
ssm_all = torch.randn(L, S, NV, D, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_k = torch.randn(L, B, K, NK, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_v = torch.randn(L, B, K, NV, D, device="npu", dtype=torch.bfloat16) * 0.1
rec_g = -torch.cumsum(torch.rand(L, B, K, NV, device="npu", dtype=torch.float32) * 0.1, dim=2)
dst_slots = torch.arange(B, device="npu", dtype=torch.int64)
last_steps = torch.tensor([0, 1, 2, 3], device="npu", dtype=torch.int64)

fused_gdn_commit(ssm_all, rec_k, rec_v, rec_g, dst_slots, last_steps, impl="auto")
```
