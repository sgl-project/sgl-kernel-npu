# fused_gdn_verify

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
from sgl_kernel_npu.operators.attention.gdn import (
    create_gdn_verify_workspace,
    fused_gdn_verify,
)


def fused_gdn_verify(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    num_k_heads,
    num_v_heads,
    head_dim=128,
    eps=1e-6,
    workspace=None,
    linear_layer_idx=0,
    *,
    impl="auto",
): ...
```

## 功能说明

在 Ascend 950（`arch35`）上利用 Cube（AIC）加速完成投机解码阶段的 Gated Delta Net（GDN）目标验证，直接从预卷积后的 `(mixed_qkv, a, b)` 计算验证输出与紧凑提交记录 `(rec_k, rec_v, rec_g)`，验证阶段**不修改** `ssm_states` 且无需写出 `K_VER` 份完整 `[NV, 128, 128]` 中间状态：

1. **AIV 预处理**：在 56-AIV 上完成 `Q`/`K` 行级 L2 归一化（`eps = 1e-6`）、GDN 门控累计和 $G_t = \sum_{j \le t} g_j$（其中 $g_t = -\exp(A_{\log})\cdot\operatorname{softplus}(a_t + \text{dt\_bias})$）、$\beta_t = \sigma(b_t)$ 及 $V \cdot \beta$。
2. **Cube WY 分块验证**：
   - 当 `B <= 8` 或 `K_VER` 非 2 的幂时，采用单阶段 `(B * NV,)` Cube kernel，通过 FP16 精确 Euler 幂级数求逆 $(I + L)^{-1}$ 计算 WY 因子 $(W, U)$、创新向量 $V_{\text{new}} = U - W S_0^\top$ 以及输出 $O = (Q S_0^\top \exp(G) + \text{ATTN}\cdot V_{\text{new}}) D^{-1/2}$。
   - 当 `B > 8` 且 `K_VER in (2, 4, 8, 16)` 时，启用两阶段多请求 R-pack Cube 流水线：Stage 1b 将 `R_PACK = BT // K_VER` 个请求打包进单个 `[BT, 128]` Cube tile 按 `GQA` 复用 $K K^\top / Q K^\top$，Stage 2 执行无掩码连续突发访存完成 `W @ S_0^T`、`Q @ S_0^T` 与 `ATTN @ V_new`。
3. 返回 `(out, rec_k, rec_v, rec_g)`，供后续 `fused_gdn_commit` 单次全层提交接受状态。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[B, K_VER, (2*NK + NV)*128]` | `torch.bfloat16` | 必填 | NPU 上连续打包的预卷积 QKV 张量。 |
| `a` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续的 softplus 前衰减门控输入。 |
| `b` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续的 sigmoid 前 beta 输入。 |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | 必填 | 连续的每头对数衰减缩放参数。 |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续的每头衰减偏置参数。 |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16` | 必填 | 连续只读 SSM 状态池。 |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | 必填 | 每条请求的当前状态槽位（`< 0` 使用零初态）。 |
| `num_k_heads` | `int` | — | — | 必填 | Q/K 头数，需整除 `num_v_heads`。 |
| `num_v_heads` | `int` | — | — | 必填 | V 头数。 |
| `head_dim` | `int` | — | — | `128` | 单头维度，仅支持 `128`。 |
| `eps` | `float` | — | — | `1e-6` | L2 归一化数值稳定项。 |
| `workspace` | `FusedGDNVerifyWorkspace` 或 `None` | — | — | `None` | 可选的持久化静态工作区（由 `create_gdn_verify_workspace` 创建），用于 ACLGraph 零分配复用。 |
| `linear_layer_idx` | `int` | — | — | `0` | 当前线性注意力层在 `workspace` 中的索引。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`（仅 `ARCH35`）、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B, K_VER, NV, 128]` | `torch.bfloat16` | 验证阶段注意力输出。 |
| `rec_k` | `torch.Tensor` | `[B, K_VER, NK, 128]` | `torch.bfloat16` | L2 归一化后的 K 提交记录。 |
| `rec_v` | `torch.Tensor` | `[B, K_VER, NV, 128]` | `torch.bfloat16` | WY 创新向量 $V_{\text{new}}$ 提交记录。 |
| `rec_g` | `torch.Tensor` | `[B, K_VER, NV]` | `torch.float32` | 累积对数衰减 $G_t$ 提交记录。 |

## triton 后端

- 仅注册于 `Generation.ARCH35`（Ascend 950），支持 `2 <= K_VER <= 16`、任意整数 `GQA = NV // NK >= 1` 与 `head_dim = 128`。
- 当传入已预分配的 `workspace` 时，所有中间 scratch 缓冲区均静态复用，完整兼容 ACLGraph 捕获。

## torch_native 后端

- 独立的 FP32 逐步递推 GDN 参考实现（不依赖 WY 矩阵分解），同步计算精确 Delta-rule 创新向量 `rec_v` 与输出 `out`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_verify

B, K, NK, NV, D = 4, 4, 16, 48, 128
mixed_qkv = torch.randn(B, K, (2 * NK + NV) * D, device="npu", dtype=torch.bfloat16)
a = torch.randn(B, K, NV, device="npu", dtype=torch.bfloat16)
b = torch.randn(B, K, NV, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(NV, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(NV, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(8, NV, D, D, device="npu", dtype=torch.bfloat16)
cache_indices = torch.arange(B, device="npu", dtype=torch.int32)

out, rec_k, rec_v, rec_g = fused_gdn_verify(
    mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, NK, NV, impl="auto"
)
```
