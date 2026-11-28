# fused_gdn_prefill

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
    build_gdn_prefill_chunk_meta,
    fused_gdn_prefill,
)

def fused_gdn_prefill(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_k_heads,
    num_v_heads,
    total_valid_tokens=None,
    ckpt_chunk_idx=None,
    ckpt_dst_slots=None,
    *,
    impl="auto",
):
    ...
```

## 功能说明

在 Ascend 950（`arch35`）上通过 3 阶段融合 Triton 流水线完成 Gated Delta Net（GDN）分块预填充，全程零拷贝直接从 `mixed_qkv` 读取 `V`：

1. **Stage 0（L2Norm + Gating）**：在 float32 下计算 `Q`/`K` 行级 L2 归一化（`eps = 1e-6`）与 GDN 门控：
   $$g_t = -\exp(A_{\log}) \cdot \operatorname{softplus}(a_t + \text{dt\_bias}), \quad \beta_t = \sigma(b_t)$$
   并通过 2D UB 块转置连续写出 `g_T [NV, T]` 和 `beta_T [NV, T]`。
2. **Stage 1（GQA 交错 Prepare）**：在 GQA V-head 间去重复用 $K K^\top$ 与 $Q K^\top$（内置 `GQA=3`、`2×GQA=2` 专用交错 kernel 与通用回退），完成 `BT=64` 局部 `cumsum(g)`、未缩放块内注意力 `ATTN [NT, NV, 64, 64]`、FP16 两级 8×8 分块精确 Euler 三角求逆 $(I + L)^{-1}$，以及 BF16 WY 表示 $(W, U)$。
3. **Stage 2（融合 `ChunkFwdH + ChunkFwdO`）**：在片上常驻递推状态 $H \in \mathbb{R}^{D_v \times D_k}$，原地读写 `ssm_states[cache_indices]`，支持核内直接写出单点 Radix Cache 检查点到 `ssm_states[ckpt_dst_slots]`，最终在 FP32 下统一乘缩放系数 $D^{-1/2}$。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `mixed_qkv` | `torch.Tensor` | `[T, (2*NK + NV)*128]` | `torch.bfloat16` | 必填 | NPU 上的连续打包 QKV 张量。 |
| `a` | `torch.Tensor` | `[T, NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续的 softplus 前衰减门控输入。 |
| `b` | `torch.Tensor` | `[T, NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续的 sigmoid 前 beta 输入。 |
| `A_log` | `torch.Tensor` | `[NV]` | `torch.float32` | 必填 | 每头对数衰减缩放参数。 |
| `dt_bias` | `torch.Tensor` | `[NV]` | `torch.bfloat16`、`torch.float32` | 必填 | 每头衰减偏置参数。 |
| `ssm_states` | `torch.Tensor` | `[num_slots, NV, 128, 128]` | `torch.bfloat16`、`torch.float32` | 必填 | 连续可写 SSM 状态池，原地更新。 |
| `cache_indices` | `torch.Tensor` | `[B]` | `torch.int32` | 必填 | 每条序列对应的状态槽位索引。 |
| `cu_seqlens` | `torch.Tensor` | `[B + 1]` | `torch.int32` | 必填 | 累积序列长度（`cu_seqlens[0] == 0`）。 |
| `chunk_indices` | `torch.Tensor` | `[NT, 2]` | `torch.int32` | 必填 | 由 `build_gdn_prefill_chunk_meta` 生成的 `(seq_id, chunk_id)` 映射表。 |
| `chunk_offsets` | `torch.Tensor` | `[B + 1]` | `torch.int32` | 必填 | 由 `build_gdn_prefill_chunk_meta` 生成的累积块偏移。 |
| `num_k_heads` | `int` | — | — | 必填 | Q/K 头数（2 的幂，且整除 `num_v_heads`）。 |
| `num_v_heads` | `int` | — | — | 必填 | V 头数。 |
| `total_valid_tokens` | `int` 或 `None` | — | — | `None` | 有效 token 数；尾部填充行 `[total_valid_tokens:]` 会清零。 |
| `ckpt_chunk_idx` | `torch.Tensor` 或 `None` | `[B]` | `torch.int32` | `None` | 可选，需保存检查点的块索引。 |
| `ckpt_dst_slots` | `torch.Tensor` 或 `None` | `[B]` | `torch.int32` | `None` | 检查点写入目标槽位（`< 0` 跳过）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`（仅 `ARCH35`）、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[T, NV, 128]` | `torch.bfloat16` | 新分配的预填充输出张量，位于同一 NPU 设备。 |

## triton 后端

- 仅注册于 `Generation.ARCH35`（Ascend 950），支持 `head_dim = 128`、`chunk_size = 64`、2 的幂 `num_k_heads` 及任意整数 `GQA = num_v_heads // num_k_heads`。
- 全程使用 64 位 token 步长索引（`tl.int64`），避免超长上下文 `int32` 溢出。

## torch_native 后端

- 独立的 FP32 PyTorch 逐步递推参考实现，原地更新 `ssm_states` 与可选检查点槽位。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import (
    build_gdn_prefill_chunk_meta,
    fused_gdn_prefill,
)

nk, nv, d = 16, 48, 128
seq_lens = [129, 64]
T = sum(seq_lens)
mixed_qkv = torch.randn(T, (2 * nk + nv) * d, device="npu", dtype=torch.bfloat16)
a = torch.randn(T, nv, device="npu", dtype=torch.bfloat16)
b = torch.randn(T, nv, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(nv, device="npu", dtype=torch.float32) * 0.2 - 1.0
dt_bias = torch.randn(nv, device="npu", dtype=torch.bfloat16) * 0.1
ssm_states = torch.zeros(4, nv, d, d, device="npu", dtype=torch.bfloat16)
cache_indices = torch.tensor([0, 1], device="npu", dtype=torch.int32)
cu_seqlens = torch.tensor([0, 129, 193], device="npu", dtype=torch.int32)
cidx, coff, _, total_valid = build_gdn_prefill_chunk_meta(seq_lens, mixed_qkv.device)

out = fused_gdn_prefill(
    mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices, cu_seqlens,
    cidx, coff, nk, nv, total_valid_tokens=total_valid, impl="auto",
)
```
