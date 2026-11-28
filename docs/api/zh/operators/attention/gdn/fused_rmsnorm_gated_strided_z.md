# fused_rmsnorm_gated_strided_z

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
from sgl_kernel_npu.operators.attention.gdn import fused_rmsnorm_gated_strided_z

def fused_rmsnorm_gated_strided_z(
    core_attn_out,
    projected_states_qkvz,
    norm_weight,
    z_offset,
    eps=1e-6,
    *,
    impl="auto",
):
    ...
```

## 功能说明

直接从打包张量 `projected_states_qkvz` 的跨步切片 `[:, z_offset : z_offset + NV * D]` 中读取 `z`，避免整张量拷贝，融合计算每头门控 RMSNorm：

$$Y_{t,h,j} = \frac{X_{t,h,j}}{\sqrt{\frac{1}{D}\sum_{k=0}^{D-1} X_{t,h,k}^2 + \varepsilon}} \cdot w_j \cdot \operatorname{SiLU}(Z_{t,h,j})$$

归约、权重缩放与 SiLU 门控均在 float32 中完成，最终写回为 `core_attn_out.dtype`。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `core_attn_out` | `torch.Tensor` | `[T, NV, D]` | `torch.bfloat16`、`torch.float16` | 必填 | NPU 上的连续注意力输出张量。 |
| `projected_states_qkvz` | `torch.Tensor` | `[T, QKVZ_DIM]` | `torch.bfloat16`、`torch.float16` | 必填 | NPU 上的连续打包 QKVZ 投影张量。 |
| `norm_weight` | `torch.Tensor` | `[D]` | `torch.bfloat16`、`torch.float16`、`torch.float32` | 必填 | 连续的每头 RMSNorm 权重向量。 |
| `z_offset` | `int` | — | — | 必填 | `z` 在 `projected_states_qkvz` 中的起始列偏移（满足 `z_offset + NV * D <= QKVZ_DIM`）。 |
| `eps` | `float` | — | — | `1e-6` | 加在均方值上的稳定项。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[T, NV * D]` | 同 `core_attn_out` | 新分配的连续门控 RMSNorm 输出。 |

## triton 后端

- 最多启动 56 个持久化 AIV program；在 `T >= 56` 且 `NV <= 64` 时按整行 `HEADS_PER_PROG = NV` 调度，小批次解码（`T <= 16`）自适应切分头组。
- 使用 `tl.int64` token 步长索引，防止超长序列（`T > 131072`）出现 `int32` 溢出。

## torch_native 后端

- 独立的 PyTorch 参考实现，对切片 `z` 执行 float32 RMSNorm 与 `F.silu`。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_rmsnorm_gated_strided_z

T, nk, nv, d = 32, 16, 48, 128
qkv_dim = (2 * nk + nv) * d
core = torch.randn(T, nv, d, device="npu", dtype=torch.bfloat16)
qkvz = torch.randn(T, qkv_dim + nv * d, device="npu", dtype=torch.bfloat16)
w = torch.randn(d, device="npu", dtype=torch.bfloat16)

y = fused_rmsnorm_gated_strided_z(core, qkvz, w, qkv_dim, eps=1e-6, impl="auto")
```
