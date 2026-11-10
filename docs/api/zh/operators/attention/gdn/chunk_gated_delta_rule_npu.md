# chunk_gated_delta_rule_npu

## 硬件支持

| 实现 | ARCH22 | ARCH35 |
| --- | --- | --- |
| `triton`（也是 `auto`） | 本组合接口尚未验证 | 本组合接口尚未验证 |

## 函数签名

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

def chunk_gated_delta_rule_npu(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, head_first=False,
    use_qk_l2norm_in_kernel=False, *, impl="auto",
):
    ...
```

`auto` 选择 `triton`。本接口对应旧 `fla.chunk` 的组合 wrapper，
与融合 CANN 接口 `chunk_gated_delta_rule` 分别注册。

## 功能描述

原样复制旧前向编排函数，只将实现 import 指向迁移后的 implementation：
可选 Q/K L2 归一化、分块累加、scaled KKT、三角求逆、WY 重算、分块状态更新、输出计算。
默认分块大小为 64；保留原实现的中间 dtype、类型转换和 launch 选择。

数学上的逐 token 递推为
`S_t = exp(g_t) * S_(t-1) + beta_t * k_t ⊗ (v_t - k_t^T exp(g_t) S_(t-1))`，
`o_t = scale * q_t^T S_t`。每 H/Hg 个 value head 共享一个 Q/K head。
可选归一化在递推前计算 `x / sqrt(sum(x*x) + 1e-6)`。
分块实现保留既有低精度舍入位置。

## 参数与返回值

| 参数 | 类型 / 形状 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `q`, `k` | NPU Tensor `[B,T,Hg,K]` | 必填 | FP16/BF16，dtype 和设备一致。 |
| `v` | NPU Tensor `[B,T,H,V]` | 必填 | 与 q/k 同 dtype；H 是 Hg 的整数倍。 |
| `g` | NPU Tensor `[B,T,H]` | 必填 | 逐 token 的对数衰减，通常 FP32；尚未累加。 |
| `beta` | NPU Tensor `[B,T,H]` | 必填 | 更新权重。 |
| `scale` | float 或 None | None | None 对应 `K**-0.5`；保留显式零值。 |
| `initial_state` | NPU Tensor `[N,H,K,V]` 或 None | None | None 对应零初始状态。 |
| `output_final_state` | bool | True | 为 false 时 final_state 返回 None。 |
| `cu_seqlens` | NPU int32/int64 Tensor `[N+1]` 或 None | None | packed B=1 边界；首项 0、末项 T，各序列长度为正。 |
| `head_first` | bool | False | True 保持抛出旧 DeprecationWarning 异常。 |
| `use_qk_l2norm_in_kernel` | bool | False | 使用既有 L2 实现归一化 Q/K。 |
| `impl` | str | `"auto"` | `auto` 或 `triton`，不自动回退。 |

| 返回值 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `o` | `[B,T,H,V]`，q.dtype | 注意力输出。 |
| `final_state` | `[N,H,K,V]`，FP32 或 None | 最终递推状态。 |
| `h` | packed 输入为 `[1,total_chunks,H,K,V]`，q.dtype | 按序列、分块顺序排列的进入分块前的状态。 |

## Triton 实现

保留旧 input guard：输入转为 contiguous，并切换至输入 Tensor 对应的 NPU 设备上下文。
保留 `torch.compiler.disable`，不新增整图支持。普通路径采用迁移后的状态 kernel
支持的 K,V <= 128。使用固定 B=1 或 packed B=1 多序列输入；固定 B>1 继承
状态/输出 kernel 的 gate 偏移限制。本次不新增参数校验或修复 kernel。
不要原地修改已缓存的序列边界。

`GDN_USE_MEGA_GDN=1` 选择逐字节复制的旧 mega helper（128-token 分块，
head dimension 128，按 B=1 组织输入），需要在独立 ARCH22 构建中选择迁移后的原生算子
`mega_chunk_gdn`。原样 helper 位于 `implementation/ascendc/attention/gdn`，
参见 [mega_chunk_gdn](mega_chunk_gdn.md)。`GDN_RECOMPUTE_SUPPRESS_LEVEL` 保留导入时读取、
控制中间结果保留的行为。两个开关均保持旧行为；默认路径调用迁移后的 Triton 实现。

本次仅重定向 import，不新增生产用 `torch_native` 后端。测试采用独立逐 token
递推参考和旧 wrapper，对比三项返回。设备精度和性能仍需在 A3/A5 上运行验证。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

q = torch.randn(1, 65, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(1, 65, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 65, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 65, 8, device="npu", dtype=q.dtype)
cu_seqlens = torch.tensor([0, 31, 65], device="npu", dtype=torch.int32)
o, final_state, h = chunk_gated_delta_rule_npu(
    q, k, v, g, beta, cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=True, impl="triton",
)
```
