# fused_gdn_decode

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

def fused_gdn_decode(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source, initial_state_indices, scale=None, use_qk_l2norm_in_kernel=False, cu_seqlens=None, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

可选归一化 q/k 后，计算衰减 $d=\exp(-\exp(A_{log})\,softplus(a+dt_{bias}))$、$S_d=dS$、$r=\sigma(b)(v-S_dk)$、$S=S_d+rk^\mathsf{T}$，并输出 $o=S(q\,scale)$。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `A_log` | torch.Tensor | `[HV]; float32` | 必填 | 对数衰减系数。 |
| `a` | torch.Tensor | `[N,HV]; q.dtype` | 必填 | Softplus 门控输入。 |
| `dt_bias` | torch.Tensor | `[HV]; float32` | 必填 | Softplus 前相加的偏置。 |
| `softplus_beta` | float | `—` | 必填 | 正的 Softplus 斜率。 |
| `softplus_threshold` | float | `—` | 必填 | 超过阈值时使用线性 Softplus。 |
| `q` | torch.Tensor | `[B,T,H,K]; float16/bfloat16` | 必填 | 查询；每个序列一个 token。 |
| `k` | torch.Tensor | `[B,T,Hg,K]; float16/bfloat16` | 必填 | 分组键头；H 必须能被 Hg 整除。 |
| `v` | torch.Tensor | `[B,T,HV,V]; q.dtype` | 必填 | 值头；HV 必须能被 H 整除。 |
| `b` | torch.Tensor | `[N,HV]; q.dtype` | 必填 | Sigmoid 更新门控输入。 |
| `initial_state_source` | torch.Tensor | `[slots,HV,V,K]; float32` | 必填 | 连续状态池，原地更新；非连续输入会被复制。 |
| `initial_state_indices` | torch.Tensor | `[N]; int32/int64` | 必填 | 每个序列的状态槽；负值按零状态计算且不写回。 |
| `scale` | float 或 None | `—` | `None` | 正的查询缩放值；None 使用 K**-0.5。 |
| `use_qk_l2norm_in_kernel` | bool | `—` | `False` | 使用范数加 1e-6 对 q、k 归一化。 |
| `cu_seqlens` | torch.Tensor 或 None | `[N+1]; int32/int64` | `None` | 累计序列偏移；变长输入须展平且 B=1。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `output` | torch.Tensor 或 None | `v.shape; q.dtype` | 单 token 输出；连续状态池的选中槽会原地更新。 |

## triton 后端

递推使用 FP32。定长输入要求 T=1；变长输入要求 B=1 且每序列一个 token。状态内存按 V,K 顺序解释。

## torch_native 后端

通过独立 PyTorch 运算生成参考结果；主机元数据传输可能触发同步，仅用于精度对比。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

q = torch.randn(1, 1, 2, 64, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(1, 1, 4, 64, device="npu", dtype=torch.float16)
a = torch.randn(1, 4, device="npu", dtype=torch.float16)
b = torch.randn_like(a)
A = torch.randn(4, device="npu")
bias = torch.randn(4, device="npu")
state = torch.zeros(1, 4, 64, 64, device="npu")
idx = torch.tensor([0], device="npu", dtype=torch.int32)
y = fused_gdn_decode(A, a, bias, 1.0, 20.0, q, k, v, b, state, idx, impl="triton")
```
