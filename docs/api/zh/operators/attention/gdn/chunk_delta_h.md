# chunk_delta_h

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_delta_h

def chunk_delta_h(k, w, u, g=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, cu_seqlens=None, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

保存块前状态 S，计算 $D=U-WS$，再更新 $S\leftarrow S\exp(g_L)+K^\mathsf{T}(D\odot\exp(g_L-g))$。正门控差对应的衰减系数置零。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `k` | torch.Tensor | `[B,T,Hg,K]; float16/bfloat16` | 必填 | 分组键头；H 必须能被 Hg 整除。 |
| `w` | torch.Tensor | `[B,T,H,K]; k.dtype` | 必填 | 用于状态投影的变换键。 |
| `u` | torch.Tensor | `[B,T,H,V]; k.dtype` | 必填 | 变换值；不修改输入。 |
| `g` | torch.Tensor | `[B,T,H]; float32` | `None` | 对数域标量门控；当前后端要求提供，不能为 None。 |
| `initial_state` | torch.Tensor 或 None | `[N,H,K,V]; float32` | `None` | 初始状态；None 表示零状态。 |
| `output_final_state` | bool | `—` | `False` | 为真时返回最终 FP32 状态。 |
| `chunk_size` | int | `—` | `64` | 正的二次幂块长；GDN 使用 64。 |
| `save_new_value` | bool | `—` | `True` | 为真时返回残差新值。 |
| `cu_seqlens` | torch.Tensor 或 None | `[N+1]; int32/int64` | `None` | 累计序列偏移；变长输入须展平且 B=1。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `h` | torch.Tensor 或 None | `[B,NT,H,K,V]; k.dtype` | 块前状态快照；变长展平输入的 NT 为所有序列块数之和。 |
| `v_new` | torch.Tensor 或 None | `u.shape; u.dtype 或 None` | save_new_value 为真时返回残差 D。 |
| `final_state` | torch.Tensor 或 None | `[N,H,K,V]; float32 或 None` | output_final_state 为真时返回最终状态。 |

## triton 后端

状态使用 FP32 累加，投影相关中间量按输入 dtype 舍入。模型用例为 K=V=128，V=64 属于额外检查；当前固定宽度内核不支持 K>128 或 V>128。

公开适配层拒绝 K/V 不在 [1,128] 内的 Triton 输入，分块长度仅接受已验证的 64；其他正二次幂长度可用于 torch_native 后端。

## torch_native 后端

CPU FP64 参考保留两个输入 dtype 舍入点，输出遵循接口类型并返回输入设备，仅用于精度比较。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_delta_h

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float16)
w = torch.randn(1, 64, 4, 128, device="npu", dtype=torch.float16)
u = torch.randn_like(w)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
h, v_new, final = chunk_delta_h(k, w, u, g, output_final_state=True, impl="triton")
```
