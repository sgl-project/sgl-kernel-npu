# fused_split_qk_norm

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import fused_split_qk_norm

def fused_split_qk_norm(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-06,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 指向相同的延迟导入入口，不自动回退。实现声明支持全部已知代际，但实际硬件执行仍待验证。

## 功能说明

将 MLA 低秩投影拆分为 Q、非旋转 K 和旋转 K。使用传入对象的权重及可选 bias，分别对 Q 和非旋转 K 做 RMSNorm；对象仅提供属性，不调用其 forward。两次归一化均使用显式传入的 eps。

$$
Z=[Z_q\mid Z_k\mid Z_r],\qquad Y_q=\frac{Z_q}{\sqrt{\mathrm{mean}(Z_q^2)+\epsilon}}w_q+b_q,\qquad Y_k=\frac{Z_k}{\sqrt{\mathrm{mean}(Z_k^2)+\epsilon}}w_k+b_k.
$$


## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `fused_qkv_a_proj_out` | Tensor | [N,Rq+Rkv+Dr] | floating | required | 连续低秩投影。 |
| `q_a_layernorm` | object | weight/bias: at least Rq | floating | required | 提供 weight 和可选 bias，与输入同设备。 |
| `kv_a_layernorm` | object | weight/bias: at least Rkv | floating | required | 提供 weight 和独立的可选 bias，与输入同设备。 |
| `q_lora_rank` | int | — | — | required | 正整数 Rq，不接受 bool。 |
| `kv_lora_rank` | int | — | — | required | 正整数 Rkv，不接受 bool。 |
| `qk_rope_dim` | int | — | — | required | 正整数 Dr，不接受 bool；旋转 K 直接复制，不归一化。 |
| `eps` | int/float | — | — | 1e-6 | 有限正 Python 标量，不接受 bool。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字参数：auto、triton、torch_native。 |

floating 指 float16、bfloat16 或 float32；参与运算的 tensor 须与输入同设备，归一化向量连续且可使用不同浮点 dtype。N 可以为零。精度比较使用有限输入。正整数选项不接受 bool。

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `q_lora` | Tensor | [N,Rq] | input.dtype | 与输入同设备的新建连续输出。 |
| `k_nope` | Tensor | [N,1,Rkv] | input.dtype | 与输入同设备的新建连续输出。 |
| `k_pe` | Tensor | [N,1,Dr] | input.dtype | 与输入同设备的新建连续输出。 |

不修改输入、权重或 bias。两后端分别使用各自 norm/norm_utils.py 检查元信息，不共享校验依赖。Triton 额外要求 NPU；非法元信息在启动前抛出 AssertionError，检查不将设备数值回读主机。

## triton 后端

保留 fused_split_qk_norm_kernel：每个 token 一个 program，方差和仿射运算使用 FP32，再转换为输出 dtype。启动前检查元信息；空 token 输入直接返回空输出，不启动 kernel。

原 `sgl_kernel_npu.norm` 脚本保持不变；这里的输入检查属于迁移后的实现。

## torch_native 后端

使用 torch 拆分及独立 FP32 RMSNorm 运算，旋转 K 单独复制；返回新 tensor，不修改输入、权重或 bias。

不调用 Triton kernel，可在 CPU 或 NPU 上独立使用。归约顺序、rsqrt 近似和乘加融合可能造成数值差异；按 dtype 容差通过不代表与框架分步路径逐位等价。

CPU 运算支持不改变包初始化要求：常规公开接口导入仍依赖 torch_npu、原生动态库及 ACL 硬件识别。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_split_qk_norm

from types import SimpleNamespace

x = torch.randn(4, 56, device='npu', dtype=torch.bfloat16)
qn = SimpleNamespace(weight=torch.ones(32, device='npu'), bias=None)
kn = SimpleNamespace(weight=torch.ones(16, device='npu'), bias=None)
args = (x, qn, kn, 32, 16, 8)
outputs = fused_split_qk_norm(*args)
reference = fused_split_qk_norm(*args, impl="torch_native")
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
