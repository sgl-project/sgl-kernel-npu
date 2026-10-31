# split_qkv_rmsnorm_rope

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 指向相同的延迟导入入口，不自动回退。实现声明支持全部已知代际，但实际硬件执行仍待验证。

## 功能说明

拆分拼接的 Q/K/V。eps 非 None 时，按 head 分别对 Q/K 做 RMSNorm 并应用成对的可选 bias；旋转每个 head 的前 R 个特征，保留剩余特征。V 直接复制。

$$
A(x,w,b)=\begin{cases}x,&\epsilon=\mathrm{None},\\x\,\mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)w+b,&\text{otherwise},\end{cases}\quad Y_{q,k}=\mathrm{RoPE}_R(A(X_{q,k})),\quad Y_v=X_v.
$$

每个 head 的前 R 个特征按 $y_i=x_i c_i+ho(x)_i s_i$ 旋转；NeoX 的 $ho$ 配对前后两半，GPT-J 配对相邻特征。尾部保留归一化后的值。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | Tensor | [N,Q+2K] | floating | required | 连续拼接的 Q/K/V。 |
| `sin` | Tensor | [N,R] or [N,1,1,R] | floating | required | 连续的重复半表 sin，恰好 N*R 个元素。 |
| `cos` | Tensor | [N,R] or [N,1,1,R] | floating | required | 连续 cos，旋转维度 R 与 sin 相同。 |
| `q_hidden_size` | int | — | — | required | 正整数 Q，必须能被 K 整除。 |
| `kv_hidden_size` | int | — | — | required | 正整数 K，必须能被 head_dim 整除。 |
| `head_dim` | int | — | — | required | 每个 head 的正整数维度 D，不接受 bool。 |
| `eps` | int/float/None | — | — | None | None 跳过归一化，否则为有限正 Python 标量。 |
| `q_weight` | Tensor/None | at least D values | floating | None | 启用归一化时必填，连续且与输入同设备。 |
| `k_weight` | Tensor/None | at least D values | floating | None | 启用归一化时必填，连续且与输入同设备。 |
| `q_bias` | Tensor/None | at least D values | floating | None | 可选，须与 k_bias 成对提供；仅在归一化时应用。 |
| `k_bias` | Tensor/None | at least D values | floating | None | 可选，须与 q_bias 成对提供；连续且与输入同设备。 |
| `is_neox_style` | bool | — | — | True | True 配对前后两半，False 使用前半表频率配对相邻通道。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字参数：auto、triton、torch_native。 |

floating 指 float16、bfloat16 或 float32；参与运算的 tensor 须与输入同设备，归一化向量连续且可使用不同浮点 dtype。N 可以为零。精度比较使用有限输入。正整数选项不接受 bool。

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | 与输入同设备的新建连续输出。 |
| `k` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |
| `v` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |

不修改输入、权重或 bias。两后端分别使用各自 norm/norm_utils.py 检查元信息，不共享校验依赖。Triton 额外要求 NPU；非法元信息在启动前抛出 AssertionError，检查不将设备数值回读主机。

## triton 后端

保留 split_qkv_rmsnorm_rope_kernel 及按 KV head 划分的列 grid。Q/K 归一化与 RoPE 使用 FP32，在写输出时转换 dtype，不增加中间激活类型转换。空 token 输入不启动 kernel。

原 `sgl_kernel_npu.norm` 脚本保持不变；这里的输入检查属于迁移后的实现。

## torch_native 后端

使用独立的 PyTorch FP32 RMSNorm 和旋转运算。NeoX 配对前后两半，GPT-J 配对相邻通道，并将缓存前半部分频率逐元素重复。V 始终复制，包括跳过归一化时。

不调用 Triton kernel，可在 CPU 或 NPU 上独立使用。归约顺序、rsqrt 近似和乘加融合可能造成数值差异；按 dtype 容差通过不代表与框架分步路径逐位等价。

CPU 运算支持不改变包初始化要求：常规公开接口导入仍依赖 torch_npu、原生动态库及 ACL 硬件识别。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

x = torch.randn(4, 256, device='npu', dtype=torch.bfloat16)
sin = torch.zeros(4, 1, 1, 32, device='npu')
cos = torch.ones_like(sin)
wq = torch.ones(64, device='npu')
wk = torch.ones_like(wq)
args = (x, sin, cos, 128, 64, 64)
options = dict(eps=1e-6, q_weight=wq, k_weight=wk)
outputs = split_qkv_rmsnorm_rope(*args, **options)
reference = split_qkv_rmsnorm_rope(*args, impl="torch_native", **options)
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
