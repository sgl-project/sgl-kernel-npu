# split_qkvgate_gemma_rmsnorm_rope

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 指向相同的延迟导入入口，不自动回退。实现声明支持全部已知代际，但实际硬件执行仍待验证。

## 功能说明

解交错每个 head 的 Q/gate，使用 Gemma 的 1+weight 系数对 Q/K 按 head 做 RMSNorm，再对前 R 个特征做 NeoX RoPE。输入顺序是 [q0,gate0,q1,gate1,...,K,V]，并非全部 Q 后接全部 gate。Gate 和 V 直接复制，不归一化、不激活、不旋转。

$$
G(x,w)=x\,\mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)(1+w),\qquad Y_q=\mathrm{RoPE}_R(G(Q,w_q)),\quad Y_k=\mathrm{RoPE}_R(G(K,w_k)),\quad Y_v=V,\quad Y_g=\mathrm{gate}.
$$

每个 head 的前 R 个特征按 $y_i=x_i c_i+ho(x)_i s_i$ 旋转；NeoX 的 $ho$ 配对前后两半，GPT-J 配对相邻特征。尾部保留归一化后的值。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | Tensor | [N,2Q+2K] | floating | required | 连续的逐 head Q/gate 对，后接 K、V。 |
| `sin` | Tensor | [N,R] or [N,1,1,R] | floating | required | 连续的重复半表 sin，恰好 N*R 个元素。 |
| `cos` | Tensor | [N,R] or [N,1,1,R] | floating | required | 连续 cos，旋转维度 R 与 sin 相同。 |
| `q_hidden_size` | int | — | — | required | 正整数 Q，必须能被 K 整除。 |
| `kv_hidden_size` | int | — | — | required | 正整数 K，必须能被 head_dim 整除。 |
| `head_dim` | int | — | — | required | 每个 head 的正整数维度 D，不接受 bool。 |
| `rope_dim` | int | — | — | required | 正偶数 R <= D，必须等于 sin/cos 的最后维度。 |
| `eps` | int/float | — | — | required | 有限正 Python 标量，必须进行归一化。 |
| `q_weight` | Tensor | at least D values | floating | required | 原始 Gemma 偏移，乘数为 1+q_weight。 |
| `k_weight` | Tensor | at least D values | floating | required | 原始 Gemma 偏移，乘数为 1+k_weight。 |
| `impl` | str | — | — | `"auto"` | 仅限关键字参数：auto、triton、torch_native。 |

floating 指 float16、bfloat16 或 float32；参与运算的 tensor 须与输入同设备，归一化向量连续且可使用不同浮点 dtype。N 可以为零。精度比较使用有限输入。正整数选项不接受 bool。

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | 与输入同设备的新建连续输出。 |
| `k` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |
| `v` | Tensor | [N,K] | input.dtype | 与输入同设备的新建连续输出。 |
| `gate` | Tensor | [N,Q] | input.dtype | 与输入同设备的新建连续输出。 |

不修改输入、权重或 bias。两后端分别使用各自 norm/norm_utils.py 检查元信息，不共享校验依赖。Triton 额外要求 NPU；非法元信息在启动前抛出 AssertionError，检查不将设备数值回读主机。

## triton 后端

保留 split_qkvgate_gemma_rmsnorm_rope_kernel。沿用原 wrapper 的限制，head_dim 必须为 2 的幂。每列处理一个 KV head 及对应 Q/gate 组；必选模型用例使用 D=256、R=64 的部分 RoPE。空 token 输入不启动 kernel。

原 `sgl_kernel_npu.norm` 脚本保持不变；这里的输入检查属于迁移后的实现。

## torch_native 后端

将 Q/gate 视为 [N,Hq,2,D]，通过 unbind 分离。FP32 归一化仅对原始 weight 加一次 1；NeoX 旋转保留非旋转尾部，V/gate 复制为连续输出。

不调用 Triton kernel，可在 CPU 或 NPU 上独立使用。归约顺序、rsqrt 近似和乘加融合可能造成数值差异；按 dtype 容差通过不代表与框架分步路径逐位等价。

CPU 运算支持不改变包初始化要求：常规公开接口导入仍依赖 torch_npu、原生动态库及 ACL 硬件识别。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

x = torch.randn(4, 384, device='npu', dtype=torch.bfloat16)
sin = torch.zeros(4, 1, 1, 32, device='npu')
cos = torch.ones_like(sin)
wq = torch.zeros(64, device='npu')
wk = torch.zeros_like(wq)
args = (x, sin, cos, 128, 64, 64, 32, 1e-6, wq, wk)
outputs = split_qkvgate_gemma_rmsnorm_rope(*args)
reference = split_qkvgate_gemma_rmsnorm_rope(*args, impl="torch_native")
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
