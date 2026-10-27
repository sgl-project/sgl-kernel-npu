# fused_rope_qk_mqa

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

def fused_rope_qk_mqa(
    query, key, cos_sin, rotary_dim, is_neox_style, *, impl="auto"
):
    ...
```

`auto` 与 `triton` 注册到同一入口，默认使用 Triton，后端按需延迟导入。输入不支持或实现不可用时会报错，不自动切换后端。

## 功能说明

对 Query 和 Key 的前 `rotary_dim` 个通道应用旋转位置编码（RoPE）。两者的头数可以不同，Key 不会展开成 Query 的头数。

设旋转维度为 $R$，$m=R/2$。缓存的每行按 `[cos | sin]` 存储，$c_{t,j}=\mathrm{cos\_sin}_{t,j}$，$s_{t,j}=\mathrm{cos\_sin}_{t,j+m}$。对 Query 或 Key 的任意一个头，配对通道为：

$$
(a_j,b_j)=\begin{cases}
(j,j+m), & \text{is\_neox\_style=True},\\
(2j,2j+1), & \text{is\_neox\_style=False}.
\end{cases}
$$

$$
Y_{t,h,a_j}=X_{t,h,a_j}c_{t,j}-X_{t,h,b_j}s_{t,j},\qquad
Y_{t,h,b_j}=X_{t,h,a_j}s_{t,j}+X_{t,h,b_j}c_{t,j}.
$$

未旋转的尾部 $Y_{t,h,R:D}$ 直接复制输入。缓存第 `t` 行直接用于第 `t` 个 token，调用前必须已按位置选择并对齐缓存行；接口不接收位置索引、不执行 gather。超过 `T` 的额外缓存行被忽略。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[T, Hq, D]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | Query 输入；各维度均须大于 0。 |
| `key` | `torch.Tensor` | `[T, Hk, D]` | 与 query 相同 | 必填 | `Hk > 0`；与 query 共享 token 数、head dimension、设备及 dtype。不要求头数具有整除关系。 |
| `cos_sin` | `torch.Tensor` | `[N, rotary_dim]`，`N >= T` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 每行前半为 cos、后半为 sin；与 query 同设备，dtype 可以不同。 |
| `rotary_dim` | `int` | — | — | 必填 | Python 整数，排除 bool；须为正偶数且不大于 `D`。 |
| `is_neox_style` | `bool` | — | — | 必填 | Python bool；True 配对前后半段，False 配对相邻偶数、奇数通道。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

三个输入均要求 `torch.strided` 布局，支持非连续及零 stride 的只读视图。元信息不满足约束时通过断言报错。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `out_query` | `torch.Tensor` | `[T, Hq, D]` | query.dtype | 新建 Query 输出，与 query 同设备。 |
| `out_key` | `torch.Tensor` | `[T, Hk, D]` | key.dtype | 新建 Key 输出，与 key 同设备。 |

返回 `(out_query, out_key)`，不修改输入，未旋转尾部值保持不变。

## triton 后端

- 保留原 `norm/fused_rope_qk_mqa.py` 中 kernel 的计算方式；旧文件和原导入路径保持不变。
- 三个 Tensor 均须位于 NPU。`Hq`、`Hk`、`rotary_dim // 2` 必须为 2 的幂；`D - rotary_dim` 可为 0，否则也须为 2 的幂。这些要求来自未加 mask 的 `tl.arange`。
- 每个 program 处理一个 token，不要求 `T` 为 2 的幂。

## torch_native 后端

- 独立使用 PyTorch 数学运算，不调用 Triton。支持一般合法尺寸，不要求头数、半旋转维度或尾部长度为 2 的幂。
- 运算类型对齐 [Triton 类型提升规则](https://triton-lang.org/main/python-api/triton-semantics.html)：float16 与 bfloat16 混用时按 float16 运算；有 float32 时使用 float32；最终转回 query/key 的 dtype。低精度中间舍入及融合乘加会造成数值差异，两后端不保证逐位相等，应按 dtype 设置容差。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import fused_rope_qk_mqa

query = torch.randn(17, 8, 128, device="npu", dtype=torch.float32)
key = torch.randn(17, 1, 128, device="npu", dtype=torch.float32)
# 每行已经对应一个输入 token；实际调用中按位置构造或选择角度。
angles = torch.randn(17, 32, device="npu", dtype=torch.float32)
cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

q_out, k_out = fused_rope_qk_mqa(query, key, cos_sin, 64, True)
q_ref, k_ref = fused_rope_qk_mqa(
    query, key, cos_sin, 64, True, impl="torch_native"
)
torch.testing.assert_close(q_out, q_ref, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(k_out, k_ref, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(q_out[..., 64:], query[..., 64:], rtol=0, atol=0)
torch.testing.assert_close(k_out[..., 64:], key[..., 64:], rtol=0, atol=0)
```
