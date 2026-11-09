# swiglu_oai_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.activation import swiglu_oai_quant

def swiglu_oai_quant(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None,
    *, impl="auto",
):
    ...
```

`auto` 默认选择 `triton`。所选实现不可用时会报错，不会自动切换到其他后端。

## 功能说明

对分半布局 `[gate | up]` 输入执行带裁剪的 SwiGLU-OAI，可选融合逐行对称 INT8 量化。最后一维的前半部分为 gate，后半部分为 up，两者不交错排列。

设最后一维为 $2D$，前导维度展平为 $S$ 行。对每个有效行计算：

$$
g_j=\min(x_j,\mathrm{limit}),\qquad
u_j=\max(-\mathrm{limit},\min(x_{D+j},\mathrm{limit})),
$$

$$
y_j=g_j\,\sigma(g_j\,\mathrm{alpha})\,(u_j+1).
$$

gate 只裁剪上界。激活计算使用 float32；关闭量化时，结果转换为 `x.dtype`。开启量化时，每行独立计算 scale。scale 非零时：

$$
s=\frac{\max_j|y_j|}{127},\qquad
q_j=\mathrm{sat}_{[-128,127]}\left(\mathrm{trunc}\left(
\mathrm{cast}_{\mathrm{x.dtype}}(y_j/s)\right)\right).
$$

归一化后的数值先转换为 `x.dtype`，再向零截断并饱和转换为 INT8。激活全零时，量化输出与 scale 均为零。使用 `q.float() * s` 可恢复近似的浮点结果；scale 不是其倒数。输入与激活中间值必须为有限数值。本算子用于推理，不保证梯度计算行为。

不传 `group_list` 时处理全部 $S$ 行；传入时只处理展平后的前 $R$ 行：

- `group_list_type=0`：不带起始哨兵的累积 token 数，`R = group_list[-1]`。
- `group_list_type=1`：各专家的 token 数，`R = group_list.sum()`。

允许专家 token 数为零，算子不会重排 token。计数必须非负，累积计数还必须单调不减。调用方须保证 `0 <= R <= S`，且计数与总数均可用有符号 int32 表示。Triton 不会将分组值复制到主机进行校验。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., 2D]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | strided Tensor，至少一维，`D > 0`；前导维度允许为空。通过 reshape/contiguous 拷贝支持非连续视图。 |
| `alpha` | `int` 或 `float` | — | — | 必填 | 有限的 sigmoid 缩放系数，可用 float32 表示。 |
| `limit` | `int` 或 `float` | — | — | 必填 | 有限、为正且可用 float32 表示的裁剪上界。 |
| `need_quant` | `bool` | — | — | `True` | 开启 INT8 输出及逐行 scale；为 True 时，Triton 要求 `D` 为 1536 的倍数。 |
| `group_list` | `torch.Tensor` 或 `None` | `[E]` | `torch.int32`、`torch.int64` | `None` | 可选非空一维 strided Tensor，与 x 同设备；描述有效前缀的计数或累积计数。 |
| `group_list_type` | `int` 或 `None` | — | — | `None` | 提供 group_list 时必填：`0` 为累积计数，`1` 为计数；Dense 模式忽略。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

分组编码或索引 dtype 不合法时抛出 `ValueError`；其余输入约束由调用方保证。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[..., D]` | need_quant 为 True 时为 `torch.int8`，否则为 x.dtype | 与 x 同设备的新输出；Grouped 模式仅展平后的前 R 行有效。 |
| `scale` | `torch.Tensor` | `[S]` | `torch.float32` | 与 x 同设备的新 scale 缓冲区；开启量化时仅前 R 个元素有效；need_quant=False 时所有元素均未定义。 |

返回 `(out, scale)`，不修改输入。Dense 模式下 `R=S`。不得使用未处理的行，也不得将其纳入精度对比。空输入返回空缓冲区，不启动 kernel。

## triton 后端

- 在昇腾 NPU 上运行。调用前须将当前 NPU 设为 `x.device`。
- 开启量化时，输出宽度 `D` 必须为 1536 的倍数；关闭量化时无此限制。
- 激活计算按整行处理，需要放入片上内存；满足 1536 的倍数要求不代表支持任意大的行宽。
- 分组计数在设备端读取。非连续输入可能产生拷贝。

## torch_native 后端

- 使用 PyTorch 运算作为参考实现，支持任意正整数输出宽度 `D`。
- 分组模式下读取有效行数可能同步设备。
- 两个后端的浮点计算差异可能使数值落在 INT8 截断边界的不同侧。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_oai_quant

x = torch.randn(17, 3072, device="npu", dtype=torch.float32)
out, scale = swiglu_oai_quant(x, 1.702, 7.0)
reference, scale_ref = swiglu_oai_quant(x, 1.702, 7.0, impl="torch_native")
torch.testing.assert_close(out, reference, rtol=0, atol=1)
torch.testing.assert_close(scale, scale_ref, rtol=5e-3, atol=0)

groups = torch.tensor([0, 2, 0, 3], device="npu", dtype=torch.int64)
grouped_out, _ = swiglu_oai_quant(
    x, 1.702, 7.0, need_quant=False, group_list=groups, group_list_type=1
)
valid_output = grouped_out[:5]  # 剩余行及所有 scale 均未定义。
```
