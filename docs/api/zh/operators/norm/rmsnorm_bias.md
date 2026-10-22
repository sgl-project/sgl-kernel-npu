# rmsnorm_bias

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。 不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.norm import rmsnorm_bias

def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    impl="auto",
):
    ...
```

## 功能说明

对二维输入的每一行进行 RMS 归一化，再乘权重、加偏置。设输入为 $X \in \mathbb{R}^{B\times H}$，权重为 $w$，偏置为 $b$：

$$
v_i = \frac{1}{H}\sum_{j=0}^{H-1}X_{ij}^{2}
$$

$$
Y_{ij} = \frac{X_{ij}}{\sqrt{v_i+\varepsilon}}w_j+b_j
$$

非量化分支将结果转换为输入 dtype。提供 `quant_scale` 时启用 int8 量化：

$$
Q_{ij} = \operatorname{cast}^{\mathrm{saturate}}_{\mathrm{int8}}
\left(Y_{ij}s_j+o_j\right)
$$

其中 $s$ 为 `quant_scale`，$o$ 为 `quant_offset`。torch_native 采用向零截断并饱和到 `[-128, 127]`，参考 [Triton Ascend 转换测试](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/unittest/pytest_ut/test_cast_full.py)。

## 参数与返回值

下表描述推荐调用约定和当前实现要求。公开 operator 直接透传参数，不执行输入检查。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 待归一化输入；按 `B > 0, H > 0` 调用。Triton 要求 NPU 上的连续二维 tensor。 |
| `norm_weight` | `torch.Tensor` | `[H]` | 浮点，建议与 input 相同 | 必填 | 每个通道的乘法权重；与 input 位于同一设备。Triton 按连续向量读取。 |
| `norm_bias` | `torch.Tensor` | `[H]` | 浮点，建议与 input 相同 | 必填 | 每个通道的加法偏置；与 input 位于同一设备。Triton 按连续向量读取。 |
| `eps` | `float` | — | — | 必填 | 加到均方值上的稳定项，建议为有限正数，如 `1e-6`。 |
| `quant_scale` | `torch.Tensor` 或 `None` | `[H]` 或 — | 浮点，推荐 `torch.float32` | `None` | 非 None 时启用量化，必须提供 quant_offset；与 input 同设备，Triton 要求连续。 |
| `quant_offset` | `torch.Tensor` 或 `None` | `[H]` 或 — | 浮点，推荐 `torch.float32` | `None` | 量化加法偏移；与 input 同设备，Triton 要求连续。scale 为 None 时忽略此参数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 当前选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | 非量化为 input.dtype；量化为 `torch.int8` | 新建输出，与 input 位于同一设备；不修改输入。 |

附加说明：输入及量化中间结果按有限值使用，NaN 的跨后端转换行为未定义。公开接口不检查输入，也不自动切换实现；包初始化仍要求 `torch_npu` 和原生动态库。

## triton 后端

- 使用 CANN Triton kernel，支持非量化和 int8 量化，要求连续输入。
- 量化使用固定 1024 元素切片，完整的 hidden size 支持范围待验证。

## torch_native 后端

- 使用 PyTorch 运算，支持非量化和 int8 量化；量化前保持 float32 中间结果，支持非连续输入。
- 对齐计算流程和量化分支条件；归约顺序及乘加融合可能造成数值差异，量化边界附近不保证逐位一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import rmsnorm_bias

x = torch.randn(4, 1024, device="npu", dtype=torch.float32)
weight = torch.randn(1024, device="npu", dtype=torch.float32)
bias = torch.randn(1024, device="npu", dtype=torch.float32)

print(list_impls(rmsnorm_bias))  # ['auto', 'torch_native', 'triton']
y_default = rmsnorm_bias(x, weight, bias, 1e-6)  # 默认 auto，使用 Triton
y_auto = rmsnorm_bias(x, weight, bias, 1e-6, impl="auto")
y_triton = rmsnorm_bias(x, weight, bias, 1e-6, impl="triton")
y_native = rmsnorm_bias(x, weight, bias, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_auto, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-4, atol=1e-4)
```
