# add_rmsnorm_bias

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.norm import add_rmsnorm_bias

def add_rmsnorm_bias(
    input,
    residual,
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

先逐元素计算输入与残差之和，再对每一行做 RMS 归一化、乘权重、加偏置。设输入为 $X \in \mathbb{R}^{B\times H}$，残差为 $R \in \mathbb{R}^{B\times H}$，权重为 $w$，偏置为 $b$：

$$
A_{ij} = X_{ij}+R_{ij}
$$

$$
v_i = \frac{1}{H}\sum_{j=0}^{H-1}A_{ij}^{2}
$$

$$
Y_{ij} = \frac{A_{ij}}{\sqrt{v_i+\varepsilon}}w_j+b_j
$$

残差和在输入 dtype 下计算并作为第二个返回值返回；归一化在 float32 下进行。非量化分支将 $Y$ 转换为输入 dtype。提供 `quant_scale` 时启用 int8 量化：

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
| `residual` | `torch.Tensor` | `[B, H]` | 建议与 input 相同 | 必填 | 在归一化前加到 input 上的残差；与 input 位于同一设备。Triton 按连续二维 tensor 读取。 |
| `norm_weight` | `torch.Tensor` | `[H]` | 浮点，建议与 input 相同 | 必填 | 每个通道的乘法权重；与 input 位于同一设备。Triton 按连续向量读取。 |
| `norm_bias` | `torch.Tensor` | `[H]` | 浮点，建议与 input 相同 | 必填 | 每个通道的加法偏置；与 input 位于同一设备。Triton 按连续向量读取。 |
| `eps` | `float` | — | — | 必填 | 加到均方值上的稳定项，建议为有限正数，如 `1e-6`。 |
| `quant_scale` | `torch.Tensor` 或 `None` | `[H]` 或 — | 浮点，推荐 `torch.float32` | `None` | 非 None 时启用量化，必须提供 quant_offset；与 input 同设备，Triton 要求连续。 |
| `quant_offset` | `torch.Tensor` 或 `None` | `[H]` 或 — | 浮点，推荐 `torch.float32` | `None` | 量化加法偏移；与 input 同设备，Triton 要求连续。scale 为 None 时忽略此参数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 当前选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | 非量化为 input.dtype；量化为 `torch.int8` | 新建归一化输出，与 input 位于同一设备；不修改输入。 |
| `add_output` | `torch.Tensor` | `[B, H]` | input.dtype | 新建的残差和输出，与 input 位于同一设备；不修改输入。 |

附加说明：输入及量化中间结果按有限值使用，NaN 的跨后端转换行为未定义。公开接口不检查输入，也不自动切换实现；包初始化仍要求 `torch_npu` 和原生动态库。

## triton 后端

- 使用 CANN Triton kernel，支持非量化和 int8 量化，要求连续的二维输入。
- 按 `(n_rows, hidden_size, eps, BLOCK_SIZE, COL_BLOCK_SIZE, SCALE)` 缓存已编译 kernel，其中行数取 `min(B, vector core 数)`。
- 量化使用固定 2048 元素切片。完整的 hidden size 支持范围待验证。

## torch_native 后端

- 使用 PyTorch 运算，支持非量化和 int8 量化；残差和在输入 dtype 下计算，归一化前转到 float32，支持非连续输入。
- 对齐计算流程和量化分支条件；归约顺序及乘加融合可能造成数值差异，量化边界附近不保证逐位一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import add_rmsnorm_bias

x = torch.randn(4, 1024, device="npu", dtype=torch.float32)
residual = torch.randn(4, 1024, device="npu", dtype=torch.float32)
weight = torch.randn(1024, device="npu", dtype=torch.float32)
bias = torch.randn(1024, device="npu", dtype=torch.float32)

print(list_impls(add_rmsnorm_bias))  # ['auto', 'torch_native', 'triton']
y_default, add_default = add_rmsnorm_bias(x, residual, weight, bias, 1e-6)  # 默认 auto
y_auto, add_auto = add_rmsnorm_bias(x, residual, weight, bias, 1e-6, impl="auto")
y_triton, add_triton = add_rmsnorm_bias(x, residual, weight, bias, 1e-6, impl="triton")
y_native, add_native = add_rmsnorm_bias(
    x, residual, weight, bias, 1e-6, impl="torch_native"
)
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_auto, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(add_default, add_native, rtol=0, atol=0)
```
