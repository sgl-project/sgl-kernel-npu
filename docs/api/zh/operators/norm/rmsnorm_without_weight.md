# rmsnorm_without_weight

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
from sgl_kernel_npu.operators.norm import fused_rmsnorm_without_weight

def fused_rmsnorm_without_weight(x, eps, *, impl="auto"):
    ...
```

## 功能说明

对输入最后一维做无权重、无偏置的 RMS 归一化。设输入为 $X \in \mathbb{R}^{B\times L\times C}$，对每个 $(b,l)$ 行：

$$
v = \frac{1}{C}\sum_{j=0}^{C-1}X_{j}^{2}
$$

$$
Y_{j} = \frac{X_{j}}{\sqrt{v+\varepsilon}}
$$

结果转换回输入 dtype，不修改输入。

## 参数与返回值

下表描述推荐调用约定和当前实现要求。公开 operator 直接透传参数，不执行输入检查。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 待归一化输入。Triton 要求 NPU 上的连续三维 tensor，且 `C` 必须为 2 的幂。 |
| `eps` | `float` | — | — | 必填 | 加到均方值上的稳定项，建议为有限正数，如 `1e-6`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 当前选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, C]` | 与 `x` 相同 | 新建输出，与 `x` 位于同一设备；不修改输入。 |

附加说明：输入按有限值使用，NaN 的跨后端行为未定义。公开接口不检查输入，也不自动切换实现；包初始化仍要求 `torch_npu` 和原生动态库。

## triton 后端

- 使用 CANN Triton kernel，把 `B*L` 个 token 展平后按行归约，仅支持非量化输出。
- 以 `num_tokens` 为 key 在 `block_l ∈ {64, 32, 16, 8, 4, 2}` 上自动调优，grid 为 vector core 数。
- 归约维度直接使用 `tl.arange(0, C)`，因此要求 `C` 为 2 的幂；要求连续三维输入。
- kernel 内 `x*x` 沿用输入 dtype，归约精度可能低于 float32 参考（bf16 归约通常由 Triton 提升到 float32），fp16 输出的数值差异会更明显。

## torch_native 后端

- 使用 PyTorch 运算：先转 float32 求均方与 `rsqrt`，再转回输入 dtype；支持任意前置维度和非连续输入。
- 归约顺序及 `rsqrt` 实现可能与 Triton 略有差异，量化无关；一般浮点容差内一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rmsnorm_without_weight

x = torch.randn(2, 16, 2048, device="npu", dtype=torch.bfloat16)

print(list_impls(fused_rmsnorm_without_weight))  # ['auto', 'torch_native', 'triton']
y_default = fused_rmsnorm_without_weight(x, 1e-6)  # 默认 auto，使用 Triton
y_auto = fused_rmsnorm_without_weight(x, 1e-6, impl="auto")
y_triton = fused_rmsnorm_without_weight(x, 1e-6, impl="triton")
y_native = fused_rmsnorm_without_weight(x, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_auto, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-2, atol=1e-2)
```
